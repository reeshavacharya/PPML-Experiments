import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.vit_classifier import get_model
from data_loader.nih_chest_dataset import (
    create_data_loaders, compute_pos_weight, CLASS_NAMES, NUM_CLASSES,
    DEFAULT_DATA_DIR, DEFAULT_MANIFEST,
)
from utils.metrics_utils import (
    ResourceMonitor, init_multilabel_metrics_csv, append_metrics_csv, log_system_info,
)
from utils.multilabel_metrics import (
    compute_mean_auc_roc, compute_macro_f1, compute_per_class_auc, compute_per_class_sensitivity,
)

RESOURCE_COLS = ["Peak RAM (MB)", "Peak VRAM (MB)", "Peak GPU Util (%)"]


def build_criterion(loss_name, pos_weight, num_classes):
    """"bce" (default, positive-weighted BCEWithLogitsLoss — matches the
    ChestX-ray14 paper's own finding that weighting improves AUC on rare
    classes) or "focal" (MONAI FocalLoss, sigmoid/multi-label mode) —
    selectable via --loss."""
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_name == "focal":
        from monai.losses import FocalLoss
        return FocalLoss(gamma=2.0, weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss '{loss_name}'. Choices: bce, focal")


def run_training(
    dataset="nih",
    epochs=50,
    batch_size=32,
    lr=3e-5,
    weight_decay=1e-5,
    loss="bce",
    data_dir=None,
    save_checkpoint=True,
    trial=None,
    num_workers=4,
    seed=42,
    train_subsample_frac=None,
    max_steps=0,
):
    """Train ViT-B/16 centrally on NIH ChestX-ray14 (pooled). Returns best
    val mean AUC-ROC — the model-selection metric per the project spec.

    Args:
        trial: Optuna Trial object for HPO pruning; None for standalone runs.
        save_checkpoint: Set False during HPO to skip model I/O overhead.
        seed: controls torch/numpy RNG for multi-seed repeats.
        train_subsample_frac: subsample only the training split (HPO speed-up).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_system_info(client_id=f"Centralized Baseline ({dataset})")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    abs_data_dir = os.path.join(project_root, data_dir)
    manifest_path = os.path.join(abs_data_dir, DEFAULT_MANIFEST)

    train_loader, val_loader, _ = create_data_loaders(
        abs_data_dir, manifest_path, client_id=0, batch_size=batch_size, num_workers=num_workers,
        train_subsample_frac=train_subsample_frac,
    )

    pos_weight = compute_pos_weight(manifest_path).to(device)
    criterion = build_criterion(loss, pos_weight, NUM_CLASSES)

    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = os.path.join(project_root, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    seed_suffix = "" if seed == 42 else f"_seed{seed}"
    best_model_path = os.path.join(checkpoint_dir, f"vit_nih_centralized{seed_suffix}_best.pth")

    metrics_csv_path = os.path.join(project_root, "train", f"centralized_nih{seed_suffix}_metrics.csv")
    init_multilabel_metrics_csv(metrics_csv_path, class_names=CLASS_NAMES, extra_cols=RESOURCE_COLS)

    monitor = ResourceMonitor()
    best_val_auc = 0.0

    for epoch in range(1, epochs + 1):
        current_lr = scheduler.get_last_lr()[0] if epoch > 1 else lr
        print(f"\nEpoch {epoch}/{epochs}  LR={current_lr:.6f}")

        monitor.start()

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_steps = 0
        total_train_steps = len(train_loader)
        log_interval = max(1, total_train_steps // 100)

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss_val = criterion(outputs, labels)

            loss_val.backward()
            optimizer.step()

            train_loss += loss_val.item()
            train_steps += 1

            if (i + 1) % log_interval == 0 or (i + 1) == total_train_steps:
                percent = int(100 * (i + 1) / total_train_steps)
                print(f"Epoch {epoch} Progress: {percent}% ({i + 1}/{total_train_steps}) - Loss: {loss_val.item():.4f}")
                sys.stdout.flush()

            if max_steps and train_steps >= max_steps:
                print(f"Epoch {epoch} early stop at {train_steps} steps (max_steps={max_steps})")
                break

        avg_train_loss = train_loss / train_steps
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss_val = criterion(outputs, labels)

                val_loss += loss_val.item()
                val_steps += 1

                probs = torch.sigmoid(outputs)
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        monitor.stop()

        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        avg_val_loss = val_loss / val_steps
        val_mean_auc = compute_mean_auc_roc(all_labels, all_probs, NUM_CLASSES)
        val_macro_f1 = compute_macro_f1(all_labels, all_probs)
        val_per_class_auc = compute_per_class_auc(all_labels, all_probs, NUM_CLASSES)
        val_per_class_sens = compute_per_class_sensitivity(all_labels, all_probs, NUM_CLASSES)

        print(
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Mean AUC: {val_mean_auc:.4f} | Val Macro F1: {val_macro_f1:.4f}\n"
            f"Peak RAM: {monitor.peak_ram:.0f} MB | Peak VRAM: {monitor.peak_vram:.0f} MB | Peak GPU Util: {monitor.peak_gpu_util:.1f}%"
        )

        row = [epoch, f"{avg_train_loss:.4f}", train_steps, f"{avg_val_loss:.4f}"]
        row += [f"{val_mean_auc:.4f}", f"{val_macro_f1:.4f}"]
        row += [f"{a:.4f}" for a in val_per_class_auc]
        row += [f"{s:.4f}" for s in val_per_class_sens]
        row += [f"{monitor.peak_ram:.1f}", f"{monitor.peak_vram:.1f}", f"{monitor.peak_gpu_util:.1f}"]
        append_metrics_csv(metrics_csv_path, row)

        if val_mean_auc > best_val_auc:
            print(f"Validation mean AUC-ROC improved from {best_val_auc:.4f} to {val_mean_auc:.4f}. Saving model...")
            best_val_auc = val_mean_auc
            if save_checkpoint:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mean_auc_roc": val_mean_auc,
                }, best_model_path)

        if trial is not None:
            import optuna
            trial.report(val_mean_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_auc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nih", choices=["nih"])
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_subsample_frac", type=float, default=None,
                         help="Subsample only the training split, e.g. 0.4 (HPO speed-up; never use for real runs)")
    parser.add_argument("--max_steps", type=int, default=0,
                         help="Limit training to N gradient steps per epoch (0=unlimited; HPO speedup)")
    a = parser.parse_args()
    run_training(
        dataset=a.dataset, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, loss=a.loss, data_dir=a.data_dir,
        num_workers=a.num_workers, seed=a.seed,
        train_subsample_frac=a.train_subsample_frac,
        max_steps=a.max_steps,
    )
