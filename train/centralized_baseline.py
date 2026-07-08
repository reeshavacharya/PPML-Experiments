import os
import sys
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.swin_unetr import get_model
from utils.dataset_config import get_dataset_config, get_data_loader_module, resolve_manifest_path
from utils.metrics_utils import (
    ResourceMonitor, init_metrics_csv, append_metrics_csv,
    compute_dice, compute_hausdorff95, compute_cldice, log_system_info,
)
from utils.inference_utils import sliding_window_predict
from monai.losses import DiceCELoss

RESOURCE_COLS = ["Peak RAM (MB)", "Peak VRAM (MB)", "Peak GPU Util (%)"]


def run_training(
    dataset="fets",
    epochs=100,
    batch_size=1,
    lr=3e-4,
    weight_decay=1e-5,
    data_dir=None,
    save_checkpoint=True,
    trial=None,
    num_workers=2,
    seed=42,
    max_steps=0,
    resume=False,
):
    """Train Swin UNETR centrally on the given dataset. Returns best val mean
    Dice (averaged over the dataset's region_names).

    Args:
        dataset: "fets" or "cas" — resolved via utils/dataset_config.py.
        trial: Optuna Trial object for HPO pruning; None for standalone runs.
        save_checkpoint: Set False during HPO to skip model I/O overhead.
        seed: controls torch/numpy RNG for multi-seed repeats.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = get_dataset_config(dataset)
    data_loader_module = get_data_loader_module(dataset)
    region_names = cfg["region_names"]
    n_regions = len(region_names)

    log_system_info(client_id=f"Centralized Baseline ({dataset})")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if data_dir is None:
        data_dir = cfg["default_data_dir"]
    abs_data_dir, manifest_path = resolve_manifest_path(project_root, data_dir, cfg)

    train_loader, val_loader, _ = data_loader_module.create_data_loaders(
        abs_data_dir, manifest_path, client_id=0, batch_size=batch_size, num_workers=num_workers,
    )

    model = get_model(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"], feature_size=cfg["feature_size"])
    model.to(device)

    criterion = DiceCELoss(**cfg["loss_kwargs"])
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = os.path.join(project_root, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    suffix = "" if dataset == "fets" else f"_{dataset}"
    # Seed 42 keeps the original unsuffixed filenames (backward-compatible with
    # existing single-seed runs); other seeds get their own artifacts so a
    # multi-seed sweep (train/aggregate_seeds.py) doesn't overwrite them.
    seed_suffix = "" if seed == 42 else f"_seed{seed}"
    best_model_path = os.path.join(checkpoint_dir, f"swin_unetr_centralized{suffix}{seed_suffix}_best.pth")
    # Separate from best_model_path: saved every epoch (not just on Dice
    # improvement) with optimizer/scheduler state, so a preempted job can be
    # resubmitted — even to a different partition, since this lives on
    # shared storage — and pick back up mid-run instead of restarting.
    resume_path = os.path.join(checkpoint_dir, f"swin_unetr_centralized{suffix}{seed_suffix}_resume.pth")

    metrics_csv_path = os.path.join(project_root, "train", f"centralized_baseline{suffix}{seed_suffix}_metrics.csv")

    start_epoch = 1
    best_val_dice = 0.0
    if resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt["best_val_dice"]
        print(f"Resuming from {resume_path}: epoch {start_epoch}, best_val_dice={best_val_dice:.4f}", flush=True)
    else:
        init_metrics_csv(
            metrics_csv_path, region_names=region_names, extra_cols=RESOURCE_COLS,
            include_cldice=cfg["include_cldice"],
        )

    monitor = ResourceMonitor()

    for epoch in range(start_epoch, epochs + 1):
        current_lr = scheduler.get_last_lr()[0] if epoch > 1 else lr
        print(f"\nEpoch {epoch}/{epochs}  LR={current_lr:.6f}")

        monitor.start()

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_steps = 0
        total_train_steps = len(train_loader)
        log_interval = max(1, total_train_steps // 100)

        for i, batch in enumerate(train_loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            labels_for_loss = cfg["loss_label_fn"](labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_for_loss)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            if (i + 1) % log_interval == 0 or (i + 1) == total_train_steps:
                percent = int(100 * (i + 1) / total_train_steps)
                print(f"Epoch {epoch} Progress: {percent}% ({i + 1}/{total_train_steps}) - Loss: {loss.item():.4f}")
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
        dice_lists = [[] for _ in range(n_regions)]
        hd95_lists = [[] for _ in range(n_regions)]
        cldice_lists = [[] for _ in range(n_regions)] if cfg["include_cldice"] else None

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                labels_for_loss = cfg["loss_label_fn"](labels)

                if cfg["sliding_window"]:
                    outputs = sliding_window_predict(model, images)
                else:
                    outputs = model(images)
                loss = criterion(outputs, labels_for_loss)

                val_loss += loss.item()
                val_steps += 1

                pred_bin = cfg["metric_pred_fn"](outputs)
                label_bin = cfg["metric_label_fn"](labels)

                dice = compute_dice(pred_bin, label_bin).mean(dim=0).cpu().numpy()
                for c in range(n_regions):
                    dice_lists[c].append(dice[c])

                hd95 = compute_hausdorff95(pred_bin[0].cpu().numpy(), label_bin[0].cpu().numpy())
                for c in range(n_regions):
                    if not np.isnan(hd95[c]):
                        hd95_lists[c].append(hd95[c])

                if cfg["include_cldice"]:
                    pred_np = pred_bin[0].cpu().numpy()
                    label_np = label_bin[0].cpu().numpy()
                    for c in range(n_regions):
                        cldice_lists[c].append(compute_cldice(pred_np[c], label_np[c]))

        monitor.stop()

        avg_val_loss = val_loss / val_steps
        avg_dice = [np.mean(d) if d else 0 for d in dice_lists]
        avg_hd95 = [np.mean(h) if h else 0 for h in hd95_lists]
        mean_dice_overall = sum(avg_dice) / n_regions
        mean_hd95_overall = sum(avg_hd95) / n_regions

        region_str = " | ".join(f"Dice {r.upper()}: {avg_dice[i]:.4f}" for i, r in enumerate(region_names))
        print(
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Mean Dice: {mean_dice_overall:.4f}\n"
            f"{region_str}\n"
            f"Peak RAM: {monitor.peak_ram:.0f} MB | Peak VRAM: {monitor.peak_vram:.0f} MB | Peak GPU Util: {monitor.peak_gpu_util:.1f}%"
        )

        row = [epoch, f"{avg_train_loss:.4f}", train_steps, f"{avg_val_loss:.4f}"]
        row += [f"{d:.4f}" for d in avg_dice] + [f"{mean_dice_overall:.4f}"]
        row += [f"{h:.2f}" for h in avg_hd95] + [f"{mean_hd95_overall:.2f}"]
        if cfg["include_cldice"]:
            avg_cldice = [np.mean(c) if c else 0 for c in cldice_lists]
            mean_cldice_overall = sum(avg_cldice) / n_regions
            row += [f"{c:.4f}" for c in avg_cldice] + [f"{mean_cldice_overall:.4f}"]
        row += [""] * (3 * n_regions)  # sensitivity/specificity/precision — computed at final test time only
        row += [f"{monitor.peak_ram:.1f}", f"{monitor.peak_vram:.1f}", f"{monitor.peak_gpu_util:.1f}"]
        append_metrics_csv(metrics_csv_path, row)

        if mean_dice_overall > best_val_dice:
            print(f"Validation Dice improved from {best_val_dice:.4f} to {mean_dice_overall:.4f}. Saving model...")
            best_val_dice = mean_dice_overall
            if save_checkpoint:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mean_dice": mean_dice_overall,
                }, best_model_path)

        if save_checkpoint:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_dice": best_val_dice,
            }, resume_path)

        # Optuna pruning: report intermediate value and check if trial should stop
        if trial is not None:
            import optuna
            trial.report(mean_dice_overall, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_dice


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fets", choices=["fets", "cas"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint/swin_unetr_centralized*_resume.pth if present")
    a = parser.parse_args()
    run_training(
        dataset=a.dataset, epochs=a.epochs, batch_size=1, lr=a.lr, weight_decay=a.weight_decay,
        data_dir=a.data_dir, num_workers=a.num_workers, seed=a.seed, resume=a.resume,
    )
