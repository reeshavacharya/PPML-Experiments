import argparse
import fcntl
import os
import subprocess
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
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

import platform as _platform

def _physical_gpu_id():
    """Real GPU identity, not CUDA_VISIBLE_DEVICES — see fl/client.py's
    _physical_gpu_id for why: SLURM remaps each job's assigned GPU to a
    job-local index (usually "0"), so two unrelated jobs co-located on the
    same node both see CUDA_VISIBLE_DEVICES=0 even on different physical
    GPUs, falsely serializing them onto one lock file.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "all")

# Lock is per physical GPU device, not per dataset — shared with
# fl/client.py's and fl/client_isic.py's clients so two unrelated jobs never
# step on the same physical GPU if actually co-located on it.
_gpu_id = _physical_gpu_id()
GPU_LOCK_PATH = os.path.join(project_root, "fl", f".gpu_lock_{_platform.node()}_{_gpu_id}")

RESOURCE_COLS = ["Peak RAM (MB)", "Peak VRAM (MB)", "Peak GPU Util (%)"]

DEFAULT_LR = 3e-5
DEFAULT_WD = 1e-5


def build_criterion(loss_name, pos_weight, num_classes):
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_name == "focal":
        from monai.losses import FocalLoss
        return FocalLoss(gamma=2.0, weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss '{loss_name}'. Choices: bce, focal")


class NIHClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, client_id, metrics_csv, criterion,
                 num_rounds=50, round_offset=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.client_id = client_id
        self.criterion = criterion
        self._current_lr = DEFAULT_LR
        self._current_wd = DEFAULT_WD
        self.optimizer = AdamW(self.model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WD)
        # Cosine decay over num_rounds rounds, mirroring the centralized ViT scheduler.
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_rounds)
        for _ in range(round_offset):
            self.scheduler.step()
        self.monitor = ResourceMonitor()
        self.metrics_csv = metrics_csv
        self.round = round_offset

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def _acquire_gpu(self):
        self._lock_fh = open(GPU_LOCK_PATH, "w")
        fcntl.flock(self._lock_fh, fcntl.LOCK_EX)
        self.model.to(self.device)

    def _release_gpu(self):
        self.model.to("cpu")
        torch.cuda.empty_cache()
        fcntl.flock(self._lock_fh, fcntl.LOCK_UN)
        self._lock_fh.close()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.round += 1

        # Allow server to override lr/weight_decay per round (used during HPO)
        lr = float(config.get("lr", self._current_lr))
        wd = float(config.get("weight_decay", self._current_wd))
        if lr != self._current_lr or wd != self._current_wd:
            self._current_lr = lr
            self._current_wd = wd
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
            print(f"[Client {self.client_id}, Round {self.round}] Optimizer updated: lr={lr:.2e} wd={wd:.2e}", flush=True)

        current_lr = self.scheduler.get_last_lr()[0]
        print(f"[Client {self.client_id}, Round {self.round}] LR={current_lr:.6f}", flush=True)

        self.monitor.start()
        self._acquire_gpu()
        try:
            # 1 local epoch per round, matching the IID FedAvg spec.
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            total_train_steps = len(self.train_loader)
            log_interval = max(1, total_train_steps // 10)

            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_steps += 1

                if (i + 1) % log_interval == 0 or (i + 1) == total_train_steps:
                    pct = int(100 * (i + 1) / total_train_steps)
                    print(f"[Client {self.client_id}, Round {self.round}] {pct}% ({i+1}/{total_train_steps}) Loss: {loss.item():.4f}")
                    sys.stdout.flush()
        finally:
            self._release_gpu()

        self.monitor.stop()
        self.scheduler.step()

        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
        print(
            f"[Client {self.client_id}, Round {self.round}] Train Loss: {avg_train_loss:.4f} | "
            f"Peak RAM: {self.monitor.peak_ram:.0f} MB | "
            f"Peak VRAM: {self.monitor.peak_vram:.0f} MB | "
            f"Peak GPU Util: {self.monitor.peak_gpu_util:.1f}%",
            flush=True,
        )

        row = [self.round, f"{avg_train_loss:.4f}", train_steps, ""]
        row += [""] * (2 + 2 * NUM_CLASSES)  # mean auc, macro f1, per-class auc, per-class sensitivity
        row += [
            f"{self.monitor.peak_ram:.1f}",
            f"{self.monitor.peak_vram:.1f}",
            f"{self.monitor.peak_gpu_util:.1f}",
        ]
        append_metrics_csv(self.metrics_csv, row)

        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "train_loss": avg_train_loss,
            "sgd_steps": train_steps,
            "institution_id": self.client_id,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self._acquire_gpu()
        try:
            self.model.eval()
            val_loss = 0.0
            val_steps = 0
            all_labels, all_probs = [], []

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    val_steps += 1

                    probs = torch.sigmoid(outputs)
                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
        finally:
            self._release_gpu()

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0

        if all_labels:
            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
            val_mean_auc = compute_mean_auc_roc(all_labels, all_probs, NUM_CLASSES)
            val_macro_f1 = compute_macro_f1(all_labels, all_probs)
            val_per_class_auc = compute_per_class_auc(all_labels, all_probs, NUM_CLASSES)
            val_per_class_sens = compute_per_class_sensitivity(all_labels, all_probs, NUM_CLASSES)
        else:
            val_mean_auc = val_macro_f1 = 0.0
            val_per_class_auc = np.zeros(NUM_CLASSES)
            val_per_class_sens = np.zeros(NUM_CLASSES)

        print(
            f"[Client {self.client_id}, Round {self.round}] Val Loss: {avg_val_loss:.4f} | "
            f"Mean AUC: {val_mean_auc:.4f} | Macro F1: {val_macro_f1:.4f}", flush=True,
        )

        row = [f"{self.round}_val", "", "", f"{avg_val_loss:.4f}"]
        row += [f"{val_mean_auc:.4f}", f"{val_macro_f1:.4f}"]
        row += [f"{a:.4f}" for a in val_per_class_auc]
        row += [f"{s:.4f}" for s in val_per_class_sens]
        row += ["", "", ""]  # resource cols not tracked during evaluate() — only fit() runs the GPU-heavy pass
        append_metrics_csv(self.metrics_csv, row)

        return float(avg_val_loss), len(self.val_loader.dataset), {
            "val_mean_auc": float(val_mean_auc),
            "institution_id": self.client_id,
        }


def main():
    parser = argparse.ArgumentParser(description="Flower Client for ViT-B/16 NIH ChestX-ray14 FL experiments")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID (1-5, one per IID shard)")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8082", help="Address of the FL server")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset (default: data/NIH-Chest)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--num_rounds", type=int, default=50, help="Total FL rounds (must match server)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run (append to existing metrics CSV)")
    parser.add_argument("--seed", type=int, default=42, help="Controls local torch RNG for multi-seed repeats")
    parser.add_argument("--train_subsample_frac", type=float, default=None,
                         help="Subsample this client's own training split, e.g. 0.4 (HPO speed-up; never use for real runs)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    log_system_info(client_id=args.client_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Client {args.client_id} using device: {device}")

    data_dir = args.data_dir if args.data_dir is not None else DEFAULT_DATA_DIR
    abs_data_dir = os.path.join(project_root, data_dir)
    manifest_path = os.path.join(abs_data_dir, DEFAULT_MANIFEST)

    train_loader, val_loader, _ = create_data_loaders(
        abs_data_dir, manifest_path, client_id=args.client_id, batch_size=args.batch_size, num_workers=args.num_workers,
        train_subsample_frac=args.train_subsample_frac,
    )

    pos_weight = compute_pos_weight(manifest_path).to(device)
    criterion = build_criterion(args.loss, pos_weight, NUM_CLASSES)

    model = get_model(num_classes=NUM_CLASSES, pretrained=True)

    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    metrics_csv = os.path.join(project_root, "fl", f"client_nih{seed_suffix}_{args.client_id}_metrics.csv")
    round_offset = 0

    if args.resume and os.path.exists(metrics_csv):
        import csv
        with open(metrics_csv, "r") as f:
            rows = list(csv.reader(f))
        for row in reversed(rows[1:]):
            entry = row[0].replace("_val", "")
            try:
                round_offset = int(entry)
                break
            except ValueError:
                continue
        print(f"[Client {args.client_id}] Resuming from round {round_offset}, appending to {metrics_csv}", flush=True)
    else:
        init_multilabel_metrics_csv(metrics_csv, class_names=CLASS_NAMES, extra_cols=RESOURCE_COLS)

    client = NIHClient(
        model, train_loader, val_loader, device, args.client_id, metrics_csv, criterion,
        num_rounds=args.num_rounds, round_offset=round_offset,
    )

    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Client {args.client_id}] Connection attempt {attempt}/{max_retries}", flush=True)
            fl.client.start_client(server_address=args.server_address, client=client.to_client())
            break
        except Exception:
            traceback.print_exc()
            if attempt == max_retries:
                print(f"[Client {args.client_id}] All {max_retries} attempts exhausted. Exiting.", flush=True)
                sys.exit(1)
            wait = min(30 * attempt, 300)
            print(f"[Client {args.client_id}] Disconnected. Retrying in {wait}s...", flush=True)
            time.sleep(wait)


if __name__ == "__main__":
    main()
