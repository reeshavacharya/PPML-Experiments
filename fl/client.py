import argparse
import fcntl
import os
import subprocess
import sys
import time
import traceback
import torch
import flwr as fl
from collections import OrderedDict
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.losses import DiceCELoss
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.swin_unetr import get_model
from utils.dataset_config import get_dataset_config, get_data_loader_module, resolve_manifest_path
from utils.metrics_utils import (
    ResourceMonitor,
    init_metrics_csv, append_metrics_csv,
    compute_dice, compute_hausdorff95, compute_cldice,
    compute_sensitivity, compute_specificity, compute_precision_metric,
    log_system_info,
)
from utils.inference_utils import sliding_window_predict

import platform as _platform

def _physical_gpu_id():
    """Real GPU identity, not CUDA_VISIBLE_DEVICES. SLURM remaps each job's
    assigned GPU to a job-local index (usually "0"), so two unrelated jobs
    co-located on the same node both see CUDA_VISIBLE_DEVICES=0 even when
    holding two different physical GPUs — that false collision serialized
    unrelated jobs onto one lock file and caused multi-hour stalls. Querying
    nvidia-smi from inside the job's cgroup still reports the true UUID of
    whichever physical card is bound to it, so the lock only shares across
    jobs that are actually on the same hardware.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "all")

# Lock is per physical GPU device so two clients on different GPUs of the
# same node can run in parallel, and two unrelated jobs sharing a node don't
# falsely collide (see _physical_gpu_id).
_gpu_id = _physical_gpu_id()
GPU_LOCK_PATH = os.path.join(project_root, "fl", f".gpu_lock_{_platform.node()}_{_gpu_id}")

RESOURCE_COLS = ["Peak RAM (MB)", "Peak VRAM (MB)", "Peak GPU Util (%)"]

DEFAULT_LR = 3e-4
DEFAULT_WD = 1e-5


class SwinUNETRClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, client_id, metrics_csv, cfg,
                 num_rounds=100, round_offset=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.client_id = client_id
        self.cfg = cfg
        self.region_names = cfg["region_names"]
        self.criterion = DiceCELoss(**cfg["loss_kwargs"])
        self._current_lr = DEFAULT_LR
        self._current_wd = DEFAULT_WD
        self.optimizer = AdamW(self.model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WD)
        # Cosine decay over num_rounds rounds, mirroring the centralized baseline scheduler.
        # Fast-forward to round_offset so LR is correct when resuming mid-training.
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

        # Allow server to override lr/weight_decay/max_steps per round (used during HPO)
        lr = float(config.get("lr", self._current_lr))
        wd = float(config.get("weight_decay", self._current_wd))
        max_steps = int(config.get("max_steps", 0))
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
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            total_train_steps = len(self.train_loader)
            log_interval = max(1, total_train_steps // 10)

            for i, batch in enumerate(self.train_loader):
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                labels_for_loss = self.cfg["loss_label_fn"](labels)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels_for_loss)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_steps += 1

                if (i + 1) % log_interval == 0 or (i + 1) == total_train_steps:
                    pct = int(100 * (i + 1) / total_train_steps)
                    print(f"[Client {self.client_id}, Round {self.round}] {pct}% ({i+1}/{total_train_steps}) Loss: {loss.item():.4f}")
                    sys.stdout.flush()

                if max_steps and train_steps >= max_steps:
                    print(f"[Client {self.client_id}, Round {self.round}] Early stop at {train_steps} steps (max_steps={max_steps})", flush=True)
                    break
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

        n_regions = len(self.region_names)
        row = [self.round, f"{avg_train_loss:.4f}", train_steps, ""]
        row += [""] * (n_regions + 1)  # Dice per region + mean
        row += [""] * (n_regions + 1)  # HD95 per region + mean
        if self.cfg["include_cldice"]:
            row += [""] * (n_regions + 1)  # clDice per region + mean
        row += [""] * (3 * n_regions)  # sensitivity/specificity/precision
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
        n_regions = len(self.region_names)

        self._acquire_gpu()
        try:
            self.model.eval()
            val_loss = 0.0
            val_steps = 0
            dice_lists = [[] for _ in range(n_regions)]
            hd95_lists = [[] for _ in range(n_regions)]
            sens_lists = [[] for _ in range(n_regions)]
            spec_lists = [[] for _ in range(n_regions)]
            prec_lists = [[] for _ in range(n_regions)]
            cldice_lists = [[] for _ in range(n_regions)] if self.cfg["include_cldice"] else None

            with torch.no_grad():
                for batch in self.val_loader:
                    images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                    labels_for_loss = self.cfg["loss_label_fn"](labels)

                    outputs = sliding_window_predict(self.model, images) if self.cfg["sliding_window"] else self.model(images)
                    loss = self.criterion(outputs, labels_for_loss)

                    val_loss += loss.item()
                    val_steps += 1

                    pred_bin = self.cfg["metric_pred_fn"](outputs)
                    label_bin = self.cfg["metric_label_fn"](labels)

                    dice = compute_dice(pred_bin, label_bin).mean(dim=0).cpu().numpy()
                    sens = compute_sensitivity(pred_bin, label_bin).mean(dim=0).cpu().numpy()
                    spec = compute_specificity(pred_bin, label_bin).mean(dim=0).cpu().numpy()
                    prec = compute_precision_metric(pred_bin, label_bin).mean(dim=0).cpu().numpy()
                    hd95 = compute_hausdorff95(pred_bin[0].cpu().numpy(), label_bin[0].cpu().numpy())

                    pred_np = pred_bin[0].cpu().numpy()
                    label_np = label_bin[0].cpu().numpy()
                    for c in range(n_regions):
                        dice_lists[c].append(dice[c])
                        sens_lists[c].append(sens[c])
                        spec_lists[c].append(spec[c])
                        prec_lists[c].append(prec[c])
                        if not np.isnan(hd95[c]):
                            hd95_lists[c].append(hd95[c])
                        if self.cfg["include_cldice"]:
                            cldice_lists[c].append(compute_cldice(pred_np[c], label_np[c]))
        finally:
            self._release_gpu()

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        avg_dice = [np.mean(d) if d else 0 for d in dice_lists]
        avg_hd95 = [np.mean(h) if h else 0 for h in hd95_lists]
        avg_sens = [np.mean(s) if s else 0 for s in sens_lists]
        avg_spec = [np.mean(s) if s else 0 for s in spec_lists]
        avg_prec = [np.mean(p) if p else 0 for p in prec_lists]
        mean_dice = sum(avg_dice) / n_regions
        mean_hd95 = sum(avg_hd95) / n_regions

        print(f"[Client {self.client_id}, Round {self.round}] Val Loss: {avg_val_loss:.4f} | Dice: {mean_dice:.4f} | HD95: {mean_hd95:.2f}", flush=True)

        row = [f"{self.round}_val", "", "", f"{avg_val_loss:.4f}"]
        row += [f"{d:.4f}" for d in avg_dice] + [f"{mean_dice:.4f}"]
        row += [f"{h:.2f}" for h in avg_hd95] + [f"{mean_hd95:.2f}"]
        if self.cfg["include_cldice"]:
            avg_cldice = [np.mean(c) if c else 0 for c in cldice_lists]
            row += [f"{c:.4f}" for c in avg_cldice] + [f"{sum(avg_cldice) / n_regions:.4f}"]
        row += [f"{s:.4f}" for s in avg_sens]
        row += [f"{s:.4f}" for s in avg_spec]
        row += [f"{p:.4f}" for p in avg_prec]
        row += ["", "", ""]  # resource cols not tracked during evaluate() — only fit() runs the GPU-heavy pass
        append_metrics_csv(self.metrics_csv, row)

        return float(avg_val_loss), len(self.val_loader.dataset), {
            "val_dice": float(mean_dice),
            "institution_id": self.client_id,
        }


def main():
    parser = argparse.ArgumentParser(description="Flower Client for SwinUNETR FL experiments")
    parser.add_argument("--dataset", type=str, default="fets", choices=["fets", "cas"])
    parser.add_argument("--client_id", type=int, required=True, help="Client ID (1-N; N and its meaning depend on --dataset)")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080", help="Address of the FL server")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset (default: per-dataset config)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--num_rounds", type=int, default=100, help="Total FL rounds (must match server)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run (append to existing metrics CSV)")
    parser.add_argument("--seed", type=int, default=42, help="Controls local torch RNG for multi-seed repeats")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    log_system_info(client_id=args.client_id)

    cfg = get_dataset_config(args.dataset)
    data_loader_module = get_data_loader_module(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Client {args.client_id} using device: {device}")

    data_dir = args.data_dir if args.data_dir is not None else cfg["default_data_dir"]
    abs_data_dir, manifest_path = resolve_manifest_path(project_root, data_dir, cfg)

    train_loader, val_loader, _ = data_loader_module.create_data_loaders(
        abs_data_dir, manifest_path, client_id=args.client_id, batch_size=1, num_workers=args.num_workers,
    )

    model = get_model(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"], feature_size=cfg["feature_size"])

    suffix = "" if args.dataset == "fets" else f"_{args.dataset}"
    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    metrics_csv = os.path.join(project_root, "fl", f"client{suffix}{seed_suffix}_{args.client_id}_metrics.csv")
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
        init_metrics_csv(
            metrics_csv, region_names=cfg["region_names"], extra_cols=RESOURCE_COLS,
            include_cldice=cfg["include_cldice"],
        )

    client = SwinUNETRClient(
        model, train_loader, val_loader, device, args.client_id, metrics_csv, cfg,
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
