import argparse
import fcntl
import os
import sys
import time
import traceback
import torch
import flwr as fl
from collections import OrderedDict
from torch.optim import AdamW
from monai.losses import DiceCELoss
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_loader.fets_dataset import create_data_loaders
from models.swin_unetr import get_model
from utils.metrics_utils import (
    init_metrics_csv, append_metrics_csv,
    convert_to_brats_regions, compute_dice, compute_hausdorff95,
    compute_sensitivity, compute_specificity, compute_precision_metric,
    log_system_info,
)

import platform as _platform
GPU_LOCK_PATH = os.path.join(project_root, "fl", f".gpu_lock_{_platform.node()}")


class FeTSClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, client_id, metrics_csv):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.client_id = client_id
        self.criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        self.optimizer = AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.metrics_csv = metrics_csv
        self.round = 0

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

        self._acquire_gpu()
        try:
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            total_train_steps = len(self.train_loader)
            log_interval = max(1, total_train_steps // 10)

            for i, batch in enumerate(self.train_loader):
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                labels_converted = convert_to_brats_regions(labels)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels_converted)

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

        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
        print(f"[Client {self.client_id}, Round {self.round}] Train Loss: {avg_train_loss:.4f}", flush=True)

        append_metrics_csv(self.metrics_csv, [
            self.round, f"{avg_train_loss:.4f}", train_steps, "",
            "", "", "", "",
            "", "", "", "",
            "", "", "",
            "", "", "",
            "", "", "",
        ])

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
            dice_lists = [[], [], []]
            hd95_lists = [[], [], []]
            sens_lists = [[], [], []]
            spec_lists = [[], [], []]
            prec_lists = [[], [], []]

            with torch.no_grad():
                for batch in self.val_loader:
                    images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                    labels_converted = convert_to_brats_regions(labels)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels_converted)

                    val_loss += loss.item()
                    val_steps += 1

                    outputs_binary = (torch.sigmoid(outputs) > 0.5).float()

                    dice = compute_dice(outputs_binary, labels_converted).mean(dim=0).cpu().numpy()
                    sens = compute_sensitivity(outputs_binary, labels_converted).mean(dim=0).cpu().numpy()
                    spec = compute_specificity(outputs_binary, labels_converted).mean(dim=0).cpu().numpy()
                    prec = compute_precision_metric(outputs_binary, labels_converted).mean(dim=0).cpu().numpy()
                    hd95 = compute_hausdorff95(
                        outputs_binary[0].cpu().numpy(),
                        labels_converted[0].cpu().numpy(),
                    )

                    for c in range(3):
                        dice_lists[c].append(dice[c])
                        sens_lists[c].append(sens[c])
                        spec_lists[c].append(spec[c])
                        prec_lists[c].append(prec[c])
                        if not np.isnan(hd95[c]):
                            hd95_lists[c].append(hd95[c])
        finally:
            self._release_gpu()

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        avg_dice = [np.mean(d) if d else 0 for d in dice_lists]
        avg_hd95 = [np.mean(h) if h else 0 for h in hd95_lists]
        avg_sens = [np.mean(s) if s else 0 for s in sens_lists]
        avg_spec = [np.mean(s) if s else 0 for s in spec_lists]
        avg_prec = [np.mean(p) if p else 0 for p in prec_lists]
        mean_dice = sum(avg_dice) / 3.0
        mean_hd95 = sum(avg_hd95) / 3.0

        print(f"[Client {self.client_id}, Round {self.round}] Val Loss: {avg_val_loss:.4f} | Dice: {mean_dice:.4f} | HD95: {mean_hd95:.2f}", flush=True)

        append_metrics_csv(self.metrics_csv, [
            f"{self.round}_val", "", "", f"{avg_val_loss:.4f}",
            f"{avg_dice[0]:.4f}", f"{avg_dice[1]:.4f}", f"{avg_dice[2]:.4f}", f"{mean_dice:.4f}",
            f"{avg_hd95[0]:.2f}", f"{avg_hd95[1]:.2f}", f"{avg_hd95[2]:.2f}", f"{mean_hd95:.2f}",
            f"{avg_sens[0]:.4f}", f"{avg_sens[1]:.4f}", f"{avg_sens[2]:.4f}",
            f"{avg_spec[0]:.4f}", f"{avg_spec[1]:.4f}", f"{avg_spec[2]:.4f}",
            f"{avg_prec[0]:.4f}", f"{avg_prec[1]:.4f}", f"{avg_prec[2]:.4f}",
        ])

        return float(avg_val_loss), len(self.val_loader.dataset), {
            "val_dice": float(mean_dice),
            "institution_id": self.client_id,
        }


def main():
    parser = argparse.ArgumentParser(description="Flower Client for FeTS 2022")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID (1-23, maps to Partition_ID)")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080", help="Address of the FL server")
    parser.add_argument("--data_dir", type=str, default="data/FeTS2022", help="Path to FeTS dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run (append to existing metrics CSV)")
    args = parser.parse_args()

    log_system_info(client_id=args.client_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Client {args.client_id} using device: {device}")

    partitioning_csv = os.path.join(project_root, args.data_dir, "MICCAI_FeTS2022_TrainingData", "partitioning_1.csv")
    if not os.path.exists(partitioning_csv):
        partitioning_csv = os.path.join(project_root, args.data_dir, "partitioning_1.csv")

    abs_data_dir = os.path.join(project_root, args.data_dir)

    train_loader, val_loader, _ = create_data_loaders(
        data_dir=abs_data_dir,
        partitioning_csv=partitioning_csv,
        client_id=args.client_id,
        batch_size=1,
        num_workers=args.num_workers,
    )

    model = get_model()

    metrics_csv = os.path.join(project_root, "fl", f"client_{args.client_id}_metrics.csv")
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
        init_metrics_csv(metrics_csv)

    client = FeTSClient(model, train_loader, val_loader, device, args.client_id, metrics_csv)
    client.round = round_offset

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
