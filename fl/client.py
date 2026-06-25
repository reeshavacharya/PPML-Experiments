import argparse
import os
import sys
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
from utils.metrics_utils import ResourceMonitor, init_metrics_csv, append_metrics_csv, convert_to_brats_regions, compute_dice, log_system_info

class FeTSClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, client_id, metrics_csv):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.client_id = client_id
        self.criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        self.optimizer = AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.monitor = ResourceMonitor()
        self.metrics_csv = metrics_csv
        self.round = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.round += 1
        self.monitor.start()
        
        self.model.train()
        train_loss = 0.0
        train_steps = 0
        total_train_steps = len(self.train_loader)
        log_interval = max(1, total_train_steps // 100)
        
        # 1 local epoch per round as per paper
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
                percent = int(100 * (i + 1) / total_train_steps)
                print(f"[Client {self.client_id}, Round {self.round}] Progress: {percent}% ({i + 1}/{total_train_steps}) - Loss: {loss.item():.4f}")
                sys.stdout.flush()
            
        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
        self.monitor.stop()
        
        print(f"[Client {self.client_id}, Round {self.round}] Train Loss: {avg_train_loss:.4f}")
        
        append_metrics_csv(self.metrics_csv, [
            self.round, 
            f"{avg_train_loss:.4f}", 
            "", "", "", "", "",
            f"{self.monitor.peak_ram:.2f}",
            f"{self.monitor.peak_vram:.2f}",
            f"{self.monitor.peak_gpu_util:.2f}"
        ])
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"train_loss": avg_train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.monitor.start()
        
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        wt_dice_list, tc_dice_list, et_dice_list = [], [], []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                labels_converted = convert_to_brats_regions(labels)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels_converted)
                
                val_loss += loss.item()
                val_steps += 1
                
                outputs_prob = torch.sigmoid(outputs)
                outputs_binary = (outputs_prob > 0.5).float()
                
                dice_scores = compute_dice(outputs_binary, labels_converted)
                mean_dice = dice_scores.mean(dim=0).cpu().numpy()
                wt_dice_list.append(mean_dice[0])
                tc_dice_list.append(mean_dice[1])
                et_dice_list.append(mean_dice[2])
                
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        avg_wt_dice = np.mean(wt_dice_list) if wt_dice_list else 0
        avg_tc_dice = np.mean(tc_dice_list) if tc_dice_list else 0
        avg_et_dice = np.mean(et_dice_list) if et_dice_list else 0
        mean_dice_overall = (avg_wt_dice + avg_tc_dice + avg_et_dice) / 3.0
        
        self.monitor.stop()
        
        print(f"[Client {self.client_id}, Round {self.round}] Val Loss: {avg_val_loss:.4f} | Val Dice: {mean_dice_overall:.4f}")
        
        append_metrics_csv(self.metrics_csv, [
            f"{self.round}_val", 
            "", 
            f"{avg_val_loss:.4f}", 
            f"{avg_wt_dice:.4f}", 
            f"{avg_tc_dice:.4f}", 
            f"{avg_et_dice:.4f}", 
            f"{mean_dice_overall:.4f}",
            f"{self.monitor.peak_ram:.2f}",
            f"{self.monitor.peak_vram:.2f}",
            f"{self.monitor.peak_gpu_util:.2f}"
        ])
        
        return float(avg_val_loss), len(self.val_loader.dataset), {"val_dice": float(mean_dice_overall)}

def main():
    parser = argparse.ArgumentParser(description="Flower Client for FeTS 2022")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID (1, 2, or 3)")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080", help="Address of the FL server")
    parser.add_argument("--data_dir", type=str, default="data/FeTS2022", help="Path to FeTS dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
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
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    metrics_csv = os.path.join(project_root, "fl", f"client_{args.client_id}_metrics.csv")
    init_metrics_csv(metrics_csv)

    client = FeTSClient(model, train_loader, val_loader, device, args.client_id, metrics_csv)

    fl.client.start_client(server_address=args.server_address, client=client.to_client())

if __name__ == "__main__":
    main()
