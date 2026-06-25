import os
import sys
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_loader.fets_dataset import create_data_loaders
from models.swin_unetr import get_model
from utils.metrics_utils import ResourceMonitor, init_metrics_csv, append_metrics_csv, convert_to_brats_regions, compute_dice, log_system_info
from monai.losses import DiceCELoss

def train(epochs=100, batch_size=1, lr=3e-4, data_dir="data/FeTS2022"):
    log_system_info(client_id="Centralized Baseline")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    partitioning_csv = os.path.join(project_root, data_dir, "MICCAI_FeTS2022_TrainingData", "partitioning_1.csv")
    if not os.path.exists(partitioning_csv):
        # Fallback in case it's in the root of data_dir
        partitioning_csv = os.path.join(project_root, data_dir, "partitioning_1.csv")
        
    abs_data_dir = os.path.join(project_root, data_dir)
    
    # client_id=0 -> Centralized (all data)
    train_loader, val_loader, _ = create_data_loaders(
        data_dir=abs_data_dir, 
        partitioning_csv=partitioning_csv, 
        client_id=0, 
        batch_size=batch_size
    )

    model = get_model()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # DiceCELoss expects (pred, target) where target is multi-channel binary (WT, TC, ET)
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = os.path.join(project_root, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "swin_unetr_centralized_best.pth")
    
    metrics_csv_path = os.path.join(project_root, "train", "centralized_baseline_metrics.csv")
    init_metrics_csv(metrics_csv_path)
        
    monitor = ResourceMonitor()
    best_val_dice = 0.0

    for epoch in range(1, epochs + 1):
        monitor.start()
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        total_train_steps = len(train_loader)
        log_interval = max(1, total_train_steps // 100)
        
        for i, batch in enumerate(train_loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            labels_converted = convert_to_brats_regions(labels)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_converted)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            if (i + 1) % log_interval == 0 or (i + 1) == total_train_steps:
                percent = int(100 * (i + 1) / total_train_steps)
                print(f"Epoch {epoch} Progress: {percent}% ({i + 1}/{total_train_steps}) - Loss: {loss.item():.4f}")
                sys.stdout.flush()
            
        avg_train_loss = train_loss / train_steps
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        wt_dice_list, tc_dice_list, et_dice_list = [], [], []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                images, labels = batch["image"].to(device), batch["label"].to(device)
                labels_converted = convert_to_brats_regions(labels)
                
                outputs = model(images)
                loss = criterion(outputs, labels_converted)
                
                val_loss += loss.item()
                val_steps += 1
                
                outputs_prob = torch.sigmoid(outputs)
                outputs_binary = (outputs_prob > 0.5).float()
                
                dice_scores = compute_dice(outputs_binary, labels_converted)
                mean_dice = dice_scores.mean(dim=0).cpu().numpy()
                wt_dice_list.append(mean_dice[0])
                tc_dice_list.append(mean_dice[1])
                et_dice_list.append(mean_dice[2])
                
        avg_val_loss = val_loss / val_steps
        avg_wt_dice = np.mean(wt_dice_list)
        avg_tc_dice = np.mean(tc_dice_list)
        avg_et_dice = np.mean(et_dice_list)
        mean_dice_overall = (avg_wt_dice + avg_tc_dice + avg_et_dice) / 3.0
        
        monitor.stop()
            
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Mean Dice: {mean_dice_overall:.4f}")
        print(f"Dice WT: {avg_wt_dice:.4f} | Dice TC: {avg_tc_dice:.4f} | Dice ET: {avg_et_dice:.4f}")
        
        append_metrics_csv(metrics_csv_path, [
            epoch, 
            f"{avg_train_loss:.4f}", 
            f"{avg_val_loss:.4f}", 
            f"{avg_wt_dice:.4f}", 
            f"{avg_tc_dice:.4f}", 
            f"{avg_et_dice:.4f}", 
            f"{mean_dice_overall:.4f}",
            f"{monitor.peak_ram:.2f}",
            f"{monitor.peak_vram:.2f}",
            f"{monitor.peak_gpu_util:.2f}"
        ])
        
        # Save best model
        if mean_dice_overall > best_val_dice:
            print(f"Validation Dice improved from {best_val_dice:.4f} to {mean_dice_overall:.4f}. Saving model...")
            best_val_dice = mean_dice_overall
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mean_dice': mean_dice_overall,
            }, best_model_path)

if __name__ == "__main__":
    train(epochs=100, batch_size=1)
