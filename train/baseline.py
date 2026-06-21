import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import numpy as np
import csv
import psutil
import threading
import time
import subprocess

class ResourceMonitor:
    def __init__(self):
        self.keep_running = True
        self.peak_ram = 0.0
        self.peak_gpu_util = 0.0
        self.peak_vram = 0.0
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True

    def start(self):
        self.keep_running = True
        self.peak_ram = 0.0
        self.peak_gpu_util = 0.0
        self.peak_vram = 0.0
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.keep_running = False
        if self.thread.is_alive():
            self.thread.join()

    def _monitor(self):
        while self.keep_running:
            # RAM in MB
            ram_mb = psutil.virtual_memory().used / (1024 * 1024)
            if ram_mb > self.peak_ram:
                self.peak_ram = ram_mb
                
            # GPU stats via nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE, text=True
                )
                lines = result.stdout.strip().split('\n')
                total_gpu_util = 0
                total_vram = 0
                for line in lines:
                    if line:
                        parts = line.split(',')
                        if len(parts) == 2:
                            util = float(parts[0].strip())
                            vram = float(parts[1].strip())
                            total_gpu_util = max(total_gpu_util, util)
                            total_vram += vram
                
                if total_gpu_util > self.peak_gpu_util:
                    self.peak_gpu_util = total_gpu_util
                if total_vram > self.peak_vram:
                    self.peak_vram = total_vram
            except Exception:
                pass
            
            time.sleep(1)

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_loader.nih_chest import create_data_loaders
from models.vit_base import get_model

def train(epochs=10, batch_size=32, lr=1e-4, data_dir="data/NIH-Chest"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    # Construct absolute path for data_dir
    abs_data_dir = os.path.join(project_root, data_dir)
    train_loader, val_loader, _ = create_data_loaders(data_dir=abs_data_dir, batch_size=batch_size)

    # Initialize model
    model = get_model(num_classes=14, pretrained=True)
    model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

    checkpoint_dir = os.path.join(project_root, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    dataset_name = os.path.basename(data_dir.strip('/'))
    model_name = "vit_base"
    best_model_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset_name}.pth")
    
    # Setup CSV logging
    train_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_csv_path = os.path.join(train_dir, "metrics.csv")
    with open(metrics_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Train Loss", "Val Loss", "Val Macro AUC", "Val Accuracy", 
            "Peak RAM (MB)", "Peak VRAM (MB)", "Peak GPU Util (%)"
        ])
        
    monitor = ResourceMonitor()
    
    best_val_auc = 0.0

    for epoch in range(1, epochs + 1):
        monitor.start()
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        total_train_batches = len(train_loader)
        last_train_pct = -1
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            pct = int(100 * (i + 1) / total_train_batches)
            if pct % 10 == 0 and pct != last_train_pct:
                print(f"  Training: {pct}% complete", flush=True)
                last_train_pct = pct
            
        avg_train_loss = train_loss / train_steps
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        total_val_batches = len(val_loader)
        last_val_pct = -1
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_steps += 1
                
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                pct = int(100 * (i + 1) / total_val_batches)
                if pct % 10 == 0 and pct != last_val_pct:
                    print(f"  Validation: {pct}% complete", flush=True)
                    last_val_pct = pct
                
        avg_val_loss = val_loss / val_steps
        
        # Calculate validation AUC
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Apply sigmoid to get probabilities
        all_preds_prob = 1 / (1 + np.exp(-all_preds))
        
        # Calculate macro AUC
        try:
            val_auc = roc_auc_score(all_labels, all_preds_prob, average='macro')
        except ValueError:
            print("Warning: Only one class present in y_true. AUC score is not defined in that case. Setting AUC to 0.")
            val_auc = 0.0
            
        preds_binary = (all_preds_prob > 0.5).astype(int)
        val_accuracy = np.mean(preds_binary == all_labels)
        
        monitor.stop()
            
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro AUC: {val_auc:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # Log metrics to CSV
        with open(metrics_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f"{avg_train_loss:.4f}", 
                f"{avg_val_loss:.4f}", 
                f"{val_auc:.4f}", 
                f"{val_accuracy:.4f}",
                f"{monitor.peak_ram:.2f}",
                f"{monitor.peak_vram:.2f}",
                f"{monitor.peak_gpu_util:.2f}"
            ])
        
        # We handle scheduler logic explicitly since it depends on torch version slightly
        # ReduceLROnPlateau expects the metric (val_auc in this case)
        # Note: older versions don't have verbose, it's deprecated in 2.2, but typical usage is fine.
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            print(f"Validation AUC improved from {best_val_auc:.4f} to {val_auc:.4f}. Saving model...")
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, best_model_path)

if __name__ == "__main__":
    train(epochs=10, batch_size=32)
