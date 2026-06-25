import os
import csv
import torch
import psutil
import threading
import time
import subprocess
import numpy as np
import platform

def log_system_info(client_id="Centralized"):
    print("="*50)
    print(f"System Information for: Client {client_id}" if isinstance(client_id, int) else f"System Information for: {client_id}")
    print(f"Node Name: {platform.node()}")
    print(f"OS Platform: {platform.platform()}")
    
    # CPU
    cpu_count = psutil.cpu_count(logical=True)
    cpu_count_phys = psutil.cpu_count(logical=False)
    print(f"CPU: {cpu_count_phys} physical cores, {cpu_count} logical cores")
    
    # RAM
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Total RAM: {total_ram_gb:.2f} GB")
    
    # GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPUs Available: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} (VRAM: {vram_gb:.2f} GB)")
    else:
        print("No CUDA GPUs available.")
    print("="*50)

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

def convert_to_brats_regions(label):
    """
    Converts raw FeTS labels (1=NCR, 2=ED, 4=ET) to 3-channel format:
    Channel 0: Whole Tumor (WT) -> 1, 2, 4
    Channel 1: Tumor Core (TC) -> 1, 4
    Channel 2: Enhancing Tumor (ET) -> 4
    Input shape: B, 1, H, W, D
    Output shape: B, 3, H, W, D
    """
    wt = (label > 0).float()
    tc = torch.logical_or(label == 1, label == 4).float()
    et = (label == 4).float()
    return torch.cat([wt, tc, et], dim=1)

def compute_dice(y_pred, y_true, epsilon=1e-5):
    """
    Compute Dice coefficient.
    y_pred: B, C, ... (probabilities or binary)
    y_true: B, C, ... (binary)
    Returns: B, C tensor of dice scores.
    """
    # flatten spatial dimensions
    y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
    y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
    
    intersection = (y_pred * y_true).sum(-1)
    denominator = y_pred.sum(-1) + y_true.sum(-1)
    
    dice = (2. * intersection + epsilon) / (denominator + epsilon)
    return dice

def init_metrics_csv(filepath, extra_cols=None):
    if extra_cols is None:
        extra_cols = []
    
    headers = [
        "Round/Epoch", 
        "Train Loss", "Val Loss", 
        "Val Dice WT", "Val Dice TC", "Val Dice ET", "Val Mean Dice",
        "Peak RAM (MB)", "Peak VRAM (MB)", "Peak GPU Util (%)"
    ] + extra_cols
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

def append_metrics_csv(filepath, row_data):
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)
