import os
import csv
import torch
import psutil
import threading
import time
import subprocess
import numpy as np
import platform
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

from utils.cldice import compute_cldice  # re-exported for a single metrics import at call sites

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
        proc = psutil.Process()
        while self.keep_running:
            # Per-process RSS so co-located clients don't inflate each other's reading
            ram_mb = proc.memory_info().rss / (1024 * 1024)
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

def compute_sensitivity(y_pred, y_true, epsilon=1e-5):
    """TP / (TP + FN). Input: B, C, ... (binary). Returns: B, C."""
    y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
    y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
    tp = (y_pred * y_true).sum(-1)
    fn = ((1 - y_pred) * y_true).sum(-1)
    return (tp + epsilon) / (tp + fn + epsilon)


def compute_specificity(y_pred, y_true, epsilon=1e-5):
    """TN / (TN + FP). Input: B, C, ... (binary). Returns: B, C."""
    y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
    y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
    tn = ((1 - y_pred) * (1 - y_true)).sum(-1)
    fp = (y_pred * (1 - y_true)).sum(-1)
    return (tn + epsilon) / (tn + fp + epsilon)


def compute_precision_metric(y_pred, y_true, epsilon=1e-5):
    """TP / (TP + FP). Input: B, C, ... (binary). Returns: B, C."""
    y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
    y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
    tp = (y_pred * y_true).sum(-1)
    fp = (y_pred * (1 - y_true)).sum(-1)
    return (tp + epsilon) / (tp + fp + epsilon)


def compute_hausdorff95(y_pred_np, y_true_np):
    """
    95th percentile Hausdorff Distance per channel.
    y_pred_np, y_true_np: numpy arrays of shape (C, H, W, D), binary.
    Returns: numpy array of shape (C,).
    """
    num_channels = y_pred_np.shape[0]
    hd95 = np.zeros(num_channels)
    conn = generate_binary_structure(3, 1)

    for c in range(num_channels):
        pred = y_pred_np[c].astype(bool)
        true = y_true_np[c].astype(bool)

        if not pred.any() and not true.any():
            hd95[c] = 0.0
            continue
        if not pred.any() or not true.any():
            hd95[c] = np.nan
            continue

        pred_border = np.logical_xor(pred, binary_erosion(pred, structure=conn))
        true_border = np.logical_xor(true, binary_erosion(true, structure=conn))

        if not pred_border.any() or not true_border.any():
            hd95[c] = 0.0
            continue

        dt_true = distance_transform_edt(~true)
        dt_pred = distance_transform_edt(~pred)

        dists = np.concatenate([dt_true[pred_border], dt_pred[true_border]])
        hd95[c] = np.percentile(dists, 95)

    return hd95


def compute_metric_statistics(values):
    """Compute mean, std, median, 25th and 75th percentile for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "q25": round(float(np.percentile(arr, 25)), 4),
        "q75": round(float(np.percentile(arr, 75)), 4),
    }


def init_metrics_csv(filepath, region_names=("wt", "tc", "et"), extra_cols=None, include_cldice=False):
    """
    region_names: per-region labels for the Dice/HD95/sensitivity/specificity/
    precision columns — defaults to FeTS's (WT, TC, ET) for backward
    compatibility. CAS callers should pass region_names=("coronary",).
    include_cldice: adds a clDice column per region + a mean column — off by
    default since it's only meaningful for tubular structures (CAS), not the
    FeTS tumor regions.
    """
    if extra_cols is None:
        extra_cols = []
    labels = [r.upper() for r in region_names]

    headers = ["Round/Epoch", "Train Loss", "SGD Steps", "Val Loss"]
    headers += [f"Val Dice {r}" for r in labels] + ["Val Mean Dice"]
    headers += [f"Val HD95 {r}" for r in labels] + ["Val Mean HD95"]
    if include_cldice:
        headers += [f"Val clDice {r}" for r in labels] + ["Val Mean clDice"]
    headers += [f"Val Sensitivity {r}" for r in labels]
    headers += [f"Val Specificity {r}" for r in labels]
    headers += [f"Val Precision {r}" for r in labels]
    headers += extra_cols

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

def append_metrics_csv(filepath, row_data):
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)


def init_classification_metrics_csv(filepath, class_names, extra_cols=None):
    """Header for the classification family (Fed-ISIC2019/ViT), parallel to
    init_metrics_csv but with per-sample scalar metrics (balanced accuracy,
    macro-F1, per-class recall, macro OvR AUC, Cohen's kappa) instead of
    per-region Dice/HD95 tensors. Rows are still written with the shared
    append_metrics_csv — only the header schema differs from segmentation."""
    if extra_cols is None:
        extra_cols = []

    headers = ["Round/Epoch", "Train Loss", "SGD Steps", "Val Loss"]
    headers += ["Val Balanced Accuracy", "Val Macro F1"]
    headers += [f"Val Recall {c}" for c in class_names]
    headers += ["Val Macro AUC OvR", "Val Cohen Kappa"]
    headers += extra_cols

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def init_multilabel_metrics_csv(filepath, class_names, extra_cols=None):
    """Header for the multi-label family (NIH ChestX-ray14/ViT), sibling to
    init_classification_metrics_csv. Per-label scalar metrics (mean AUC-ROC,
    macro-F1, per-label AUC, per-label sensitivity at a fixed threshold)
    instead of ISIC's single-label balanced-accuracy/kappa shape — 14
    independent binary findings per image, not one mutually-exclusive
    winner. Rows are still written with the shared append_metrics_csv."""
    if extra_cols is None:
        extra_cols = []

    headers = ["Round/Epoch", "Train Loss", "SGD Steps", "Val Loss"]
    headers += ["Val Mean AUC ROC", "Val Macro F1"]
    headers += [f"Val AUC {c}" for c in class_names]
    headers += [f"Val Sensitivity {c}" for c in class_names]
    headers += extra_cols

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
