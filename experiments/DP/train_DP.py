import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import sys

# Ensure root directory is in path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from datasets.medmnist_loader import get_dataloader, DATASETS_CONFIG
from utils.metrics import calculate_metrics

import importlib.util
spec = importlib.util.spec_from_file_location("ViT_Base", str(Path(__file__).resolve().parent.parent.parent / "models" / "ViT-Base.py"))
ViT_Base = importlib.util.module_from_spec(spec)
sys.modules["ViT_Base"] = ViT_Base
spec.loader.exec_module(ViT_Base)
from ViT_Base import build_vit_base

# --- HYPERPARAMETER SWEEPS & CONFIGURATION ---
TARGET_EPSILONS = [1.0, 3.0, 8.0, 15.0]
CLIPPING_BOUNDS = [0.1, 1.0, 5.0]
TARGET_DELTA = 1e-5  # Must be < 1/N. 1e-5 is safe for MedMNIST dataset sizes.

EPOCHS = 10
LEARNING_RATE = 1e-4

# Gradient Accumulation Config to prevent OOM errors with ViT-Base + DP
PHYSICAL_BATCH_SIZE = 2   # What actually fits in VRAM (adjust down to 2 if it crashes)
LOGICAL_BATCH_SIZE = 64   # The batch size Opacus uses to calculate noise
ACCUMULATION_STEPS = LOGICAL_BATCH_SIZE // PHYSICAL_BATCH_SIZE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_CSV = "dp_experiment_results.csv"

def train_one_epoch_dp(model, dataloader, optimizer, criterion, privacy_engine, epoch):
    """
    Implements the DP-SGD training loop using BatchMemoryManager.
    
    NOTE: Instead of manual `if (step + 1) % ACCUMULATION_STEPS == 0` tracking, 
    we use Opacus's native BatchMemoryManager which dynamically modifies the optimizer's 
    `step` and `zero_grad` methods to properly accumulate gradients logically while 
    maintaining mathematically rigorous DP clipping and noise addition.
    """
    model.train()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    
    total_loss = 0.0
    num_samples = 0
    
    total_iters = len(dataloader)
    update_interval = max(1, total_iters // 100)
    
    # We can use standard zero_grad/step because BatchMemoryManager intercepts them
    for step, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", miniters=update_interval, maxinterval=float('inf'))):
        optimizer.zero_grad()
        
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
            labels = labels.float()
        else:
            labels = labels.squeeze(1).long()
            
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)
        
    avg_loss = total_loss / num_samples
    peak_mem_gb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
    
    return avg_loss, peak_mem_gb

def evaluate_model(model, dataloader, criterion, task_type):
    """
    Implements standard evaluation loop.
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if task_type == 'multi-label':
                labels_formatted = labels.float()
            else:
                labels_formatted = labels.squeeze(1).long()
                
            outputs = model(images)
            loss = criterion(outputs, labels_formatted)
            
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            
            all_logits.append(outputs.cpu())
            all_targets.append(labels.cpu())
            
    avg_loss = total_loss / num_samples
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = calculate_metrics(all_logits, all_targets, task_type)
    
    return avg_loss, metrics['accuracy'], metrics['f1_macro'], metrics['auc_macro']

def main():
    results_path = Path("experiments") / "DP" / RESULTS_CSV
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize the CSV file with headers if it doesn't exist.
    if not results_path.exists():
        csv_headers = [
            "Dataset", 
            "Epoch", 
            "Target_Epsilon", 
            "Delta", 
            "Clipping_Bound", 
            "Noise_Multiplier", 
            "Actual_Epsilon_Spent", 
            "Peak_VRAM_GB", 
            "Sec_Per_Epoch", 
            "Val_Accuracy", 
            "Val_Macro_F1", 
            "Val_Macro_AUC"
        ]
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    
    datasets_to_run = ['PathMNIST', 'ChestMNIST']
    
    checkpoint_dir = Path("checkpoints") / "dp"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name in datasets_to_run:
        print(f"\n{'='*50}\nStarting DP Pipeline for {dataset_name}\n{'='*50}")
        
        config = DATASETS_CONFIG[dataset_name]
        task_type = config['task']
        num_classes = config['num_classes']
        
        # 1. Load Data
        # We load the train dataloader with LOGICAL_BATCH_SIZE so Opacus calculates the noise multiplier correctly.
        # BatchMemoryManager will break this down to PHYSICAL_BATCH_SIZE later.
        train_loader, _ = get_dataloader(dataset_name, split='train', batch_size=LOGICAL_BATCH_SIZE)
        val_loader, _ = get_dataloader(dataset_name, split='val', batch_size=PHYSICAL_BATCH_SIZE)
        
        if task_type == 'multi-label':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        for C in CLIPPING_BOUNDS:
            for eps in TARGET_EPSILONS:
                print(f"\n--- Running: Dataset={dataset_name} | Epsilon={eps} | Clip={C} ---")
                
                # 2. Initialize a FRESH model
                model = build_vit_base(num_classes, is_dp=True)
                model = model.to(DEVICE)
                optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
                
                # 3. Attach Opacus Privacy Engine
                privacy_engine = PrivacyEngine()
                
                model, optimizer, train_loader_dp = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    epochs=EPOCHS,
                    target_epsilon=eps,
                    target_delta=TARGET_DELTA,
                    max_grad_norm=C,
                )
                
                # 4. Training Loop
                best_val_f1 = 0.0
                
                for epoch in range(1, EPOCHS + 1):
                    start_time = time.time()
                    
                    # Wrap the dataloader with BatchMemoryManager to handle PHYSICAL_BATCH_SIZE limitations
                    with BatchMemoryManager(
                        data_loader=train_loader_dp, 
                        max_physical_batch_size=PHYSICAL_BATCH_SIZE, 
                        optimizer=optimizer
                    ) as memory_safe_data_loader:
                        train_loss, peak_memory_gb = train_one_epoch_dp(model, memory_safe_data_loader, optimizer, criterion, privacy_engine, epoch)
                        
                    val_loss, val_acc, val_f1, val_auc = evaluate_model(model, val_loader, criterion, task_type)
                    
                    epoch_duration = time.time() - start_time
                    
                    # Get actual privacy spent
                    epsilon_spent = privacy_engine.get_epsilon(TARGET_DELTA)
                    noise_multiplier = optimizer.noise_multiplier
                    
                    print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f} | Epsilon Spent: {epsilon_spent:.4f} | Peak Mem: {peak_memory_gb:.2f} GB")
                    
                    # Append results to CSV
                    with open(results_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            dataset_name, 
                            epoch, 
                            eps, 
                            TARGET_DELTA, 
                            C, 
                            noise_multiplier, 
                            epsilon_spent, 
                            peak_memory_gb, 
                            epoch_duration, 
                            val_acc, 
                            val_f1, 
                            val_auc
                        ])
                    
                    # 5. Save best model configuration
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        save_path = checkpoint_dir / f"vit_{dataset_name}_eps{eps}_C{C}.pth"
                        
                        # Opacus wraps model in GradSampleModule, so we access _module
                        state_dict = model._module.state_dict() if hasattr(model, '_module') else model.state_dict()
                        torch.save(state_dict, save_path)
                        print(f"-> Saved new best model to {save_path.resolve()}")
                    
                # Clean up memory before the next hyperparameter sweep
                del model, optimizer, privacy_engine, train_loader_dp
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()