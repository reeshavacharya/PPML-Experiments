import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Ensure root directory is in path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from datasets.medmnist_loader import DATASETS_CONFIG, get_dataloader
# Assuming the file is named ViT-Base.py we need to import it carefully due to the hyphen
import importlib.util
spec = importlib.util.spec_from_file_location("ViT_Base", str(Path(__file__).resolve().parent.parent.parent / "models" / "ViT-Base.py"))
ViT_Base = importlib.util.module_from_spec(spec)
sys.modules["ViT_Base"] = ViT_Base
spec.loader.exec_module(ViT_Base)
from ViT_Base import build_vit_base

def train_models(datasets=None, batch_size=8):
    # --- Configuration ---
    BATCH_SIZE = batch_size
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}\n")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    print("Batch Size:", BATCH_SIZE)

    # Filter datasets if specified
    datasets_to_train = DATASETS_CONFIG
    if datasets is not None:
        datasets_to_train = {k: v for k, v in DATASETS_CONFIG.items() if k in datasets}

    checkpoint_dir = Path('checkpoints') / 'baseline'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Loop over all datasets and train a separate model for each
    for dataset_name, config in datasets_to_train.items():
        print("="*60)
        print(f" Starting Training Pipeline for: {dataset_name} ")
        print("="*60)
        
        task_type = config['task']
        num_classes = config['num_classes']
        
        # --- Data Loading ---
        train_loader, _ = get_dataloader(dataset_name, split='train', batch_size=BATCH_SIZE)
        val_loader, _ = get_dataloader(dataset_name, split='val', batch_size=BATCH_SIZE)
        
        print(f"\n{dataset_name} Task: {task_type} | Classes: {num_classes}")
        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        # --- Model Setup ---
        print("Initializing ViT-Base (~86M parameters)...")
        model = build_vit_base(num_classes, is_dp=False)
        model = model.to(DEVICE)

        # --- Loss and Optimizer ---
        # Multi-label requires BCE loss, Multi-class requires standard Cross Entropy
        if task_type == 'multi-label':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
            
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
        
        # --- Training Loop ---
        best_val_loss = float('inf')

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            
            print(f"\n[{dataset_name}] Epoch {epoch+1}/{EPOCHS}")
            for images, labels in train_loader:
                images = images.to(DEVICE)
                
                # Shape labels appropriately based on task format
                if task_type == 'multi-label':
                    labels = labels.to(DEVICE).float()
                else:
                    # MedMNIST multi-class outputs [B, 1], CrossEntropyLoss requires [B]
                    labels = labels.to(DEVICE).squeeze(1).long()
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                # Use batch size for correct weighting
                train_loss += loss.item() * images.size(0)
                
            train_loss = train_loss / len(train_loader.dataset)
            
            # --- Validation Loop ---
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    
                    if task_type == 'multi-label':
                        labels = labels.to(DEVICE).float()
                    else:
                        labels = labels.to(DEVICE).squeeze(1).long()
                        
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    
            val_loss = val_loss / len(val_loader.dataset)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save the best model for this specific dataset
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = checkpoint_dir / f'vit_base_{dataset_name}_best.pth'
                torch.save(model.state_dict(), save_path)
                print(f"-> Saved new best model: {save_path.resolve()}")
                
        print(f"\nFinished training {dataset_name}.\n")

if __name__ == '__main__':
    train_models()
