import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16
from medmnist import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, BloodMNIST, OrganAMNIST

def train_models():
    # --- Configuration ---
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}\n")

    # Mapping each MedMNIST dataset to its specific properties
    datasets_config = {
        'PathMNIST':   {'dataset_cls': PathMNIST,   'task': 'multi-class', 'num_classes': 9},
        'ChestMNIST':  {'dataset_cls': ChestMNIST,  'task': 'multi-label', 'num_classes': 14},
        'DermaMNIST':  {'dataset_cls': DermaMNIST,  'task': 'multi-class', 'num_classes': 7},
        'OCTMNIST':    {'dataset_cls': OCTMNIST,    'task': 'multi-class', 'num_classes': 4},
        'BloodMNIST':  {'dataset_cls': BloodMNIST,  'task': 'multi-class', 'num_classes': 8},
        'OrganAMNIST': {'dataset_cls': OrganAMNIST, 'task': 'multi-class', 'num_classes': 11}
    }

    # ViT-Base expects 224x224 RGB inputs. 
    # medmnist outputs PIL images, so we convert grayscale datasets to RGB.
    data_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loop over all datasets and train a separate model for each
    for dataset_name, config in datasets_config.items():
        print("="*60)
        print(f" Starting Training Pipeline for: {dataset_name} ")
        print("="*60)
        
        DatasetClass = config['dataset_cls']
        task_type = config['task']
        num_classes = config['num_classes']
        
        # --- Data Loading ---
        # download=True will fetch the .npz files into ~/.medmnist if not present
        train_dataset = DatasetClass(split='train', transform=data_transform, download=True)
        val_dataset = DatasetClass(split='val', transform=data_transform, download=True)
        
        print(f"\n{dataset_name} Task: {task_type} | Classes: {num_classes}")
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        # --- Model Setup ---
        print("Initializing ViT-Base (~86M parameters)...")
        model = vit_b_16(weights=None) 
        
        # Replace the classification head for the dataset's specific class count
        model.heads = nn.Sequential(
            nn.Linear(model.heads[0].in_features, num_classes)
        )
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
                
                train_loss += loss.item() * images.size(0)
                
            train_loss = train_loss / len(train_dataset)
            
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
                    
            val_loss = val_loss / len(val_dataset)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save the best model for this specific dataset
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f'vit_base_{dataset_name}_best.pth'
                torch.save(model.state_dict(), save_path)
                print(f"-> Saved new best model: {save_path}")
                
        print(f"\nFinished training {dataset_name}.\n")

if __name__ == '__main__':
    train_models()