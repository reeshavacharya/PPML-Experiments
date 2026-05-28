import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm

from dataset import NIHChestDataset, get_transforms, DISEASES

def train_model():
    # --- Configuration ---
    DATA_DIR = './data/NIH-CHEST'
    BATCH_SIZE = 32 # Adjust based on your VRAM (32 requires ~12-16GB VRAM)
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_CLASSES = len(DISEASES) # 14
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")

    # --- Data Loading ---
    train_tf, val_tf = get_transforms()
    
    print("Loading datasets...")
    train_dataset = NIHChestDataset(DATA_DIR, split='train', transform=train_tf)
    val_dataset = NIHChestDataset(DATA_DIR, split='val', transform=val_tf)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model Setup ---
    print("Initializing ViT-Base (~86M parameters)...")
    # Note: Use weights=None to train entirely from scratch, or use weights=ViT_B_16_Weights.DEFAULT 
    # for ImageNet transfer learning (highly recommended for faster convergence).
    model = vit_b_16(weights=None) 
    
    # Replace the classification head for our 14 multi-label classes
    model.heads = nn.Sequential(
        nn.Linear(model.heads[0].in_features, NUM_CLASSES)
    )
    model = model.to(DEVICE)
    
    # Print param count to verify ~86M
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {num_params / 1e6:.2f}M")

    # Multi-label classification requires BCE loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # --- Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        progress_bar = tqdm(train_loader, desc="Training")
        
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
            
        train_loss = train_loss / len(train_dataset)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_dataset)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'vit_base_nih_best.pth')
            print("Saved new best model.")

if __name__ == '__main__':
    train_model()