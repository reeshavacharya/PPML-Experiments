import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_loader.nih_chest import create_data_loaders
from models.monai_vit import get_model

def test(batch_size=32, data_dir="data/NIH-Chest"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create test data loader
    abs_data_dir = os.path.join(project_root, data_dir)
    _, _, test_loader = create_data_loaders(data_dir=abs_data_dir, batch_size=batch_size)

    # Initialize model
    model = get_model(num_classes=14, pretrained=False)
    
    # Load best model weights
    dataset_name = os.path.basename(data_dir.strip('/'))
    model_name = "vit_base"
    best_model_path = os.path.join(project_root, "checkpoint", f"{model_name}_{dataset_name}.pth")
    if not os.path.exists(best_model_path):
        print(f"Error: Best model checkpoint not found at {best_model_path}")
        return
        
    print(f"Loading best model from {best_model_path}...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    
    test_loss = 0.0
    test_steps = 0
    all_preds = []
    all_labels = []
    
    total_test_batches = len(test_loader)
    last_test_pct = -1
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            test_steps += 1
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            pct = int(100 * (i + 1) / total_test_batches)
            if pct % 10 == 0 and pct != last_test_pct:
                print(f"  Testing: {pct}% complete", flush=True)
                last_test_pct = pct
            
    avg_test_loss = test_loss / test_steps
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_preds_prob = 1 / (1 + np.exp(-all_preds))
    
    # Label names from NIH-Chest dataset
    label_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
        'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    print("\nPer-class ROC-AUC Scores:")
    print("-" * 30)
    aucs = []
    for i, label in enumerate(label_names):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds_prob[:, i])
            aucs.append(auc)
            print(f"{label:20s}: {auc:.4f}")
        except ValueError:
            print(f"{label:20s}: N/A (Only one class in ground truth)")
            
    macro_auc = np.mean(aucs) if aucs else 0.0
    print("-" * 30)
    print(f"Macro Average AUC   : {macro_auc:.4f}")

if __name__ == "__main__":
    test(batch_size=32)
