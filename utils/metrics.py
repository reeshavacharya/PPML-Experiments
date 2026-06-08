import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def calculate_metrics(logits, targets, task_type, threshold=0.5):
    """
    Calculates Overall Accuracy, Macro F1-Score, and Macro AUC-ROC based on the MedMNIST task type.
    
    CRITICAL: This function should be called ONCE at the end of the validation epoch 
    using the accumulated logits and targets for the entire dataset. Calling this per-batch 
    will cause roc_auc_score to crash if a batch is missing a particular class.
    
    Args:
        logits (torch.Tensor): Raw outputs from the ViT model (shape: [N_samples, num_classes]).
        targets (torch.Tensor): Ground truth labels.
        task_type (str): 'multi-class' or 'multi-label' (from DATASETS_CONFIG).
        threshold (float): Probability threshold for multi-label classification.
        
    Returns:
        dict: A dictionary containing 'accuracy', 'f1_macro', and 'auc_macro'.
    """
    
    # Ensure targets are appropriately sized for processing
    if targets.ndim > 1 and targets.shape[-1] == 1:
        targets = targets.squeeze(1)

    if task_type == 'multi-class':
        # 1. Apply Softmax to logits to get probabilities.
        probs = F.softmax(logits, dim=1)
        
        # 2. Detach and convert both probabilities and targets to CPU NumPy arrays.
        probs_np = probs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # 3. Apply argmax to probabilities to get discrete predictions.
        preds_np = np.argmax(probs_np, axis=1)
        
        # 4. Calculate accuracy_score.
        acc = accuracy_score(targets_np, preds_np)
        
        # 5. Calculate f1_score (average='macro').
        f1 = f1_score(targets_np, preds_np, average='macro')
        
        # 6. Calculate roc_auc_score (pass targets and PROBABILITIES, set multi_class='ovr', average='macro').
        # Using multi_class='ovr' works for both binary and true multi-class scenarios
        try:
            auc = roc_auc_score(targets_np, probs_np, multi_class='ovr', average='macro')
        except ValueError as e:
            # Fallback if there's only one class present in y_true, etc.
            print(f"Warning: ROC AUC calculation failed: {e}")
            auc = float('nan')

    elif task_type == 'multi-label':
        # 1. Apply Sigmoid to logits to get probabilities.
        probs = torch.sigmoid(logits)
        
        # 2. Detach and convert both probabilities and targets to CPU NumPy arrays.
        probs_np = probs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # 3. Convert probabilities to binary predictions (1 if >= threshold, else 0).
        preds_np = (probs_np >= threshold).astype(int)
        
        # 4. Calculate accuracy_score (this requires an exact match for all 14 diseases in a row).
        acc = accuracy_score(targets_np, preds_np)
        
        # 5. Calculate f1_score (average='macro').
        f1 = f1_score(targets_np, preds_np, average='macro')
        
        # 6. Calculate roc_auc_score (pass targets and PROBABILITIES, average='macro').
        try:
            auc = roc_auc_score(targets_np, probs_np, average='macro')
        except ValueError as e:
            print(f"Warning: ROC AUC calculation failed: {e}")
            auc = float('nan')
            
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return {"accuracy": acc, "f1_macro": f1, "auc_macro": auc}