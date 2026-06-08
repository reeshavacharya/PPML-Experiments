import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from opacus.validators import ModuleValidator

def build_vit_base(num_classes, is_dp=False):
    """
    Initializes a ViT-Base (vit_b_16) model (~86M parameters).
    
    Args:
        num_classes (int): Number of output classes.
        is_dp (bool): If True, replaces LayerNorm with GroupNorm for Opacus compatibility 
                      and uses ImageNet weights for DP fine-tuning.
        
    Returns:
        model: PyTorch ViT model.
    """
    if is_dp:
        # For DP, we MUST start with a model that hasn't seen sensitive MedMNIST data.
        # We load public ImageNet weights to allow for faster DP fine-tuning.
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    else:
        # For baseline from-scratch training (if needed)
        model = vit_b_16(weights=None) 
    
    # Replace the classification head for the specific dataset
    model.heads = nn.Sequential(
        nn.Linear(model.heads[0].in_features, num_classes)
    )
    
    # --- The Critical DP Fix ---
    if is_dp:
        # Validates and automatically replaces incompatible layers (like LayerNorm)
        errors = ModuleValidator.validate(model, strict=False)
        if len(errors) > 0:
            model = ModuleValidator.fix(model)
            
    return model