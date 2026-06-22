import torch
import torch.nn as nn
from monai.networks.nets import ViT

class MONAIViTWrapper(nn.Module):
    def __init__(self, num_classes=14, spatial_dims=2):
        super().__init__()
        # Standard ViT-Base parameters matching torchvision's ViT-B_16
        self.vit = ViT(
            in_channels=3,            # RGB images
            img_size=(224, 224),      # Input image size
            patch_size=(16, 16),      # Patch size
            hidden_size=768,          # ViT-Base hidden size
            mlp_dim=3072,             # ViT-Base MLP dimension
            num_layers=12,            # 12 transformer layers
            num_heads=12,             # 12 attention heads
            proj_type="conv",         # Convolutional patch embedding
            pos_embed_type="learnable", # Learnable positional embedding
            classification=True,      # Use classification head
            num_classes=num_classes,  # 14 classes
            post_activation="Tanh",   # Default post activation
            spatial_dims=spatial_dims # 2D input
        )
        
    def forward(self, x):
        # MONAI ViT returns (x, hidden_states_out) when classification=True
        # We only want the classification logits for our PyTorch training loop
        out, _ = self.vit(x)
        return out

def get_model(num_classes=14, pretrained=False, spatial_dims=2):
    """
    Returns a Vision Transformer from MONAI configured for multi-label classification.
    """
    model = MONAIViTWrapper(num_classes=num_classes, spatial_dims=spatial_dims)
    return model

if __name__ == "__main__":
    # Test model initialization
    model = get_model()
    print("MONAI ViT Wrapper initialized.")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")

