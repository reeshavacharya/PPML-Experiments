import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=14, pretrained=True):
    """
    Returns a ViT-B/16 model modified for `num_classes` outputs.
    """
    if pretrained:
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
    else:
        model = models.vit_b_16()
        
    # Modify the classification head for the new number of classes
    # The default head is an nn.Linear(in_features=768, out_features=1000)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model