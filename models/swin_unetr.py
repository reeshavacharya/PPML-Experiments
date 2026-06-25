import torch
from monai.networks.nets import SwinUNETR

def get_model(in_channels=4, out_channels=3, feature_size=48, use_checkpoint=True, spatial_dims=3):
    """
    Instantiates the Swin UNETR model using MONAI.
    
    Args:
        in_channels: number of input MRI modalities (e.g. 4 for flair, t1, t1ce, t2)
        out_channels: number of output segmentation channels (e.g. 3 for WT, TC, ET)
        feature_size: feature dimension size
        use_checkpoint: whether to use gradient checkpointing to save memory
        spatial_dims: number of spatial dimensions (3 for volumetric data)
        
    Returns:
        SwinUNETR model
    """
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
    )
    return model


