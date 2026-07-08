import timm


def get_model(num_classes=8, pretrained=True):
    """Returns a timm ViT-B/16 (patch 16, 224x224 input, ImageNet-21k
    pretrained) with its classification head resized to `num_classes`."""
    return timm.create_model(
        "vit_base_patch16_224.augreg_in21k",
        pretrained=pretrained,
        num_classes=num_classes,
    )
