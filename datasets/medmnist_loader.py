from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, BloodMNIST, OrganAMNIST

DATASETS_CONFIG = {
    'PathMNIST':   {'dataset_cls': PathMNIST,   'task': 'multi-class', 'num_classes': 9},
    'ChestMNIST':  {'dataset_cls': ChestMNIST,  'task': 'multi-label', 'num_classes': 14},
    'DermaMNIST':  {'dataset_cls': DermaMNIST,  'task': 'multi-class', 'num_classes': 7},
    'OCTMNIST':    {'dataset_cls': OCTMNIST,    'task': 'multi-class', 'num_classes': 4},
    'BloodMNIST':  {'dataset_cls': BloodMNIST,  'task': 'multi-class', 'num_classes': 8},
    'OrganAMNIST': {'dataset_cls': OrganAMNIST, 'task': 'multi-class', 'num_classes': 11}
}

def build_transform():
    """
    Returns the transformation pipeline for ViT-Base models.
    Converts to RGB, resizes to 224x224, converts to tensor, and normalizes.
    """
    return transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_dataloader(dataset_name, split, batch_size, num_workers=4, pin_memory=True):
    """
    Creates and returns a DataLoader for the specified MedMNIST dataset and split.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'PathMNIST')
        split (str): Split to load ('train', 'val', or 'test')
        batch_size (int): Number of samples per batch
        num_workers (int): Number of subprocesses to use for data loading
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory
        
    Returns:
        DataLoader, config: The initialized DataLoader and the dataset config dict.
    """
    if dataset_name not in DATASETS_CONFIG:
        raise ValueError(f"Dataset {dataset_name} is not supported. Supported datasets: {list(DATASETS_CONFIG.keys())}")
        
    config = DATASETS_CONFIG[dataset_name]
    DatasetClass = config['dataset_cls']
    
    dataset_root = Path('data')
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    transform = build_transform()
    
    dataset = DatasetClass(
        split=split,
        transform=transform,
        download=True,
        root=str(dataset_root)
    )
    
    # Shuffle only if it's the training split
    shuffle = (split == 'train')
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle
    )
    
    return loader, config
