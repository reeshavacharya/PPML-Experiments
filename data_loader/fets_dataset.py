import os
import pandas as pd
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    NormalizeIntensityd,
    SpatialPadd,
    DivisiblePadd,
    ToTensord,
)
from monai.data import Dataset, DataLoader

def get_client_splits(data_dir, partitioning_csv, client_id):
    """
    Reads the partitioning file, filters by client_id, creates data dicts,
    and returns deterministic 80-10-10 train/val/test splits for that specific client.
    """
    df = pd.read_csv(partitioning_csv)
    
    if client_id == 1:
        df = df[df['Partition_ID'] == 1]
    elif client_id == 2:
        df = df[df['Partition_ID'] == 18]
    elif client_id == 3:
        df = df[~df['Partition_ID'].isin([1, 18])]
    else:
        raise ValueError(f"Invalid client_id: {client_id}")
    
    subjects = df['Subject_ID'].tolist()
    
    data_dicts = []
    base_folder = os.path.join(data_dir, "MICCAI_FeTS2022_TrainingData")
    for subject in subjects:
        subject_folder = os.path.join(base_folder, subject)
        if not os.path.exists(subject_folder):
            continue
            
        data_dicts.append({
            "image": [
                os.path.join(subject_folder, f"{subject}_flair.nii.gz"),
                os.path.join(subject_folder, f"{subject}_t1.nii.gz"),
                os.path.join(subject_folder, f"{subject}_t1ce.nii.gz"),
                os.path.join(subject_folder, f"{subject}_t2.nii.gz")
            ],
            "label": os.path.join(subject_folder, f"{subject}_seg.nii.gz")
        })
        
    if len(data_dicts) == 0:
        return [], [], []

    # 80-10-10 split independently for this specific client
    train_files, val_test_files = train_test_split(data_dicts, test_size=0.20, random_state=42)
    val_files, test_files = train_test_split(val_test_files, test_size=0.50, random_state=42)
    return train_files, val_files, test_files

def create_data_loaders(data_dir, partitioning_csv, client_id=None, batch_size=1, num_workers=4):
    """
    Creates train, validation, and test data loaders.
    If client_id is 0 or None (Centralized), it precisely aggregates the independent 80-10-10 splits 
    from clients 1, 2, and 3 to ensure a perfectly fair baseline comparison.
    """
    if client_id in [0, None]:
        train_files, val_files, test_files = [], [], []
        for cid in [1, 2, 3]:
            t, v, te = get_client_splits(data_dir, partitioning_csv, cid)
            train_files.extend(t)
            val_files.extend(v)
            test_files.extend(te)
    else:
        train_files, val_files, test_files = get_client_splits(data_dir, partitioning_csv, client_id)
        
    if len(train_files) == 0:
        raise ValueError(f"No training data found for client {client_id}")

    # Note: FeTS images are usually 240x240x155. We crop them to 128x128x128 patches for Swin UNETR training as per paper.
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=1,
            neg=1,
            num_samples=1, # Number of patches per image
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ToTensord(keys=["image", "label"])
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        DivisiblePadd(keys=["image", "label"], k=32),
        ToTensord(keys=["image", "label"])
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    test_ds = Dataset(data=test_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
