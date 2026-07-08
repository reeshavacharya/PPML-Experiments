"""Fed-ISIC2019 data loading: manifest-driven splits + torchvision transform
pipeline for ViT-B/16 classification.

Mirrors the shape of data_loader/cas_dataset.py (get_centralized_splits /
get_client_splits / create_data_loaders) so call sites in train/, test/, and
fl/ stay similar across dataset families. Key differences from the
segmentation loaders:
  - 2D RGB dermoscopy images (not 3D volumes) — no MONAI dict-transforms or
    sliding-window inference needed, plain torchvision Dataset/DataLoader.
  - 8 mutually-exclusive classes (single `target` int per image), not
    per-region binary masks.
  - The train/val split comes from data/ISIC/isic_split.csv (built by
    `python data_loader/isic_prepare.py`), which adds two *different*
    stratified 80/20 val carves on top of train_test_split.csv's pooled
    18,597/4,650 train/test split and 6-center `center` column:
    "pooled_split" (centralized) and "client_split" (FL, stratified within
    each center independently). Test is always the shared, undivided
    4,650-row pool (server-side global test), same convention as CAS.
  - client_id is 1-based (1..6); center in the manifest is 0-based (0..5),
    so client_id - 1 == center.
"""
import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGES_DIRNAME = "ISIC_2019_Training_Input"
IMG_SIZE = 224
NUM_CLASSES = 8
NUM_CLIENTS = 6
CLASS_NAMES = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
DEFAULT_DATA_DIR = "data/ISIC"
DEFAULT_MANIFEST = "isic_split.csv"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ShadeOfGray:
    """Shades-of-Gray color constancy (Minkowski p-norm illuminant estimate,
    p=6 — the standard setting used in dermoscopy preprocessing, e.g. the
    ISIC 2019 winning solutions). Operates on a PIL RGB image, returns a PIL
    RGB image.
    """

    def __init__(self, power=6):
        self.power = power

    def __call__(self, img):
        arr = np.asarray(img).astype(np.float32)
        arr_power = np.power(arr, self.power)
        illum = np.power(np.mean(arr_power, axis=(0, 1)), 1.0 / self.power)
        illum_norm = illum / np.sqrt(np.sum(illum ** 2))
        gain = 1.0 / (illum_norm * np.sqrt(3))
        corrected = np.clip(arr * gain, 0, 255).astype(np.uint8)
        return Image.fromarray(corrected)


def build_transforms(train):
    # ShadeOfGray/Resize run on PIL, everything else on tensor: torchvision's
    # PIL-backed ColorJitter(hue=...) hits a numpy>=2 uint8-overflow bug
    # (np.uint8(negative_hue*255) raises under NEP 50), while the tensor
    # backend doesn't go through that code path — so ToTensor happens before
    # crop/flip/rotation/jitter rather than after.
    ops = [ShadeOfGray(power=6), transforms.Resize(IMG_SIZE), transforms.ToTensor()]
    if train:
        ops += [
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # brightness-contrast
            transforms.ColorJitter(hue=0.05, saturation=0.2),      # color jitter
        ]
    else:
        ops += [transforms.CenterCrop(IMG_SIZE)]
    ops += [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    if train:
        ops += [transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))]  # coarse dropout
    return transforms.Compose(ops)


class ISICDataset(Dataset):
    def __init__(self, data_dir, rows, transform):
        self.data_dir = data_dir
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(self.data_dir, IMAGES_DIRNAME, f"{row['image']}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(row["target"])
        return image, label


def _read_manifest(manifest_csv):
    with open(manifest_csv, newline="") as f:
        return list(csv.DictReader(f))


def get_centralized_splits(manifest_csv):
    """Centralized train/val = pooled 80/20 stratified split of the 18,597
    train rows; test = shared 4,650-row pool."""
    rows = _read_manifest(manifest_csv)
    train_rows = [r for r in rows if r["pooled_split"] == "train"]
    val_rows = [r for r in rows if r["pooled_split"] == "val"]
    test_rows = [r for r in rows if r["pooled_split"] == "test"]
    return train_rows, val_rows, test_rows


def get_client_splits(manifest_csv, client_id):
    """Client's own center-stratified 80/20 train/val shard, plus the shared
    (undivided) pooled test set."""
    if client_id < 1:
        raise ValueError(f"client_id must be >= 1 for get_client_splits (got {client_id})")
    center = str(client_id - 1)

    rows = _read_manifest(manifest_csv)
    train_rows = [r for r in rows if r["center"] == center and r["client_split"] == "train"]
    val_rows = [r for r in rows if r["center"] == center and r["client_split"] == "val"]
    test_rows = [r for r in rows if r["pooled_split"] == "test"]

    if not train_rows:
        raise ValueError(f"No training data found for client {client_id} (center {center})")

    return train_rows, val_rows, test_rows


def create_data_loaders(data_dir, manifest_csv, client_id=None, batch_size=32, num_workers=4,
                         train_subsample_frac=None):
    """
    data_dir: ISIC root containing ISIC_2019_Training_Input/ (e.g. data/ISIC).
    If client_id is 0 or None, returns the pooled centralized splits.
    Otherwise returns the given client's (1-based) center shard + shared test pool.

    train_subsample_frac: if set (0, 1], randomly subsamples only the
    training rows (val/test stay full-size for stable, comparable metrics
    across HPO trials). Seeded for reproducibility across trials/clients.
    Intended for fast HPO search only — never use for real training runs.
    """
    if client_id in (0, None):
        train_rows, val_rows, test_rows = get_centralized_splits(manifest_csv)
    else:
        train_rows, val_rows, test_rows = get_client_splits(manifest_csv, client_id)

    if train_subsample_frac is not None:
        k = max(1, int(len(train_rows) * train_subsample_frac))
        train_rows = random.Random(42).sample(train_rows, k)

    train_ds = ISICDataset(data_dir, train_rows, build_transforms(train=True))
    val_ds = ISICDataset(data_dir, val_rows, build_transforms(train=False))
    test_ds = ISICDataset(data_dir, test_rows, build_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def compute_class_weights(manifest_csv, num_classes=NUM_CLASSES):
    """Inverse-frequency class weights from the pooled train rows (used for
    both the focal-loss `weight` and the weighted-CE alternative), so
    centralized and every FL client train with the same weighting."""
    train_rows, _, _ = get_centralized_splits(manifest_csv)
    counts = np.zeros(num_classes, dtype=np.float64)
    for r in train_rows:
        counts[int(r["target"])] += 1
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
