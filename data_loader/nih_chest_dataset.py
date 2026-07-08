"""NIH ChestX-ray14 data loading: manifest-driven splits + torchvision
transform pipeline for ViT-B/16 multi-label classification.

Mirrors the shape of data_loader/isic_dataset.py (get_centralized_splits /
get_client_splits / create_data_loaders), but for a genuinely different task
shape: 14 independent binary findings per image (sigmoid, multi-hot target)
rather than ISIC's 8 mutually-exclusive classes (softmax, single-label).

The train/val split comes from data/NIH-Chest/nih_split.csv (built by
`python data_loader/nih_chest_prepare.py`), which carves validation
patient-wise from NIH's official train_val_list.txt/test_list.txt patient-
level split — carving by image instead (as data_loader/nih_chest.py, the
legacy loader, does) leaks the same patient's ~3.64 images across train/val
and inflates AUC. Test is always the official, untouched 25,596-image pool.
"""
import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_SIZE = 224
NUM_CLASSES = 14
NUM_CLIENTS = 5
CLASS_NAMES = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
)
_CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_DATA_DIR = "data/NIH-Chest"
DEFAULT_MANIFEST = "nih_split.csv"
NUM_IMAGE_FOLDERS = 12


def build_transforms(train):
    # ToTensor runs before the geometric/color ops (not after, as
    # torchvision examples often show) — the PIL-backed color-jitter path
    # has a known numpy>=2 uint8-overflow bug (hit and fixed for ISIC's
    # ColorJitter(hue=...)); running these ops on tensors avoids it entirely.
    ops = [transforms.Resize(256), transforms.ToTensor()]
    if train:
        ops += [
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),  # rotation/translation
            transforms.ColorJitter(brightness=0.2),  # brightness
            # No horizontal flip: chest anatomy (heart/aorta position) is
            # not left-right symmetric, so flipping risks teaching a
            # physically wrong prior rather than a useful invariance.
        ]
    else:
        ops += [transforms.CenterCrop(IMG_SIZE)]
    ops += [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(ops)


def _encode_labels(finding_labels):
    labels = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    if finding_labels and finding_labels != "No Finding":
        for name in finding_labels.split("|"):
            name = name.strip()
            if name in _CLASS_INDEX:
                labels[_CLASS_INDEX[name]] = 1.0
    return labels


def build_image_index(data_dir):
    """Scans images_001..images_012/images/ once; returns {filename: full_path}.
    Shared across train/val/test datasets rather than rescanning per split
    (the legacy data_loader/nih_chest.py rescans once per split)."""
    index = {}
    for i in range(1, NUM_IMAGE_FOLDERS + 1):
        folder = os.path.join(data_dir, f"images_{i:03d}", "images")
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                index[fname] = os.path.join(folder, fname)
    return index


class NIHChestDataset(Dataset):
    def __init__(self, rows, image_index, transform):
        self.rows = rows
        self.image_index = image_index
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = self.image_index[row["image"]]
        image = Image.open(img_path).convert("RGB")  # grayscale PNG -> 3 identical channels
        image = self.transform(image)
        labels = _encode_labels(row["finding_labels"])
        return image, labels


def _read_manifest(manifest_csv):
    with open(manifest_csv, newline="") as f:
        return list(csv.DictReader(f))


def get_centralized_splits(manifest_csv):
    """Centralized train/val = pooled patient-wise ~90/10 carve of the
    28,008 train_val patients; test = the official 25,596-image pool."""
    rows = _read_manifest(manifest_csv)
    train_rows = [r for r in rows if r["pooled_split"] == "train"]
    val_rows = [r for r in rows if r["pooled_split"] == "val"]
    test_rows = [r for r in rows if r["pooled_split"] == "test"]
    return train_rows, val_rows, test_rows


def get_client_splits(manifest_csv, client_id):
    """Client's own IID patient shard, further split ~90/10 (patient-wise,
    independent of every other client and of the pooled carve), plus the
    shared (undivided) official test pool."""
    if client_id < 1:
        raise ValueError(f"client_id must be >= 1 for get_client_splits (got {client_id})")

    rows = _read_manifest(manifest_csv)
    client_id_str = str(client_id)
    train_rows = [r for r in rows if r["client_id"] == client_id_str and r["client_split"] == "train"]
    val_rows = [r for r in rows if r["client_id"] == client_id_str and r["client_split"] == "val"]
    test_rows = [r for r in rows if r["pooled_split"] == "test"]

    if not train_rows:
        raise ValueError(f"No training data found for client {client_id}")

    return train_rows, val_rows, test_rows


def create_data_loaders(data_dir, manifest_csv, client_id=None, batch_size=32, num_workers=4,
                         train_subsample_frac=None):
    """
    data_dir: NIH-Chest root containing images_001..images_012/ (e.g. data/NIH-Chest).
    If client_id is 0 or None, returns the pooled centralized splits.
    Otherwise returns the given client's (1-based) IID shard + shared test pool.

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

    image_index = build_image_index(data_dir)

    train_ds = NIHChestDataset(train_rows, image_index, build_transforms(train=True))
    val_ds = NIHChestDataset(val_rows, image_index, build_transforms(train=False))
    test_ds = NIHChestDataset(test_rows, image_index, build_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def compute_pos_weight(manifest_csv, num_classes=NUM_CLASSES):
    """Per-class (N_negative / N_positive) from the pooled train rows, for
    torch.nn.BCEWithLogitsLoss(pos_weight=...) — the positive-weighting the
    ChestX-ray14 paper itself found improved AUC, especially on rare classes
    (e.g. Hernia, Cardiomegaly, Pneumonia)."""
    train_rows, _, _ = get_centralized_splits(manifest_csv)
    pos_counts = np.zeros(num_classes, dtype=np.float64)
    for r in train_rows:
        labels = _encode_labels(r["finding_labels"]).numpy()
        pos_counts += labels
    total = len(train_rows)
    pos_counts = np.clip(pos_counts, 1, None)
    pos_weight = (total - pos_counts) / pos_counts
    return torch.tensor(pos_weight, dtype=torch.float32)
