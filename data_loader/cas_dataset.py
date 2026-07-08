"""ImageCAS data loading: manifest-driven splits + MONAI transform pipeline.

Mirrors the shape of data_loader/fets_dataset.py (get_client_splits /
create_data_loaders) so call sites in train/, test/, and fl/ stay similar
across datasets. Key differences from FeTS:
  - Single-channel CCTA input, binary (background / coronary) label — no
    multi-region label conversion needed (ImageCAS classes are mutually
    exclusive, unlike overlapping BraTS regions), so DiceCELoss's built-in
    to_onehot_y=True handles it directly.
  - Native voxel spacing is anisotropic in-plane (~0.32-0.47mm, mean 0.354mm)
    with a fixed 0.5mm slice thickness (measured via cas_prepare.py inspect
    across a 30-volume sample). Resampling to 0.5mm iso only coarsens the
    in-plane axis to match the axis that's already fixed at 0.5mm.
  - Full volumes (512x512x200-275 native) don't fit a single forward pass
    even after resampling, so val/test loaders return whole (cropped, padded)
    volumes for sliding-window inference (see utils/inference_utils.py),
    not direct model(images) calls like the smaller BraTS crops.

The train/val/test split and the IID client sharding both come from a single
manifest CSV (data/CAS/imagecas_split.csv, built by
`python data_loader/cas_prepare.py split`) so centralized and federated runs
share byte-identical case assignments: centralized train/val is the union of
all clients' train/val shards, and test is one shared, undivided pool.
"""
import csv
import os
import random

# pandas/monai are imported lazily inside the functions that need them, so the
# pure-stdlib manifest-building path (build_split_manifest/write_manifest) can
# run in environments without the full GPU training stack installed.

EXTRACTED_DIRNAME = "_extracted"
PREFIX_RANGES = [(1, 200), (201, 400), (401, 600), (601, 800), (801, 1000)]

# Measured from a 30-volume sample via `cas_prepare.py inspect`: in-plane
# 0.318-0.465mm (mean 0.354), slice thickness fixed at 0.5mm.
RESAMPLE_SPACING = (0.5, 0.5, 0.5)
INTENSITY_WINDOW = (-200, 800)  # HU, typical CCTA soft-tissue/contrast window
PATCH_SIZE = (96, 96, 96)


def _prefix_for_id(subject_id):
    for start, end in PREFIX_RANGES:
        if start <= subject_id <= end:
            return f"{start}-{end}"
    raise ValueError(f"Subject ID {subject_id} outside known ImageCAS ID ranges 1-1000")


def resolve_case_paths(cas_dir, subject_id):
    """cas_dir is the CAS root (e.g. data/CAS). Returns (img_path, label_path)."""
    prefix = _prefix_for_id(subject_id)
    case_dir = os.path.join(cas_dir, EXTRACTED_DIRNAME, prefix, prefix)
    img_path = os.path.join(case_dir, f"{subject_id}.img.nii.gz")
    label_path = os.path.join(case_dir, f"{subject_id}.label.nii.gz")
    return img_path, label_path


def find_all_case_ids(cas_dir):
    """Scans the extracted tree; returns sorted subject IDs that have both img+label."""
    ids = []
    for start, end in PREFIX_RANGES:
        prefix = f"{start}-{end}"
        case_dir = os.path.join(cas_dir, EXTRACTED_DIRNAME, prefix, prefix)
        if not os.path.isdir(case_dir):
            continue
        for fname in os.listdir(case_dir):
            if fname.endswith(".img.nii.gz"):
                sid = int(fname[: -len(".img.nii.gz")])
                if os.path.exists(os.path.join(case_dir, f"{sid}.label.nii.gz")):
                    ids.append(sid)
    return sorted(ids)


def build_split_manifest(cas_dir, num_clients=5, seed=42, train_frac=0.7, val_frac=0.1):
    """Global 70/10/20 split of the 1000 cases, then IID-shards train and val
    across num_clients independently. Test is left undivided (Client_ID=0,
    shared/server-only). Returns a list of (Subject_ID, Split, Client_ID) rows.
    """
    ids = find_all_case_ids(cas_dir)
    if len(ids) != 1000:
        raise ValueError(f"Expected 1000 ImageCAS cases in {cas_dir}, found {len(ids)}")

    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    n_train = round(len(shuffled) * train_frac)
    n_val = round(len(shuffled) * val_frac)
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    def shard(id_list):
        shard_ids = id_list[:]
        rng.shuffle(shard_ids)
        per_client = len(shard_ids) // num_clients
        assignment = {}
        for i, sid in enumerate(shard_ids):
            assignment[sid] = min(i // per_client + 1, num_clients)
        return assignment

    train_assignment = shard(train_ids)
    val_assignment = shard(val_ids)

    rows = [(sid, "train", train_assignment[sid]) for sid in train_ids]
    rows += [(sid, "val", val_assignment[sid]) for sid in val_ids]
    rows += [(sid, "test", 0) for sid in test_ids]
    rows.sort(key=lambda r: r[0])
    return rows


def write_manifest(cas_dir, rows, filename="imagecas_split.csv"):
    path = os.path.join(cas_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject_ID", "Split", "Client_ID"])
        writer.writerows(rows)
    return path


def _to_data_dicts(cas_dir, subject_ids):
    data_dicts = []
    for sid in subject_ids:
        img_path, label_path = resolve_case_paths(cas_dir, int(sid))
        data_dicts.append({"image": img_path, "label": label_path})
    return data_dicts


def get_centralized_splits(cas_dir, manifest_csv):
    """Centralized train/val = union of all clients' shards; test = shared pool."""
    import pandas as pd
    df = pd.read_csv(manifest_csv)
    train_ids = df.loc[df["Split"] == "train", "Subject_ID"].tolist()
    val_ids = df.loc[df["Split"] == "val", "Subject_ID"].tolist()
    test_ids = df.loc[df["Split"] == "test", "Subject_ID"].tolist()
    return (
        _to_data_dicts(cas_dir, train_ids),
        _to_data_dicts(cas_dir, val_ids),
        _to_data_dicts(cas_dir, test_ids),
    )


def get_client_splits(cas_dir, manifest_csv, client_id):
    """Client's own IID train/val shard, plus the shared (undivided) test pool."""
    import pandas as pd
    df = pd.read_csv(manifest_csv)
    if client_id < 1:
        raise ValueError(f"client_id must be >= 1 for get_client_splits (got {client_id})")

    train_ids = df.loc[(df["Split"] == "train") & (df["Client_ID"] == client_id), "Subject_ID"].tolist()
    val_ids = df.loc[(df["Split"] == "val") & (df["Client_ID"] == client_id), "Subject_ID"].tolist()
    test_ids = df.loc[df["Split"] == "test", "Subject_ID"].tolist()

    if len(train_ids) == 0:
        raise ValueError(f"No training data found for client {client_id}")

    return (
        _to_data_dicts(cas_dir, train_ids),
        _to_data_dicts(cas_dir, val_ids),
        _to_data_dicts(cas_dir, test_ids),
    )


def create_data_loaders(data_dir, manifest_csv, client_id=None, batch_size=1, num_workers=4):
    """
    data_dir: CAS root containing _extracted/ (e.g. data/CAS).
    If client_id is 0 or None, returns the pooled centralized splits.
    Otherwise returns the given client's IID train/val shard + shared test pool.
    """
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        ScaleIntensityRanged,
        CropForegroundd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandRotate90d,
        RandShiftIntensityd,
        DivisiblePadd,
        ToTensord,
    )
    from monai.data import Dataset, DataLoader

    if client_id in (0, None):
        train_files, val_files, test_files = get_centralized_splits(data_dir, manifest_csv)
    else:
        train_files, val_files, test_files = get_client_splits(data_dir, manifest_csv, client_id)

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=RESAMPLE_SPACING, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys="image",
            a_min=INTENSITY_WINDOW[0], a_max=INTENSITY_WINDOW[1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=2, neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=RESAMPLE_SPACING, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys="image",
            a_min=INTENSITY_WINDOW[0], a_max=INTENSITY_WINDOW[1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        DivisiblePadd(keys=["image", "label"], k=32),
        ToTensord(keys=["image", "label"]),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    test_ds = Dataset(data=test_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
