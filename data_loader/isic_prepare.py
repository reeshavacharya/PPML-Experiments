"""Builds the Fed-ISIC2019 val split manifest.

data/ISIC/train_test_split.csv already gives the pooled 18,597/4,650 train/test
split (its `fold` column) and the 6 natural FL centers (`center`, 0-5), but no
validation split. The centralized and FL settings need *different* val carves
from the same 18,597 train rows:
  - Centralized: one stratified 80/20 split of the pooled train rows.
  - FL: each of the 6 centers stratified 80/20 *independently* (a client only
    ever sees its own center's rows).
These are genuinely different partitions of the same rows (pooled-stratified
!= union of per-center-stratified), so both are precomputed once here into
two columns and written to data/ISIC/isic_split.csv. Every centralized run,
every FL client, HPO trial, and resumed run then reads this single manifest,
so all of them share byte-identical assignments for a given --seed.

Usage:
    python data_loader/isic_prepare.py --data_dir data/ISIC --seed 42
"""
import argparse
import csv
import os

from sklearn.model_selection import train_test_split

DEFAULT_VAL_FRAC = 0.2


def _stratified_val_split(rows, val_frac, seed):
    """rows: list of dict rows (already filtered to fold=="train").
    Returns a dict row-index -> "train"/"val". Falls back to a non-stratified
    split if any class has too few members to stratify (needs >=2 per class).
    """
    indices = list(range(len(rows)))
    targets = [r["target"] for r in rows]
    try:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_frac, random_state=seed, stratify=targets,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_frac, random_state=seed, stratify=None,
        )
    assignment = {i: "train" for i in train_idx}
    assignment.update({i: "val" for i in val_idx})
    return assignment


def build_manifest(rows, val_frac=DEFAULT_VAL_FRAC, seed=42):
    """rows: list of dicts from train_test_split.csv (all 23,247 rows).
    Adds "pooled_split" (centralized: train/val/test) and "client_split"
    (FL: train/val/test, stratified within each row's own center).
    """
    train_rows = [r for r in rows if r["fold"] == "train"]
    test_rows = [r for r in rows if r["fold"] == "test"]

    pooled_assignment = _stratified_val_split(train_rows, val_frac, seed)
    for i, r in enumerate(train_rows):
        r["pooled_split"] = pooled_assignment[i]
    for r in test_rows:
        r["pooled_split"] = "test"

    centers = sorted(set(r["center"] for r in train_rows), key=int)
    for center in centers:
        center_rows = [r for r in train_rows if r["center"] == center]
        center_assignment = _stratified_val_split(center_rows, val_frac, seed)
        for i, r in enumerate(center_rows):
            r["client_split"] = center_assignment[i]
    for r in test_rows:
        r["client_split"] = "test"

    return train_rows + test_rows


def write_manifest(rows, out_path, fieldnames):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build the Fed-ISIC2019 val split manifest")
    parser.add_argument("--data_dir", type=str, default="data/ISIC")
    parser.add_argument("--input", type=str, default="train_test_split.csv")
    parser.add_argument("--output", type=str, default="isic_split.csv")
    parser.add_argument("--val_frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = os.path.join(args.data_dir, args.input)
    out_path = os.path.join(args.data_dir, args.output)

    with open(in_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) + ["pooled_split", "client_split"]
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {in_path}")
    rows = build_manifest(rows, val_frac=args.val_frac, seed=args.seed)

    n_pooled_val = sum(1 for r in rows if r["pooled_split"] == "val")
    n_client_val = sum(1 for r in rows if r["client_split"] == "val")
    print(f"Pooled split:  train={sum(1 for r in rows if r['pooled_split']=='train')} "
          f"val={n_pooled_val} test={sum(1 for r in rows if r['pooled_split']=='test')}")
    print(f"Client split:  train={sum(1 for r in rows if r['client_split']=='train')} "
          f"val={n_client_val} test={sum(1 for r in rows if r['client_split']=='test')}")

    write_manifest(rows, out_path, fieldnames)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
