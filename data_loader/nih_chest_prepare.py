"""Builds the NIH ChestX-ray14 split manifest.

NIH ships an official *patient-wise* split (train_val_list.txt: 86,524 images
/ 28,008 patients; test_list.txt: 25,596 images / 2,797 patients; verified
zero patient overlap). That split is authoritative and used as-is here. What
it does *not* provide is a validation carve — and with ~3.64 images/patient,
carving val by image (like the legacy data_loader/nih_chest.py does) leaks
the same patient into both train and val. This script carves val patient-
wise instead, both for the pooled/centralized view and independently for
each of 5 IID FL client shards (mirrors data_loader/isic_prepare.py's
"two different splits of the same rows" pattern):
  - pooled_split: train/val/test — patient-wise ~90/10 carve of the 28,008
    train_val patients, for centralized training.
  - client_id: 1-5, an IID random patient shard (0 for test, shared/pooled).
  - client_split: train/val/test — each client's *own* patient-wise ~90/10
    carve of just its own shard, independent of the pooled carve and of
    every other client (mirrors ISIC's per-client val convention).

Usage:
    python data_loader/nih_chest_prepare.py --data_dir data/NIH-Chest --seed 42
"""
import argparse
import csv
import os
import random

DEFAULT_VAL_FRAC = 0.1
DEFAULT_NUM_CLIENTS = 5


def _split_patients(patients, val_frac, rng):
    """Returns the subset of `patients` assigned to val; the rest are train."""
    shuffled = list(patients)
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_frac)
    return set(shuffled[:n_val])


def build_manifest(entry_rows, train_val_images, test_images, num_clients=DEFAULT_NUM_CLIENTS,
                    val_frac=DEFAULT_VAL_FRAC, seed=42):
    """entry_rows: list of dict rows from Data_Entry_2017.csv.
    train_val_images/test_images: sets of Image Index strings from the
    official split files. Returns a list of output row dicts.
    """
    rng = random.Random(seed)
    img_to_row = {r["Image Index"]: r for r in entry_rows}

    train_val_rows = [img_to_row[img] for img in train_val_images if img in img_to_row]
    test_rows = [img_to_row[img] for img in test_images if img in img_to_row]

    patients = sorted({r["Patient ID"] for r in train_val_rows}, key=int)

    # Pooled (centralized) val carve: patient-wise ~90/10 of all train_val patients.
    pooled_val_patients = _split_patients(patients, val_frac, rng)
    for r in train_val_rows:
        r["pooled_split"] = "val" if r["Patient ID"] in pooled_val_patients else "train"
    for r in test_rows:
        r["pooled_split"] = "test"

    # IID client shard: round-robin over an independently shuffled patient order.
    shard_order = list(patients)
    rng.shuffle(shard_order)
    client_of = {pid: (i % num_clients) + 1 for i, pid in enumerate(shard_order)}
    for r in train_val_rows:
        r["client_id"] = client_of[r["Patient ID"]]
    for r in test_rows:
        r["client_id"] = 0

    # Each client's own patient-wise val carve — independent of the pooled
    # carve and of every other client (mirrors ISIC's per-client convention).
    patients_by_client = {c: [] for c in range(1, num_clients + 1)}
    for pid in patients:
        patients_by_client[client_of[pid]].append(pid)

    client_val_patients = {
        c: _split_patients(plist, val_frac, rng) for c, plist in patients_by_client.items()
    }
    for r in train_val_rows:
        c = r["client_id"]
        r["client_split"] = "val" if r["Patient ID"] in client_val_patients[c] else "train"
    for r in test_rows:
        r["client_split"] = "test"

    return train_val_rows + test_rows


def _to_output_row(r):
    return {
        "image": r["Image Index"],
        "finding_labels": r["Finding Labels"],
        "patient_id": r["Patient ID"],
        "pooled_split": r["pooled_split"],
        "client_id": r["client_id"],
        "client_split": r["client_split"],
    }


def write_manifest(rows, out_path):
    fieldnames = ["image", "finding_labels", "patient_id", "pooled_split", "client_id", "client_split"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_to_output_row(r) for r in rows)


def main():
    parser = argparse.ArgumentParser(description="Build the NIH ChestX-ray14 split manifest")
    parser.add_argument("--data_dir", type=str, default="data/NIH-Chest")
    parser.add_argument("--entry_csv", type=str, default="Data_Entry_2017.csv")
    parser.add_argument("--train_val_list", type=str, default="train_val_list.txt")
    parser.add_argument("--test_list", type=str, default="test_list.txt")
    parser.add_argument("--output", type=str, default="nih_split.csv")
    parser.add_argument("--num_clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--val_frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    entry_path = os.path.join(args.data_dir, args.entry_csv)
    with open(entry_path, newline="") as f:
        entry_rows = list(csv.DictReader(f))
    print(f"Loaded {len(entry_rows)} rows from {entry_path}")

    with open(os.path.join(args.data_dir, args.train_val_list)) as f:
        train_val_images = {l.strip() for l in f if l.strip()}
    with open(os.path.join(args.data_dir, args.test_list)) as f:
        test_images = {l.strip() for l in f if l.strip()}
    print(f"train_val: {len(train_val_images)} images | test: {len(test_images)} images")

    rows = build_manifest(
        entry_rows, train_val_images, test_images,
        num_clients=args.num_clients, val_frac=args.val_frac, seed=args.seed,
    )

    n_pooled_train = sum(1 for r in rows if r["pooled_split"] == "train")
    n_pooled_val = sum(1 for r in rows if r["pooled_split"] == "val")
    n_test = sum(1 for r in rows if r["pooled_split"] == "test")
    print(f"Pooled split: train={n_pooled_train} val={n_pooled_val} test={n_test}")

    for c in range(1, args.num_clients + 1):
        n_tr = sum(1 for r in rows if r["client_id"] == c and r["client_split"] == "train")
        n_va = sum(1 for r in rows if r["client_id"] == c and r["client_split"] == "val")
        print(f"  Client {c}: train={n_tr} val={n_va}")

    out_path = os.path.join(args.data_dir, args.output)
    write_manifest(rows, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
