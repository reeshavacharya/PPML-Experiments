"""One-time ImageCAS data preparation: reassemble split archives, extract, and
inspect the resulting layout / voxel spacing before committing to a resample target.

Usage:
    python data_loader/cas_prepare.py extract [--prefixes 1-200,201-400,...]
    python data_loader/cas_prepare.py inspect [--sample 20]
    python data_loader/cas_prepare.py split [--num_clients 5] [--seed 42]

`extract` is CPU/IO-bound only (no GPU) but processes ~83GB of zips into an
estimated 150-250GB extracted — run it as a batch job on the cluster
(see data_loader/slurm/prepare_cas_data.sh), not on a login node.

`inspect` walks the extracted tree, prints the directory/file naming actually
used by the release (this varies between ImageCAS distributions), and reports
voxel spacing statistics across a sample of cases so the resample target in
cas_dataset.py can be chosen deliberately instead of assumed.

`split` builds data/CAS/imagecas_split.csv: a global 70/10/20 train/val/test
split of the 1000 cases (seed-fixed), with the train and val pools each
IID-sharded across --num_clients. Centralized pools ignore Client_ID (union of
all clients); FL filters by Client_ID. Test is a single shared, undivided pool
used identically by both settings — this mirrors how fets_dataset.py's
create_data_loaders(client_id=0) aggregates the per-client 80/10/10 splits
into the centralized set, just with the split done globally first since CAS
clients are synthetic IID shards rather than natural institutions.
"""
import argparse
import os
import shutil
import subprocess
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_loader.cas_dataset import build_split_manifest, write_manifest

CAS_DIR = os.path.join(project_root, "data", "CAS")
EXTRACT_DIR = os.path.join(CAS_DIR, "_extracted")

ALL_PREFIXES = ["1-200", "201-400", "401-600", "601-800", "801-1000"]


def extract_one(prefix, keep_intermediate=False):
    """Join <prefix>.change2zip + <prefix>.z01..zNN into a single zip and extract it."""
    change2zip = os.path.join(CAS_DIR, f"{prefix}.change2zip")
    if not os.path.exists(change2zip):
        print(f"[{prefix}] SKIP: {change2zip} not found", flush=True)
        return False

    dest = os.path.join(EXTRACT_DIR, prefix)
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"[{prefix}] Already extracted at {dest}, skipping.", flush=True)
        return True
    os.makedirs(dest, exist_ok=True)

    base_zip = os.path.join(CAS_DIR, f"{prefix}.zip")
    joined_zip = os.path.join(CAS_DIR, f"{prefix}.joined.zip")

    print(f"[{prefix}] Copying {change2zip} -> {base_zip}", flush=True)
    shutil.copyfile(change2zip, base_zip)

    print(f"[{prefix}] Joining split archive (zip -s 0) -> {joined_zip}", flush=True)
    result = subprocess.run(
        ["zip", "-s", "0", base_zip, "--out", joined_zip],
        cwd=CAS_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[{prefix}] zip join FAILED:\n{result.stdout}\n{result.stderr}", flush=True)
        return False

    print(f"[{prefix}] Extracting {joined_zip} -> {dest}", flush=True)
    result = subprocess.run(["unzip", "-q", joined_zip, "-d", dest], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[{prefix}] unzip FAILED:\n{result.stdout}\n{result.stderr}", flush=True)
        return False

    if not keep_intermediate:
        os.remove(base_zip)
        os.remove(joined_zip)

    print(f"[{prefix}] Done.", flush=True)
    return True


def cmd_extract(args):
    prefixes = args.prefixes.split(",") if args.prefixes else ALL_PREFIXES
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    ok = True
    for prefix in prefixes:
        ok = extract_one(prefix, keep_intermediate=args.keep_intermediate) and ok
    sys.exit(0 if ok else 1)


def cmd_inspect(args):
    import nibabel as nib

    if not os.path.isdir(EXTRACT_DIR):
        print(f"Nothing extracted yet at {EXTRACT_DIR}. Run the 'extract' subcommand first.")
        sys.exit(1)

    print("=" * 70)
    print("DIRECTORY SAMPLE (first 3 levels, first 20 entries per level)")
    print("=" * 70)
    for prefix in sorted(os.listdir(EXTRACT_DIR)):
        prefix_path = os.path.join(EXTRACT_DIR, prefix)
        if not os.path.isdir(prefix_path):
            continue
        entries = sorted(os.listdir(prefix_path))[:20]
        print(f"\n{prefix}/  ({len(os.listdir(prefix_path))} entries)")
        for e in entries:
            full = os.path.join(prefix_path, e)
            marker = "/" if os.path.isdir(full) else ""
            print(f"  {e}{marker}")
            if os.path.isdir(full):
                for e2 in sorted(os.listdir(full))[:10]:
                    print(f"    {e2}")

    print("\n" + "=" * 70)
    print("SEARCHING FOR .nii.gz FILES")
    print("=" * 70)
    nii_files = []
    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.endswith(".nii.gz"):
                nii_files.append(os.path.join(root, f))
    print(f"Found {len(nii_files)} .nii.gz files total.")
    for f in nii_files[:10]:
        print(f"  {os.path.relpath(f, EXTRACT_DIR)}")

    if not nii_files:
        print("No .nii.gz files found — check the directory sample above for the actual format.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"VOXEL SPACING (sample of {args.sample} volumes)")
    print("=" * 70)
    img_files = [f for f in nii_files if "label" not in os.path.basename(f).lower()]
    sample = img_files[:: max(1, len(img_files) // args.sample)][: args.sample]
    spacings = []
    shapes = []
    for f in sample:
        try:
            im = nib.load(f)
            zooms = im.header.get_zooms()[:3]
            spacings.append(zooms)
            shapes.append(im.shape[:3])
            print(f"  {os.path.relpath(f, EXTRACT_DIR):60s} spacing={tuple(round(z, 3) for z in zooms)} shape={im.shape[:3]}")
        except Exception as e:
            print(f"  {f}: FAILED to read ({e})")

    if spacings:
        import numpy as np
        arr = np.array(spacings)
        print(f"\nSpacing summary over {len(spacings)} volumes:")
        print(f"  min : {arr.min(axis=0).round(3)}")
        print(f"  mean: {arr.mean(axis=0).round(3)}")
        print(f"  max : {arr.max(axis=0).round(3)}")
        shape_arr = np.array(shapes)
        print(f"Shape summary: min={shape_arr.min(axis=0)} mean={shape_arr.mean(axis=0).round(0)} max={shape_arr.max(axis=0)}")
        print("\nCompare this against the planned 0.5mm iso resample target in cas_dataset.py before running training.")


def cmd_split(args):
    rows = build_split_manifest(
        CAS_DIR, num_clients=args.num_clients, seed=args.seed,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )
    path = write_manifest(CAS_DIR, rows)

    n_train = sum(1 for r in rows if r[1] == "train")
    n_val = sum(1 for r in rows if r[1] == "val")
    n_test = sum(1 for r in rows if r[1] == "test")
    print(f"Wrote manifest: {path}")
    print(f"  train={n_train}  val={n_val}  test={n_test}  (seed={args.seed})")
    for split_name, n in (("train", n_train), ("val", n_val)):
        per_client = {}
        for sid, split, client in rows:
            if split == split_name:
                per_client[client] = per_client.get(client, 0) + 1
        print(f"  {split_name} per client: {dict(sorted(per_client.items()))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Reassemble and extract the split zip archives")
    p_extract.add_argument("--prefixes", type=str, default=None,
                            help="Comma-separated subset, e.g. '1-200,201-400'. Default: all 5.")
    p_extract.add_argument("--keep_intermediate", action="store_true",
                            help="Keep the joined .zip copies after extraction (uses ~2x disk).")
    p_extract.set_defaults(func=cmd_extract)

    p_inspect = sub.add_parser("inspect", help="Report extracted layout and voxel spacing stats")
    p_inspect.add_argument("--sample", type=int, default=20, help="Number of volumes to check spacing for")
    p_inspect.set_defaults(func=cmd_inspect)

    p_split = sub.add_parser("split", help="Build the global 70/10/20 + IID client-sharded manifest CSV")
    p_split.add_argument("--num_clients", type=int, default=5)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--train_frac", type=float, default=0.7)
    p_split.add_argument("--val_frac", type=float, default=0.1)
    p_split.set_defaults(func=cmd_split)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
