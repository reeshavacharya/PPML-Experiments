"""Aggregate multi-seed centralized/FL NIH ChestX-ray14 test results into mean +/- std.

Reads the per-seed result JSONs written by test/centralized_multilabel.py
(both "centralized" and "fl" settings) and reports mean +/- std for the
headline multi-label metrics. Parallels train/aggregate_seeds_isic.py, kept
as a separate script since the metric names/shapes are unrelated (per-label
AUC/sensitivity vs. ISIC's balanced-accuracy/kappa/per-class-recall).

Usage:
    python train/aggregate_seeds_nih.py --setting centralized --seeds 0 1 2 3 4
    python train/aggregate_seeds_nih.py --setting fl --seeds 0 1 2 3 4
"""
import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_loader.nih_chest_dataset import CLASS_NAMES
from utils.metrics_utils import compute_metric_statistics

SCALAR_METRICS = ["mean_auc_roc", "macro_f1"]


def _seed_suffix(seed):
    return "" if seed == 42 else f"_seed{seed}"


def _result_path(setting, seed):
    seed_suffix = _seed_suffix(seed)
    prefix = "centralized_nih" if setting == "centralized" else "fl_test_nih"
    return os.path.join(project_root, "test", f"{prefix}{seed_suffix}_results.json")


def load_results(setting, seeds):
    results = []
    missing = []
    for seed in seeds:
        path = _result_path(setting, seed)
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
        else:
            missing.append(path)
    if missing:
        print(f"WARNING: missing {len(missing)} result file(s):")
        for m in missing:
            print(f"  {m}")
    return results


def aggregate(setting, seeds):
    results = load_results(setting, seeds)

    if not results:
        print("No result files found — nothing to aggregate.")
        return None

    print(f"\nAggregating {len(results)}/{len(seeds)} seed(s) for nih/{setting}")
    print("=" * 70)

    summary = {"dataset": "nih", "setting": setting, "n_seeds": len(results), "seeds": seeds}

    for key in SCALAR_METRICS:
        values = [r[key] for r in results if key in r]
        if not values:
            continue
        stats = compute_metric_statistics(values)
        summary[key] = stats
        print(f"  {key:24s} mean={stats['mean']:.4f}  std={stats['std']:.4f}  (n={len(values)})")

    for c in CLASS_NAMES:
        for prefix in ("auc", "sensitivity"):
            key = f"{prefix}_{c}"
            values = [r[key] for r in results if key in r]
            if not values:
                continue
            stats = compute_metric_statistics(values)
            summary[key] = stats
            print(f"  {key:24s} mean={stats['mean']:.4f}  std={stats['std']:.4f}  (n={len(values)})")

    print("=" * 70)

    output_path = os.path.join(project_root, "test", f"aggregate_{setting}_nih_seeds.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved aggregate summary to {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", type=str, required=True, choices=["centralized", "fl"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    aggregate(args.setting, args.seeds)
