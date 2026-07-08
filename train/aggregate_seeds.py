"""Aggregate multi-seed centralized/FL test results into mean +/- std.

Reads the per-seed result JSONs written by test/centralized_baseline.py
(centralized) and the FL equivalent, and reports mean +/- std for the
headline metrics (Dice, HD95, clDice, sensitivity, precision) per region.

Usage:
    python train/aggregate_seeds.py --dataset cas --setting centralized --seeds 0 1 2 3 4
    python train/aggregate_seeds.py --dataset cas --setting fl --seeds 0 1 2 3 4
"""
import argparse
import glob
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.dataset_config import get_dataset_config
from utils.metrics_utils import compute_metric_statistics

METRIC_PREFIXES = ["dice", "hd95", "cldice", "sensitivity", "precision"]


def _seed_suffix(seed):
    return "" if seed == 42 else f"_seed{seed}"


def _result_path(dataset, setting, seed):
    suffix = "" if dataset == "fets" else f"_{dataset}"
    seed_suffix = _seed_suffix(seed)
    if setting == "centralized":
        return os.path.join(project_root, "test", f"centralized_baseline{suffix}{seed_suffix}_results.json")
    # FL results are written per-run by fl/server.py's --result_path or training_summary;
    # this expects a matching fl_test_{dataset}{seed_suffix}_results.json produced by
    # running test-time evaluation of the FL best checkpoint (same shape as centralized).
    return os.path.join(project_root, "test", f"fl_test{suffix}{seed_suffix}_results.json")


def load_results(dataset, setting, seeds):
    results = []
    missing = []
    for seed in seeds:
        path = _result_path(dataset, setting, seed)
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


def aggregate(dataset, setting, seeds):
    cfg = get_dataset_config(dataset)
    region_names = cfg["region_names"]
    results = load_results(dataset, setting, seeds)

    if not results:
        print("No result files found — nothing to aggregate.")
        return None

    print(f"\nAggregating {len(results)}/{len(seeds)} seed(s) for {dataset}/{setting}")
    print("=" * 70)

    summary = {"dataset": dataset, "setting": setting, "n_seeds": len(results), "seeds": seeds}
    for prefix in METRIC_PREFIXES:
        for region in region_names:
            key = f"{prefix}_{region}"
            values = [r[key] for r in results if key in r]
            if not values:
                continue
            stats = compute_metric_statistics(values)
            summary[key] = stats
            print(f"  {key:24s} mean={stats['mean']:.4f}  std={stats['std']:.4f}  (n={len(values)})")

    mean_key = "mean_dice"
    mean_values = [r[mean_key] for r in results if mean_key in r]
    if mean_values:
        stats = compute_metric_statistics(mean_values)
        summary[mean_key] = stats
        print(f"  {mean_key:24s} mean={stats['mean']:.4f}  std={stats['std']:.4f}  (n={len(mean_values)})")

    print("=" * 70)

    suffix = "" if dataset == "fets" else f"_{dataset}"
    output_path = os.path.join(project_root, "test", f"aggregate_{setting}{suffix}_seeds.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved aggregate summary to {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="fets", choices=["fets", "cas"])
    parser.add_argument("--setting", type=str, required=True, choices=["centralized", "fl"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    aggregate(args.dataset, args.setting, args.seeds)
