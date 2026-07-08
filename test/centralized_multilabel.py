import os
import sys
import json

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.vit_classifier import get_model
from data_loader.nih_chest_dataset import (
    create_data_loaders, CLASS_NAMES, NUM_CLASSES, DEFAULT_DATA_DIR, DEFAULT_MANIFEST,
)
from utils.metrics_utils import log_system_info
from utils.multilabel_metrics import compute_all_metrics


def test(dataset="nih", setting="centralized", data_dir=None, seed=42, batch_size=32):
    """setting: "centralized" evaluates the centrally-trained checkpoint;
    "fl" evaluates the FL best-round checkpoint — same official test pool,
    same metrics, so the two are directly comparable."""
    log_system_info(client_id=f"{setting.capitalize()} Test ({dataset})")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    abs_data_dir = os.path.join(project_root, data_dir)
    manifest_path = os.path.join(abs_data_dir, DEFAULT_MANIFEST)

    _, _, test_loader = create_data_loaders(
        abs_data_dir, manifest_path, client_id=0, batch_size=batch_size, num_workers=4,
    )
    print(f"Test set size: {len(test_loader.dataset)} samples")

    model = get_model(num_classes=NUM_CLASSES, pretrained=False)

    seed_suffix = "" if seed == 42 else f"_seed{seed}"
    model_prefix = "vit_nih_centralized" if setting == "centralized" else "vit_nih_fl"
    best_model_path = os.path.join(project_root, "checkpoint", f"{model_prefix}{seed_suffix}_best.pth")
    if not os.path.exists(best_model_path):
        print(f"Error: Checkpoint not found at {best_model_path}")
        return

    print(f"Loading checkpoint from {best_model_path}...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    model.to(device)
    model.eval()

    all_labels, all_probs = [], []
    total = len(test_loader)
    log_interval = max(1, total // 10)

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

            if (i + 1) % log_interval == 0 or (i + 1) == total:
                pct = int(100 * (i + 1) / total)
                print(f"  Testing: {pct}% ({i + 1}/{total})", flush=True)

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    metrics = compute_all_metrics(all_labels, all_probs, NUM_CLASSES)

    results = {
        "dataset": dataset,
        "setting": setting,
        "seed": seed,
        "model": model_prefix,
        "checkpoint": best_model_path,
        "num_test_samples": len(test_loader.dataset),
        "mean_auc_roc": round(metrics["mean_auc_roc"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "threshold": metrics["threshold"],
    }
    for i, c in enumerate(CLASS_NAMES):
        results[f"auc_{c}"] = round(metrics["per_class_auc"][i], 4)
        results[f"sensitivity_{c}"] = round(metrics["per_class_sensitivity"][i], 4)

    print("\n" + "=" * 50)
    print(f"{setting.capitalize()} Test Results ({dataset})")
    print("=" * 50)
    print(f"  Samples          : {results['num_test_samples']}")
    print(f"  Mean AUC-ROC     : {results['mean_auc_roc']}")
    print(f"  Macro F1 (@{results['threshold']})   : {results['macro_f1']}")
    print("  Per-class AUC    : " + "  ".join(f"{c}={results[f'auc_{c}']}" for c in CLASS_NAMES))
    print("  Per-class Sens.  : " + "  ".join(f"{c}={results[f'sensitivity_{c}']}" for c in CLASS_NAMES))
    print("=" * 50)

    result_prefix = "centralized_nih" if setting == "centralized" else "fl_test_nih"
    output_path = os.path.join(project_root, "test", f"{result_prefix}{seed_suffix}_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nih", choices=["nih"])
    parser.add_argument("--setting", type=str, default="centralized", choices=["centralized", "fl"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42, help="Must match the --seed used for training this checkpoint")
    a = parser.parse_args()
    test(dataset=a.dataset, setting=a.setting, data_dir=a.data_dir, seed=a.seed, batch_size=a.batch_size)
