import os
import sys
import json
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.swin_unetr import get_model
from utils.dataset_config import get_dataset_config, get_data_loader_module, resolve_manifest_path
from utils.metrics_utils import (
    compute_dice, compute_hausdorff95, compute_cldice,
    compute_sensitivity, compute_specificity, compute_precision_metric,
    log_system_info,
)
from utils.inference_utils import sliding_window_predict
from monai.losses import DiceCELoss


def test(dataset="fets", setting="centralized", data_dir=None, seed=42):
    """setting: "centralized" evaluates the centrally-trained checkpoint;
    "fl" evaluates the FL best-round checkpoint — same test loader, same
    metrics, so the two are directly comparable."""
    cfg = get_dataset_config(dataset)
    data_loader_module = get_data_loader_module(dataset)
    region_names = cfg["region_names"]
    n_regions = len(region_names)

    log_system_info(client_id=f"{setting.capitalize()} Test ({dataset})")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if data_dir is None:
        data_dir = cfg["default_data_dir"]
    abs_data_dir, manifest_path = resolve_manifest_path(project_root, data_dir, cfg)

    _, _, test_loader = data_loader_module.create_data_loaders(
        abs_data_dir, manifest_path, client_id=0, batch_size=1, num_workers=4,
    )
    print(f"Test set size: {len(test_loader.dataset)} samples")

    model = get_model(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"], feature_size=cfg["feature_size"])

    suffix = "" if dataset == "fets" else f"_{dataset}"
    seed_suffix = "" if seed == 42 else f"_seed{seed}"
    model_prefix = "swin_unetr_centralized" if setting == "centralized" else "swin_unetr_fl"
    best_model_path = os.path.join(project_root, "checkpoint", f"{model_prefix}{suffix}{seed_suffix}_best.pth")
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

    criterion = DiceCELoss(**cfg["loss_kwargs"])

    test_loss = 0.0
    test_steps = 0
    dice_lists = [[] for _ in range(n_regions)]
    hd95_lists = [[] for _ in range(n_regions)]
    sens_lists = [[] for _ in range(n_regions)]
    spec_lists = [[] for _ in range(n_regions)]
    prec_lists = [[] for _ in range(n_regions)]
    cldice_lists = [[] for _ in range(n_regions)] if cfg["include_cldice"] else None

    total = len(test_loader)
    log_interval = max(1, total // 10)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            labels_for_loss = cfg["loss_label_fn"](labels)

            outputs = sliding_window_predict(model, images) if cfg["sliding_window"] else model(images)
            loss = criterion(outputs, labels_for_loss)

            test_loss += loss.item()
            test_steps += 1

            pred_bin = cfg["metric_pred_fn"](outputs)
            label_bin = cfg["metric_label_fn"](labels)

            dice = compute_dice(pred_bin, label_bin).mean(dim=0).cpu().numpy()
            sens = compute_sensitivity(pred_bin, label_bin).mean(dim=0).cpu().numpy()
            spec = compute_specificity(pred_bin, label_bin).mean(dim=0).cpu().numpy()
            prec = compute_precision_metric(pred_bin, label_bin).mean(dim=0).cpu().numpy()
            hd95 = compute_hausdorff95(pred_bin[0].cpu().numpy(), label_bin[0].cpu().numpy())

            pred_np = pred_bin[0].cpu().numpy()
            label_np = label_bin[0].cpu().numpy()
            for c in range(n_regions):
                dice_lists[c].append(float(dice[c]))
                sens_lists[c].append(float(sens[c]))
                spec_lists[c].append(float(spec[c]))
                prec_lists[c].append(float(prec[c]))
                if not np.isnan(hd95[c]):
                    hd95_lists[c].append(float(hd95[c]))
                if cfg["include_cldice"]:
                    cldice_lists[c].append(compute_cldice(pred_np[c], label_np[c]))

            if (i + 1) % log_interval == 0 or (i + 1) == total:
                pct = int(100 * (i + 1) / total)
                print(f"  Testing: {pct}% ({i + 1}/{total})", flush=True)

    avg_test_loss = test_loss / test_steps
    avg_dice = [np.mean(d) if d else 0 for d in dice_lists]
    avg_hd95 = [np.mean(h) if h else 0 for h in hd95_lists]
    avg_sens = [np.mean(s) if s else 0 for s in sens_lists]
    avg_spec = [np.mean(s) if s else 0 for s in spec_lists]
    avg_prec = [np.mean(p) if p else 0 for p in prec_lists]
    mean_dice = sum(avg_dice) / n_regions
    mean_hd95 = sum(avg_hd95) / n_regions

    results = {
        "dataset": dataset,
        "setting": setting,
        "seed": seed,
        "model": f"{model_prefix}{suffix}",
        "checkpoint": best_model_path,
        "num_test_samples": len(test_loader.dataset),
        "test_loss": round(float(avg_test_loss), 4),
        "mean_dice": round(float(mean_dice), 4),
        "mean_hd95": round(float(mean_hd95), 2),
    }
    if cfg["include_cldice"]:
        avg_cldice = [np.mean(c) if c else 0 for c in cldice_lists]
        results["mean_cldice"] = round(float(sum(avg_cldice) / n_regions), 4)

    for i, r in enumerate(region_names):
        results[f"dice_{r}"] = round(float(avg_dice[i]), 4)
        results[f"hd95_{r}"] = round(float(avg_hd95[i]), 2)
        results[f"sensitivity_{r}"] = round(float(avg_sens[i]), 4)
        results[f"specificity_{r}"] = round(float(avg_spec[i]), 4)
        results[f"precision_{r}"] = round(float(avg_prec[i]), 4)
        if cfg["include_cldice"]:
            results[f"cldice_{r}"] = round(float(avg_cldice[i]), 4)

    print("\n" + "=" * 50)
    print(f"{setting.capitalize()} Test Results ({dataset})")
    print("=" * 50)
    print(f"  Samples       : {results['num_test_samples']}")
    print(f"  Test Loss     : {results['test_loss']}")
    print(f"  Mean Dice     : {results['mean_dice']}")
    print("    " + "  ".join(f"{r.upper()}={results[f'dice_{r}']}" for r in region_names))
    print(f"  Mean HD95     : {results['mean_hd95']}")
    print("    " + "  ".join(f"{r.upper()}={results[f'hd95_{r}']}" for r in region_names))
    if cfg["include_cldice"]:
        print(f"  Mean clDice   : {results['mean_cldice']}")
        print("    " + "  ".join(f"{r.upper()}={results[f'cldice_{r}']}" for r in region_names))
    print("  Sensitivity   : " + "  ".join(f"{r.upper()}={results[f'sensitivity_{r}']}" for r in region_names))
    print("  Specificity   : " + "  ".join(f"{r.upper()}={results[f'specificity_{r}']}" for r in region_names))
    print("  Precision     : " + "  ".join(f"{r.upper()}={results[f'precision_{r}']}" for r in region_names))
    print("=" * 50)

    result_prefix = "centralized_baseline" if setting == "centralized" else "fl_test"
    output_path = os.path.join(project_root, "test", f"{result_prefix}{suffix}{seed_suffix}_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fets", choices=["fets", "cas"])
    parser.add_argument("--setting", type=str, default="centralized", choices=["centralized", "fl"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Must match the --seed used for training this checkpoint")
    a = parser.parse_args()
    test(dataset=a.dataset, setting=a.setting, data_dir=a.data_dir, seed=a.seed)
