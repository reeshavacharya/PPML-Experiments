import os
import sys
import json
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_loader.fets_dataset import create_data_loaders
from models.swin_unetr import get_model
from utils.metrics_utils import (
    convert_to_brats_regions, compute_dice, compute_hausdorff95,
    compute_sensitivity, compute_specificity, compute_precision_metric,
    log_system_info,
)
from monai.losses import DiceCELoss


def test(data_dir="data/FeTS2022"):
    log_system_info(client_id="Centralized Baseline Test")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    abs_data_dir = os.path.join(project_root, data_dir)
    partitioning_csv = os.path.join(abs_data_dir, "MICCAI_FeTS2022_TrainingData", "partitioning_1.csv")
    if not os.path.exists(partitioning_csv):
        partitioning_csv = os.path.join(abs_data_dir, "partitioning_1.csv")

    _, _, test_loader = create_data_loaders(
        data_dir=abs_data_dir,
        partitioning_csv=partitioning_csv,
        client_id=0,
        batch_size=1,
        num_workers=4,
    )
    print(f"Test set size: {len(test_loader.dataset)} samples")

    model = get_model()

    best_model_path = os.path.join(project_root, "checkpoint", "swin_unetr_centralized_best.pth")
    if not os.path.exists(best_model_path):
        print(f"Error: Checkpoint not found at {best_model_path}")
        return

    print(f"Loading checkpoint from {best_model_path}...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix("module.")] = v
    model.load_state_dict(cleaned, strict=True)

    model.to(device)
    model.eval()

    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

    test_loss = 0.0
    test_steps = 0
    dice_lists = [[], [], []]
    hd95_lists = [[], [], []]
    sens_lists = [[], [], []]
    spec_lists = [[], [], []]
    prec_lists = [[], [], []]

    total = len(test_loader)
    log_interval = max(1, total // 10)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            labels_converted = convert_to_brats_regions(labels)

            outputs = model(images)
            loss = criterion(outputs, labels_converted)

            test_loss += loss.item()
            test_steps += 1

            outputs_binary = (torch.sigmoid(outputs) > 0.5).float()

            dice = compute_dice(outputs_binary, labels_converted).mean(dim=0).cpu().numpy()
            sens = compute_sensitivity(outputs_binary, labels_converted).mean(dim=0).cpu().numpy()
            spec = compute_specificity(outputs_binary, labels_converted).mean(dim=0).cpu().numpy()
            prec = compute_precision_metric(outputs_binary, labels_converted).mean(dim=0).cpu().numpy()
            hd95 = compute_hausdorff95(
                outputs_binary[0].cpu().numpy(),
                labels_converted[0].cpu().numpy(),
            )

            for c in range(3):
                dice_lists[c].append(float(dice[c]))
                sens_lists[c].append(float(sens[c]))
                spec_lists[c].append(float(spec[c]))
                prec_lists[c].append(float(prec[c]))
                if not np.isnan(hd95[c]):
                    hd95_lists[c].append(float(hd95[c]))

            if (i + 1) % log_interval == 0 or (i + 1) == total:
                pct = int(100 * (i + 1) / total)
                print(f"  Testing: {pct}% ({i + 1}/{total})", flush=True)

    avg_test_loss = test_loss / test_steps
    avg_dice = [np.mean(d) if d else 0 for d in dice_lists]
    avg_hd95 = [np.mean(h) if h else 0 for h in hd95_lists]
    avg_sens = [np.mean(s) if s else 0 for s in sens_lists]
    avg_spec = [np.mean(s) if s else 0 for s in spec_lists]
    avg_prec = [np.mean(p) if p else 0 for p in prec_lists]
    mean_dice = sum(avg_dice) / 3.0
    mean_hd95 = sum(avg_hd95) / 3.0
    region_names = ["wt", "tc", "et"]

    results = {
        "model": "swin_unetr_centralized",
        "checkpoint": best_model_path,
        "num_test_samples": len(test_loader.dataset),
        "test_loss": round(float(avg_test_loss), 4),
        "mean_dice": round(float(mean_dice), 4),
        "mean_hd95": round(float(mean_hd95), 2),
    }
    for i, r in enumerate(region_names):
        results[f"dice_{r}"] = round(float(avg_dice[i]), 4)
        results[f"hd95_{r}"] = round(float(avg_hd95[i]), 2)
        results[f"sensitivity_{r}"] = round(float(avg_sens[i]), 4)
        results[f"specificity_{r}"] = round(float(avg_spec[i]), 4)
        results[f"precision_{r}"] = round(float(avg_prec[i]), 4)

    print("\n" + "=" * 50)
    print("Centralized Baseline Test Results")
    print("=" * 50)
    print(f"  Samples       : {results['num_test_samples']}")
    print(f"  Test Loss     : {results['test_loss']}")
    print(f"  Mean Dice     : {results['mean_dice']}")
    print(f"    WT={results['dice_wt']}  TC={results['dice_tc']}  ET={results['dice_et']}")
    print(f"  Mean HD95     : {results['mean_hd95']}")
    print(f"    WT={results['hd95_wt']}  TC={results['hd95_tc']}  ET={results['hd95_et']}")
    print(f"  Sensitivity   : WT={results['sensitivity_wt']}  TC={results['sensitivity_tc']}  ET={results['sensitivity_et']}")
    print(f"  Specificity   : WT={results['specificity_wt']}  TC={results['specificity_tc']}  ET={results['specificity_et']}")
    print(f"  Precision     : WT={results['precision_wt']}  TC={results['precision_tc']}  ET={results['precision_et']}")
    print("=" * 50)

    output_path = os.path.join(project_root, "test", "centralized_baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    test()
