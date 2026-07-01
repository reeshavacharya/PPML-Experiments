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
    compute_metric_statistics, log_system_info,
)
from monai.losses import DiceCELoss

REGION_NAMES = ["wt", "tc", "et"]


def infer_client(client_id, model, device, data_dir, partitioning_csv, num_workers=2):
    _, _, test_loader = create_data_loaders(
        data_dir=data_dir,
        partitioning_csv=partitioning_csv,
        client_id=client_id,
        batch_size=1,
        num_workers=num_workers,
    )
    num_samples = len(test_loader.dataset)
    if num_samples == 0:
        print(f"  [Client {client_id}] No test samples, skipping", flush=True)
        return None

    print(f"\n[Client {client_id}] Test set: {num_samples} samples")

    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    test_loss = 0.0
    test_steps = 0

    per_sample = {m: [[] for _ in range(3)] for m in ["dice", "hd95", "sens", "spec", "prec"]}

    total = len(test_loader)
    log_interval = max(1, total // 5)

    model.eval()
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
                per_sample["dice"][c].append(float(dice[c]))
                per_sample["sens"][c].append(float(sens[c]))
                per_sample["spec"][c].append(float(spec[c]))
                per_sample["prec"][c].append(float(prec[c]))
                if not np.isnan(hd95[c]):
                    per_sample["hd95"][c].append(float(hd95[c]))

            if (i + 1) % log_interval == 0 or (i + 1) == total:
                pct = int(100 * (i + 1) / total)
                print(f"  [Client {client_id}] {pct}% ({i+1}/{total})", flush=True)

    avg_loss = test_loss / test_steps

    results = {
        "client_id": client_id,
        "num_test_samples": num_samples,
        "test_loss": round(float(avg_loss), 4),
    }

    metric_keys = [("dice", "Dice"), ("hd95", "HD95"), ("sens", "Sensitivity"), ("spec", "Specificity"), ("prec", "Precision")]
    for key, _ in metric_keys:
        for c, r in enumerate(REGION_NAMES):
            results[f"{key}_{r}"] = compute_metric_statistics(per_sample[key][c])

    all_dice = []
    for c in range(3):
        all_dice.extend(per_sample["dice"][c])
    results["mean_dice"] = compute_metric_statistics(all_dice)

    all_hd95 = []
    for c in range(3):
        all_hd95.extend(per_sample["hd95"][c])
    results["mean_hd95"] = compute_metric_statistics(all_hd95)

    return results


def main():
    log_system_info(client_id="FL Inference (FedPIDAvg)")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = os.path.join(project_root, "data", "FeTS2022")
    partitioning_csv = os.path.join(data_dir, "MICCAI_FeTS2022_TrainingData", "partitioning_1.csv")
    if not os.path.exists(partitioning_csv):
        partitioning_csv = os.path.join(data_dir, "partitioning_1.csv")

    checkpoint_dir = os.path.join(project_root, "checkpoint")
    best_path = os.path.join(checkpoint_dir, "swin_unetr_fl_best.pth")
    if not os.path.exists(best_path):
        print(f"Error: No FL checkpoint found at {best_path}")
        return

    print(f"Loading checkpoint: {best_path}")
    model = get_model()
    state_dict = torch.load(best_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix("module.")] = v
    model.load_state_dict(cleaned, strict=True)
    model.to(device)

    output_dir = os.path.join(project_root, "fl", "slurm")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for client_id in range(1, 24):
        results = infer_client(client_id, model, device, data_dir, partitioning_csv)
        if results is None:
            continue

        results["model"] = "swin_unetr_fl_fedpidavg"
        results["checkpoint"] = best_path
        all_results.append(results)

        print(f"\n{'=' * 50}")
        print(f"Client {client_id} (Institution {client_id})")
        print(f"{'=' * 50}")
        print(f"  Samples     : {results['num_test_samples']}")
        print(f"  Test Loss   : {results['test_loss']}")
        print(f"  Mean Dice   : {results['mean_dice']['mean']}")
        for r in REGION_NAMES:
            d = results[f'dice_{r}']
            h = results[f'hd95_{r}']
            print(f"  {r.upper():>2}: Dice={d['mean']:.4f}±{d['std']:.4f}  HD95={h['mean']:.2f}±{h['std']:.2f}")
        print(f"{'=' * 50}")

        out_path = os.path.join(output_dir, f"client_{client_id}_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # Global cross-client summary
    if all_results:
        global_summary = {"num_institutions": len(all_results)}
        for key in ["dice", "hd95", "sens", "spec", "prec"]:
            for r in REGION_NAMES:
                per_client_means = [res[f"{key}_{r}"]["mean"] for res in all_results if f"{key}_{r}" in res]
                global_summary[f"{key}_{r}"] = compute_metric_statistics(per_client_means)

        all_mean_dice = [res["mean_dice"]["mean"] for res in all_results]
        global_summary["mean_dice"] = compute_metric_statistics(all_mean_dice)

        all_mean_hd95 = [res["mean_hd95"]["mean"] for res in all_results]
        global_summary["mean_hd95"] = compute_metric_statistics(all_mean_hd95)

        global_path = os.path.join(output_dir, "global_results.json")
        with open(global_path, "w") as f:
            json.dump(global_summary, f, indent=2)

        print(f"\n{'=' * 50}")
        print("Global Summary (across {len(all_results)} institutions)")
        print(f"{'=' * 50}")
        print(f"  Mean Dice: {global_summary['mean_dice']}")
        print(f"  Mean HD95: {global_summary['mean_hd95']}")
        print(f"  Saved to {global_path}")


if __name__ == "__main__":
    main()
