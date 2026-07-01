import os
import sys
import json
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_loader.fets_dataset import get_client_splits
from models.swin_unetr import get_model
from utils.metrics_utils import (
    convert_to_brats_regions, compute_dice, compute_hausdorff95,
    compute_sensitivity, compute_specificity, compute_precision_metric,
    compute_metric_statistics, log_system_info,
)
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    NormalizeIntensityd, CropForegroundd, DivisiblePadd, ToTensord,
)
from monai.data import Dataset, DataLoader
from monai.losses import DiceCELoss

REGION_NAMES = ["wt", "tc", "et"]

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    DivisiblePadd(keys=["image", "label"], k=32),
    ToTensord(keys=["image", "label"]),
])


def collect_all_test_samples(data_dir, partitioning_csv):
    """
    Returns a flat list of (client_id, data_dict) tuples for every test
    sample across all 23 institutions, using the same deterministic splits
    as inference.py.
    """
    samples = []
    for cid in range(1, 24):
        _, _, test_files = get_client_splits(data_dir, partitioning_csv, cid)
        for f in test_files:
            samples.append((cid, f))
    return samples


def run_full_inference(samples, model, device):
    """
    Runs inference on every (client_id, data_dict) sample and returns a list
    of dicts, each containing the owning client_id and per-region metrics.
    """
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    results = []
    total = len(samples)
    log_interval = max(1, total // 10)

    model.eval()
    with torch.no_grad():
        for i, (cid, data_dict) in enumerate(samples):
            ds = Dataset(data=[data_dict], transform=val_transforms)
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
            batch = next(iter(loader))

            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            labels_conv = convert_to_brats_regions(labels)

            outputs = model(images)
            loss = criterion(outputs, labels_conv)
            outputs_binary = (torch.sigmoid(outputs) > 0.5).float()

            dice = compute_dice(outputs_binary, labels_conv).mean(dim=0).cpu().numpy()
            sens = compute_sensitivity(outputs_binary, labels_conv).mean(dim=0).cpu().numpy()
            spec = compute_specificity(outputs_binary, labels_conv).mean(dim=0).cpu().numpy()
            prec = compute_precision_metric(outputs_binary, labels_conv).mean(dim=0).cpu().numpy()
            hd95 = compute_hausdorff95(
                outputs_binary[0].cpu().numpy(),
                labels_conv[0].cpu().numpy(),
            )

            entry = {
                "client_id": cid,
                "loss": float(loss.item()),
                "dice_wt": float(dice[0]),
                "dice_tc": float(dice[1]),
                "dice_et": float(dice[2]),
                "sens_wt": float(sens[0]),
                "sens_tc": float(sens[1]),
                "sens_et": float(sens[2]),
                "spec_wt": float(spec[0]),
                "spec_tc": float(spec[1]),
                "spec_et": float(spec[2]),
                "prec_wt": float(prec[0]),
                "prec_tc": float(prec[1]),
                "prec_et": float(prec[2]),
                "hd95_wt": float(hd95[0]) if not np.isnan(hd95[0]) else None,
                "hd95_tc": float(hd95[1]) if not np.isnan(hd95[1]) else None,
                "hd95_et": float(hd95[2]) if not np.isnan(hd95[2]) else None,
            }
            results.append(entry)

            if (i + 1) % log_interval == 0 or (i + 1) == total:
                pct = int(100 * (i + 1) / total)
                print(f"  [{i+1}/{total}] {pct}%  (inst {cid})", flush=True)

    return results


def aggregate_for_client(client_id, all_sample_results):
    """
    Given all per-sample results, aggregates only the samples NOT belonging
    to client_id (i.e., the cross-institutional test set for that client).
    Returns a summary dict in the same format as inference.py client JSONs.
    """
    cross = [r for r in all_sample_results if r["client_id"] != client_id]
    n = len(cross)
    if n == 0:
        return None

    per = {m: [[], [], []] for m in ["dice", "hd95", "sens", "spec", "prec"]}
    total_loss = 0.0

    for r in cross:
        total_loss += r["loss"]
        for ci, region in enumerate(REGION_NAMES):
            per["dice"][ci].append(r[f"dice_{region}"])
            per["sens"][ci].append(r[f"sens_{region}"])
            per["spec"][ci].append(r[f"spec_{region}"])
            per["prec"][ci].append(r[f"prec_{region}"])
            v = r[f"hd95_{region}"]
            if v is not None:
                per["hd95"][ci].append(v)

    summary = {
        "client_id": client_id,
        "num_cross_inst_samples": n,
        "test_loss": round(total_loss / n, 4),
    }

    metric_keys = [("dice", "dice"), ("hd95", "hd95"), ("sens", "sens"),
                   ("spec", "spec"), ("prec", "prec")]
    for key, _ in metric_keys:
        for ci, region in enumerate(REGION_NAMES):
            summary[f"{key}_{region}"] = compute_metric_statistics(per[key][ci])

    all_dice = []
    for ci in range(3):
        all_dice.extend(per["dice"][ci])
    summary["mean_dice"] = compute_metric_statistics(all_dice)

    all_hd95 = []
    for ci in range(3):
        all_hd95.extend(per["hd95"][ci])
    summary["mean_hd95"] = compute_metric_statistics(all_hd95)

    return summary


def main():
    log_system_info(client_id="FL Cross-Institutional Inference (FedPIDAvg)")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
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
    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    model.to(device)

    output_dir = os.path.join(project_root, "fl", "slurm")
    os.makedirs(output_dir, exist_ok=True)

    # --- Single inference pass over all 137 test samples ---
    print("\nCollecting test samples from all 23 institutions...")
    samples = collect_all_test_samples(data_dir, partitioning_csv)
    print(f"Total test samples: {len(samples)}")

    print("\nRunning inference on all test samples...")
    all_sample_results = run_full_inference(samples, model, device)

    # --- Per-client cross-institutional aggregation ---
    all_client_summaries = []

    for client_id in range(1, 24):
        summary = aggregate_for_client(client_id, all_sample_results)
        if summary is None:
            print(f"[Client {client_id}] No cross-institutional samples (skipped)")
            continue

        summary["model"] = "swin_unetr_fl_fedpidavg"
        summary["checkpoint"] = best_path
        all_client_summaries.append(summary)

        print(f"\n{'=' * 50}")
        print(f"Client {client_id} — cross-institutional ({summary['num_cross_inst_samples']} samples)")
        print(f"{'=' * 50}")
        print(f"  Test Loss  : {summary['test_loss']}")
        print(f"  Mean Dice  : {summary['mean_dice']['mean']:.4f}")
        for r in REGION_NAMES:
            d = summary[f"dice_{r}"]
            h = summary[f"hd95_{r}"]
            print(f"  {r.upper():>2}: Dice={d['mean']:.4f}±{d['std']:.4f}  HD95={h['mean']:.2f}±{h['std']:.2f}")
        print(f"{'=' * 50}")

        out_path = os.path.join(output_dir, f"cross_inst_client_{client_id}_results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    # --- Global cross-institutional summary (mean of per-client cross-inst means) ---
    if all_client_summaries:
        global_summary = {"num_institutions": len(all_client_summaries)}
        for key in ["dice", "hd95", "sens", "spec", "prec"]:
            for r in REGION_NAMES:
                per_client_means = [
                    s[f"{key}_{r}"]["mean"]
                    for s in all_client_summaries
                    if f"{key}_{r}" in s
                ]
                global_summary[f"{key}_{r}"] = compute_metric_statistics(per_client_means)

        all_mean_dice = [s["mean_dice"]["mean"] for s in all_client_summaries]
        global_summary["mean_dice"] = compute_metric_statistics(all_mean_dice)

        all_mean_hd95 = [s["mean_hd95"]["mean"] for s in all_client_summaries]
        global_summary["mean_hd95"] = compute_metric_statistics(all_mean_hd95)

        global_path = os.path.join(output_dir, "cross_inst_global_results.json")
        with open(global_path, "w") as f:
            json.dump(global_summary, f, indent=2)

        print(f"\n{'=' * 50}")
        print(f"Global Cross-Institutional Summary ({len(all_client_summaries)} institutions)")
        print(f"{'=' * 50}")
        print(f"  Mean Dice: {global_summary['mean_dice']['mean']:.4f} ± {global_summary['mean_dice']['std']:.4f}")
        print(f"  Mean HD95: {global_summary['mean_hd95']['mean']:.2f} ± {global_summary['mean_hd95']['std']:.2f}")
        print(f"  Saved to {global_path}")


if __name__ == "__main__":
    main()
