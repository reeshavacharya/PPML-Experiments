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
    """Returns [(client_id, data_dict), ...] for all 23 institutions' test splits."""
    samples = []
    for cid in range(1, 24):
        _, _, test_files = get_client_splits(data_dir, partitioning_csv, cid)
        for f in test_files:
            samples.append((cid, f))
    return samples


def run_full_inference(samples, model, device):
    """
    Single inference pass over all samples. Returns a list of per-sample dicts
    that include the owning client_id and all region metrics.
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

            results.append({
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
            })

            if (i + 1) % log_interval == 0 or (i + 1) == total:
                pct = int(100 * (i + 1) / total)
                print(f"  [{i+1}/{total}] {pct}%  (inst {cid})", flush=True)

    return results


def aggregate(client_id, pool, sample_key):
    """
    Aggregates metrics over `pool` (a list of per-sample result dicts).
    `sample_key` is the field name written to the summary JSON for the sample count.
    Returns None if pool is empty.
    """
    n = len(pool)
    if n == 0:
        return None

    per = {m: [[], [], []] for m in ["dice", "hd95", "sens", "spec", "prec"]}
    total_loss = 0.0

    for r in pool:
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
        sample_key: n,
        "test_loss": round(total_loss / n, 4),
    }

    for key in ["dice", "hd95", "sens", "spec", "prec"]:
        for ci, region in enumerate(REGION_NAMES):
            summary[f"{key}_{region}"] = compute_metric_statistics(per[key][ci])

    all_dice = [v for ci in range(3) for v in per["dice"][ci]]
    summary["mean_dice"] = compute_metric_statistics(all_dice)

    all_hd95 = [v for ci in range(3) for v in per["hd95"][ci]]
    summary["mean_hd95"] = compute_metric_statistics(all_hd95)

    return summary


def global_summary_across_clients(client_summaries):
    """Aggregates per-client summary dicts into a federation-level summary."""
    gs = {"num_institutions": len(client_summaries)}
    for key in ["dice", "hd95", "sens", "spec", "prec"]:
        for region in REGION_NAMES:
            means = [s[f"{key}_{region}"]["mean"] for s in client_summaries if f"{key}_{region}" in s]
            gs[f"{key}_{region}"] = compute_metric_statistics(means)
    gs["mean_dice"] = compute_metric_statistics([s["mean_dice"]["mean"] for s in client_summaries])
    gs["mean_hd95"] = compute_metric_statistics([s["mean_hd95"]["mean"] for s in client_summaries])
    return gs


def print_client_summary(label, summary):
    print(f"\n{'=' * 50}")
    print(f"{label}")
    print(f"{'=' * 50}")
    print(f"  Test Loss  : {summary['test_loss']}")
    print(f"  Mean Dice  : {summary['mean_dice']['mean']:.4f}")
    for r in REGION_NAMES:
        d = summary[f"dice_{r}"]
        h = summary[f"hd95_{r}"]
        print(f"  {r.upper():>2}: Dice={d['mean']:.4f}±{d['std']:.4f}  HD95={h['mean']:.2f}±{h['std']:.2f}")
    print(f"{'=' * 50}")


def main():
    log_system_info(client_id="FL Global + Cross-Institutional Inference (FedPIDAvg)")

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

    # ── Single inference pass over all 137 test samples ──────────────────────
    print("\nCollecting test samples from all 23 institutions...")
    samples = collect_all_test_samples(data_dir, partitioning_csv)
    print(f"Total test samples: {len(samples)}")

    print("\nRunning inference...")
    all_results = run_full_inference(samples, model, device)

    global_summaries = []
    cross_inst_summaries = []

    for client_id in range(1, 24):
        # ── (1) Global pooled: all 137 samples ───────────────────────────────
        pooled = all_results
        g = aggregate(client_id, pooled, "num_pooled_samples")
        if g:
            g["model"] = "swin_unetr_fl_fedpidavg"
            g["checkpoint"] = best_path
            global_summaries.append(g)
            print_client_summary(
                f"Client {client_id} — global pooled ({g['num_pooled_samples']} samples)", g
            )
            with open(os.path.join(output_dir, f"global_client_{client_id}_results.json"), "w") as f:
                json.dump(g, f, indent=2)

        # ── (2) Cross-institutional: all samples except client_id's own ──────
        cross = [r for r in all_results if r["client_id"] != client_id]
        c = aggregate(client_id, cross, "num_cross_inst_samples")
        if c:
            c["model"] = "swin_unetr_fl_fedpidavg"
            c["checkpoint"] = best_path
            cross_inst_summaries.append(c)
            print_client_summary(
                f"Client {client_id} — cross-institutional ({c['num_cross_inst_samples']} samples)", c
            )
            with open(os.path.join(output_dir, f"cross_inst_client_{client_id}_results.json"), "w") as f:
                json.dump(c, f, indent=2)

    # ── Federation-level summaries ────────────────────────────────────────────
    if global_summaries:
        gs = global_summary_across_clients(global_summaries)
        path = os.path.join(output_dir, "global_pooled_results.json")
        with open(path, "w") as f:
            json.dump(gs, f, indent=2)
        print(f"\n[Global Pooled] Mean Dice={gs['mean_dice']['mean']:.4f}  Mean HD95={gs['mean_hd95']['mean']:.2f}mm  → {path}")

    if cross_inst_summaries:
        cs = global_summary_across_clients(cross_inst_summaries)
        path = os.path.join(output_dir, "cross_inst_global_results.json")
        with open(path, "w") as f:
            json.dump(cs, f, indent=2)
        print(f"[Cross-Inst.]   Mean Dice={cs['mean_dice']['mean']:.4f}  Mean HD95={cs['mean_hd95']['mean']:.2f}mm  → {path}")


if __name__ == "__main__":
    main()
