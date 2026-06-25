import os
import sys
import json
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_loader.fets_dataset import create_data_loaders
from models.swin_unetr import get_model
from utils.metrics_utils import convert_to_brats_regions, compute_dice, log_system_info
from monai.losses import DiceCELoss


def infer_client(client_id, model, device, data_dir, partitioning_csv, num_workers=4):
    _, _, test_loader = create_data_loaders(
        data_dir=data_dir,
        partitioning_csv=partitioning_csv,
        client_id=client_id,
        batch_size=1,
        num_workers=num_workers,
    )
    print(f"\n[Client {client_id}] Test set: {len(test_loader.dataset)} samples")

    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    test_loss = 0.0
    test_steps = 0
    wt_dice_list, tc_dice_list, et_dice_list = [], [], []

    total = len(test_loader)
    log_interval = max(1, total // 10)

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
            dice_scores = compute_dice(outputs_binary, labels_converted)
            mean_dice = dice_scores.mean(dim=0).cpu().numpy()
            wt_dice_list.append(float(mean_dice[0]))
            tc_dice_list.append(float(mean_dice[1]))
            et_dice_list.append(float(mean_dice[2]))

            if (i + 1) % log_interval == 0 or (i + 1) == total:
                pct = int(100 * (i + 1) / total)
                print(f"  [Client {client_id}] Testing: {pct}% ({i + 1}/{total})", flush=True)

    avg_loss = test_loss / test_steps
    avg_wt = np.mean(wt_dice_list)
    avg_tc = np.mean(tc_dice_list)
    avg_et = np.mean(et_dice_list)
    avg_dice = (avg_wt + avg_tc + avg_et) / 3.0

    return {
        "client_id": client_id,
        "num_test_samples": len(test_loader.dataset),
        "test_loss": round(float(avg_loss), 4),
        "dice_wt": round(float(avg_wt), 4),
        "dice_tc": round(float(avg_tc), 4),
        "dice_et": round(float(avg_et), 4),
        "mean_dice": round(float(avg_dice), 4),
    }


def main():
    log_system_info(client_id="FL Inference")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = os.path.join(project_root, "data", "FeTS2022")
    partitioning_csv = os.path.join(data_dir, "MICCAI_FeTS2022_TrainingData", "partitioning_1.csv")
    if not os.path.exists(partitioning_csv):
        partitioning_csv = os.path.join(data_dir, "partitioning_1.csv")

    # Find best FL checkpoint
    checkpoint_dir = os.path.join(project_root, "checkpoint")
    best_path = os.path.join(checkpoint_dir, "swin_unetr_fl_best.pth")
    if not os.path.exists(best_path):
        # Fallback to round 88 (known best from first training run)
        best_path = os.path.join(checkpoint_dir, "swin_unetr_fl_round_88.pth")
    if not os.path.exists(best_path):
        print(f"Error: No FL checkpoint found in {checkpoint_dir}")
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

    for client_id in [1, 2, 3]:
        results = infer_client(client_id, model, device, data_dir, partitioning_csv, num_workers=2)
        results["model"] = "swin_unetr_fl"
        results["checkpoint"] = best_path

        print(f"\n{'=' * 40}")
        print(f"Client {client_id} Test Results")
        print(f"{'=' * 40}")
        print(f"  Samples     : {results['num_test_samples']}")
        print(f"  Test Loss   : {results['test_loss']}")
        print(f"  Dice WT     : {results['dice_wt']}")
        print(f"  Dice TC     : {results['dice_tc']}")
        print(f"  Dice ET     : {results['dice_et']}")
        print(f"  Mean Dice   : {results['mean_dice']}")
        print(f"{'=' * 40}")

        out_path = os.path.join(output_dir, f"client_{client_id}_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
