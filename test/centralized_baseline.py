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

    # Strip DataParallel 'module.' prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix("module.")] = v
    model.load_state_dict(cleaned, strict=True)

    model.to(device)
    model.eval()

    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

    test_loss = 0.0
    test_steps = 0
    wt_dice_list, tc_dice_list, et_dice_list = [], [], []

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
            dice_scores = compute_dice(outputs_binary, labels_converted)
            mean_dice = dice_scores.mean(dim=0).cpu().numpy()
            wt_dice_list.append(float(mean_dice[0]))
            tc_dice_list.append(float(mean_dice[1]))
            et_dice_list.append(float(mean_dice[2]))

            if (i + 1) % log_interval == 0 or (i + 1) == total:
                pct = int(100 * (i + 1) / total)
                print(f"  Testing: {pct}% ({i + 1}/{total})", flush=True)

    avg_test_loss = test_loss / test_steps
    avg_wt = np.mean(wt_dice_list)
    avg_tc = np.mean(tc_dice_list)
    avg_et = np.mean(et_dice_list)
    mean_dice = (avg_wt + avg_tc + avg_et) / 3.0

    results = {
        "model": "swin_unetr_centralized",
        "checkpoint": best_model_path,
        "num_test_samples": len(test_loader.dataset),
        "test_loss": round(float(avg_test_loss), 4),
        "dice_wt": round(float(avg_wt), 4),
        "dice_tc": round(float(avg_tc), 4),
        "dice_et": round(float(avg_et), 4),
        "mean_dice": round(float(mean_dice), 4),
    }

    print("\n" + "=" * 40)
    print("Centralized Baseline Test Results")
    print("=" * 40)
    print(f"  Test Loss   : {results['test_loss']}")
    print(f"  Dice WT     : {results['dice_wt']}")
    print(f"  Dice TC     : {results['dice_tc']}")
    print(f"  Dice ET     : {results['dice_et']}")
    print(f"  Mean Dice   : {results['mean_dice']}")
    print("=" * 40)

    output_path = os.path.join(project_root, "test", "centralized_baseline_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    test()
