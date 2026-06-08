import json
from pathlib import Path
import torch
import sys

# Ensure root directory is in path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from datasets.medmnist_loader import DATASETS_CONFIG, get_dataloader
from medmnist.evaluator import getAUC, getACC

import importlib.util
spec = importlib.util.spec_from_file_location("ViT_Base", str(Path(__file__).resolve().parent.parent.parent / "models" / "ViT-Base.py"))
ViT_Base = importlib.util.module_from_spec(spec)
sys.modules["ViT_Base"] = ViT_Base
spec.loader.exec_module(ViT_Base)
from ViT_Base import build_vit_base

def evaluate_dataset(dataset_name, config, device, batch_size=8):
    checkpoint_path = Path("checkpoints") / "baseline" / f"vit_base_{dataset_name}_best.pth"
    if not checkpoint_path.exists():
        # Fallback to root if checkpoints weren't moved
        fallback_path = Path(f"vit_base_{dataset_name}_best.pth")
        if fallback_path.exists():
            checkpoint_path = fallback_path
        else:
            raise FileNotFoundError(f"Checkpoint not found for {dataset_name}: {checkpoint_path.resolve()}")

    test_loader, _ = get_dataloader(dataset_name, split='test', batch_size=batch_size)

    model = build_vit_base(config["num_classes"], is_dp=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_scores = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            if labels.ndim > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(1)

            if labels.ndim == 2 and labels.shape[1] > 1:
                scores = torch.sigmoid(outputs)
            else:
                scores = torch.softmax(outputs, dim=1)

            all_scores.append(scores.cpu())
            all_targets.append(labels.cpu())

    y_score = torch.cat(all_scores, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()

    task = test_loader.dataset.info["task"]
    auc = float(getAUC(y_true, y_score, task))
    acc = float(getACC(y_true, y_score, task))

    return {
        "dataset": dataset_name,
        "task": task,
        "split": "test",
        "checkpoint": str(checkpoint_path.resolve()),
        "data_root": str((Path("data")).resolve()),
        "num_samples": int(len(test_loader.dataset)),
        "AUC": auc,
        "ACC": acc,
    }


def main():
    output_dir = Path("experiments") / "test-results" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    summary = {}
    for dataset_name, config in DATASETS_CONFIG.items():
        try:
            metrics = evaluate_dataset(dataset_name, config, device, batch_size=batch_size)
            output_path = output_dir / f"{dataset_name}_test_results.json"
            output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            summary[dataset_name] = metrics
            print(f"{dataset_name}: AUC={metrics['AUC']:.4f} ACC={metrics['ACC']:.4f}")
            print(f"Wrote test results to {output_path.resolve()}")
        except FileNotFoundError as e:
            print(f"Skipping {dataset_name}: {e}")

    summary_path = output_dir / "all_datasets_test_results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote combined results to {summary_path.resolve()}")


if __name__ == '__main__':
    main()
