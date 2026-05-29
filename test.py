import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16

from dataset import NIHChestDataset, get_transforms, DISEASES


def sigmoid(x):
    return torch.sigmoid(x)


def compute_metrics(all_probs, all_targets, threshold=0.5):
    eps = 1e-8
    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1.0 - targets)).sum(dim=0)
    fn = ((1.0 - preds) * targets).sum(dim=0)
    tn = ((1.0 - preds) * (1.0 - targets)).sum(dim=0)

    per_class = {}
    class_f1s = []
    class_accuracies = []

    for idx, disease in enumerate(DISEASES):
        class_tp = tp[idx].item()
        class_fp = fp[idx].item()
        class_fn = fn[idx].item()
        class_tn = tn[idx].item()

        precision = class_tp / (class_tp + class_fp + eps)
        recall = class_tp / (class_tp + class_fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        accuracy = (class_tp + class_tn) / (class_tp + class_tn + class_fp + class_fn + eps)

        class_f1s.append(f1)
        class_accuracies.append(accuracy)
        per_class[disease] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "support": targets[:, idx].sum().item(),
        }

    micro_tp = tp.sum().item()
    micro_fp = fp.sum().item()
    micro_fn = fn.sum().item()
    micro_tn = tn.sum().item()

    micro_precision = micro_tp / (micro_tp + micro_fp + eps)
    micro_recall = micro_tp / (micro_tp + micro_fn + eps)
    micro_f1 = 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
    micro_accuracy = (micro_tp + micro_tn) / (micro_tp + micro_tn + micro_fp + micro_fn + eps)

    macro_precision = sum(v["precision"] for v in per_class.values()) / len(per_class)
    macro_recall = sum(v["recall"] for v in per_class.values()) / len(per_class)
    macro_f1 = sum(class_f1s) / len(class_f1s)
    macro_accuracy = sum(class_accuracies) / len(class_accuracies)

    subset_accuracy = (preds.eq(targets).all(dim=1)).float().mean().item()
    exact_match_rate = subset_accuracy
    label_accuracy = preds.eq(targets).float().mean().item()

    return {
        "threshold": threshold,
        "num_samples": int(targets.shape[0]),
        "num_labels": int(targets.shape[1]),
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "accuracy": micro_accuracy,
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "accuracy": macro_accuracy,
        },
        "subset_accuracy": subset_accuracy,
        "exact_match_rate": exact_match_rate,
        "label_accuracy": label_accuracy,
        "per_class": per_class,
    }


def main():
    data_dir = './data/NIH-CHEST'
    checkpoint_path = Path('vit_base_nih_best.pth')
    output_path = Path('test_results.json')
    batch_size = 8
    threshold = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path.resolve()}')

    _, test_transform = get_transforms()
    test_dataset = NIHChestDataset(data_dir, split='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = vit_b_16(weights=None)
    model.heads = nn.Sequential(nn.Linear(model.heads[0].in_features, len(DISEASES)))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            all_probs.append(sigmoid(outputs).cpu())
            all_targets.append(labels.cpu())

    avg_loss = total_loss / len(test_dataset)
    metrics = compute_metrics(all_probs, all_targets, threshold=threshold)
    metrics['loss'] = avg_loss
    metrics['checkpoint'] = str(checkpoint_path.resolve())
    metrics['dataset'] = str(Path(data_dir).resolve())

    output_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'Test loss: {avg_loss:.4f}')
    print(f'Wrote test results to {output_path.resolve()}')


if __name__ == '__main__':
    main()
