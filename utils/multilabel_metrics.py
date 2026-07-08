"""Multi-label classification metrics for NIH ChestX-ray14 (14 independent
binary findings per image).

Parallels utils/classification_metrics.py's role for ISIC's single-label
family: task-specific metric implementations kept in their own module,
imported directly by the multi-label train/FL/test call sites. All
functions take y_true (N, C) multi-hot arrays and y_prob (N, C) sigmoid
probabilities; the reported operating point is a fixed threshold (default
0.5) rather than a per-run derived optimum, so results stay comparable
across seeds/checkpoints.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score

DEFAULT_THRESHOLD = 0.5


def compute_per_class_auc(y_true, y_prob, num_classes):
    """Per-class one-vs-rest AUC. A label with no positive (or no negative)
    samples in this batch has undefined AUC — reported as 0.0 for that
    label (same convention as utils/classification_metrics.py's ISIC
    macro-AUC handling) rather than letting it silently become NaN."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    aucs = np.zeros(num_classes)
    for c in range(num_classes):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs[c] = roc_auc_score(col, y_prob[:, c])
        except ValueError:
            aucs[c] = 0.0
    return aucs


def compute_mean_auc_roc(y_true, y_prob, num_classes):
    """Mean AUC-ROC over all labels — the primary model-selection metric."""
    return float(np.mean(compute_per_class_auc(y_true, y_prob, num_classes)))


def compute_macro_f1(y_true, y_prob, threshold=DEFAULT_THRESHOLD):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_per_class_sensitivity(y_true, y_prob, num_classes, threshold=DEFAULT_THRESHOLD):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return recall_score(y_true, y_pred, average=None, zero_division=0)


def compute_all_metrics(y_true, y_prob, num_classes, threshold=DEFAULT_THRESHOLD):
    """y_true, y_prob: (N, num_classes). Returns a dict with every reported
    metric, JSON-serializable."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    per_class_auc = compute_per_class_auc(y_true, y_prob, num_classes)
    per_class_sens = compute_per_class_sensitivity(y_true, y_prob, num_classes, threshold)
    return {
        "mean_auc_roc": float(np.mean(per_class_auc)),
        "per_class_auc": [float(a) for a in per_class_auc],
        "macro_f1": float(compute_macro_f1(y_true, y_prob, threshold)),
        "per_class_sensitivity": [float(s) for s in per_class_sens],
        "threshold": threshold,
    }
