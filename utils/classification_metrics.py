"""Classification metrics for Fed-ISIC2019 (8-class, single-label).

Parallels utils/cldice.py's role for the segmentation family: task-specific
metric implementations kept in their own module, imported directly by the
classification train/FL/test call sites (utils/metrics_utils.py's
ResourceMonitor/log_system_info/append_metrics_csv are already task-agnostic
and are reused as-is, not duplicated here).

All functions take y_true (N,) int class indices and, where relevant, y_prob
(N, C) softmax probabilities or y_pred (N,) predicted class indices.
"""
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
)


def compute_balanced_accuracy(y_true, y_pred):
    """Normalized multi-class accuracy — the primary model-selection metric."""
    return balanced_accuracy_score(y_true, y_pred)


def compute_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_per_class_recall(y_true, y_pred, num_classes):
    return recall_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)


def compute_macro_auc_ovr(y_true, y_prob, num_classes):
    """Macro one-vs-rest AUC, computed per-class and averaged only over
    classes with both a positive and a negative sample present. sklearn's
    built-in multi_class="ovr" macro average folds an undefined per-class
    AUC (a class entirely absent from y_true) into a NaN for the *whole*
    macro average rather than raising — a near-certain occurrence on Fed-
    ISIC2019's smaller non-IID clients (e.g. one center only ever has 3 of
    the 8 classes), so it's handled explicitly here instead of relying on a
    try/except around the one-shot sklearn call."""
    y_true = np.asarray(y_true)
    aucs = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue  # class absent, or the only class present — AUC undefined
        try:
            aucs.append(roc_auc_score(y_bin, y_prob[:, c]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else 0.0


def compute_cohen_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)


def compute_confusion_matrix(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def compute_all_metrics(y_true, y_pred, y_prob, num_classes):
    """y_true, y_pred: (N,) int arrays. y_prob: (N, num_classes) softmax probs.
    Returns a dict with every reported metric (confusion matrix as a nested
    list so it's JSON-serializable)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    per_class_recall = compute_per_class_recall(y_true, y_pred, num_classes)
    return {
        "balanced_accuracy": float(compute_balanced_accuracy(y_true, y_pred)),
        "macro_f1": float(compute_macro_f1(y_true, y_pred)),
        "per_class_recall": [float(r) for r in per_class_recall],
        "macro_auc_ovr": float(compute_macro_auc_ovr(y_true, y_prob, num_classes)),
        "cohen_kappa": float(compute_cohen_kappa(y_true, y_pred)),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred, num_classes).tolist(),
    }
