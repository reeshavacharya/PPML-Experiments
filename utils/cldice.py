"""clDice (centerline Dice) — vessel-connectivity metric for thin tubular
structures like coronary arteries. Unlike volumetric Dice, clDice penalizes
broken/discontinuous predictions even when they overlap the ground truth mask
well, which is the failure mode that matters most for coronary segmentation.

Eval-time only (skeletonization isn't differentiable in the form used here),
computed on full-volume binary numpy arrays per case — same calling
convention as compute_hausdorff95 in utils/metrics_utils.py.
"""
import numpy as np
from skimage.morphology import skeletonize


def compute_cldice(pred_bin_np, true_bin_np):
    """
    pred_bin_np, true_bin_np: binary numpy arrays, shape (H, W, D) or (C, H, W, D).
    Returns clDice score (float) — mean over channels if a channel dim is present.
    """
    if pred_bin_np.ndim == 4:
        return float(np.mean([
            _cldice_single(pred_bin_np[c], true_bin_np[c])
            for c in range(pred_bin_np.shape[0])
        ]))
    return _cldice_single(pred_bin_np, true_bin_np)


def _cldice_single(pred, true, epsilon=1e-5):
    pred = pred.astype(bool)
    true = true.astype(bool)

    if not pred.any() and not true.any():
        return 1.0
    if not pred.any() or not true.any():
        return 0.0

    skel_pred = skeletonize(pred)
    skel_true = skeletonize(true)

    t_prec = (np.logical_and(skel_pred, true).sum() + epsilon) / (skel_pred.sum() + epsilon)
    t_sens = (np.logical_and(skel_true, pred).sum() + epsilon) / (skel_true.sum() + epsilon)

    return float(2 * t_prec * t_sens / (t_prec + t_sens))
