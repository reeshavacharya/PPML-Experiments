"""Dataset registry — the single place that knows about both FeTS2022 and
ImageCAS. Every script that supports `--dataset {fets,cas}` resolves this
dict instead of hardcoding a data loader module, label scheme, or FL
strategy, so adding a third dataset later means adding one entry here, not
editing every training/FL script.

Each entry exposes three small callables so eval/train loops stay dataset-
agnostic (loop over `region_names`, call these instead of branching):
  - loss_label_fn(label) -> tensor passed to the loss criterion as target.
  - metric_label_fn(label) -> binary tensor, channel count == len(region_names),
    used as the ground truth for compute_dice/compute_hausdorff95/etc.
  - metric_pred_fn(raw_model_output) -> binary tensor, same channel count,
    used as the prediction for those same metric functions.

FeTS's BraTS regions (WT/TC/ET) overlap, so loss and metrics both need the
same multi-channel sigmoid target (loss_label_fn is metric_label_fn).
ImageCAS's background/coronary classes are mutually exclusive, so the loss
uses the raw single-channel label directly (DiceCELoss(to_onehot_y=True)
one-hots it internally) while metrics just need it cast to float.
"""
import os

import torch

from utils.metrics_utils import convert_to_brats_regions

_fets_metric_pred_fn = lambda outputs: (torch.sigmoid(outputs) > 0.5).float()
_cas_metric_pred_fn = lambda outputs: (torch.argmax(outputs, dim=1, keepdim=True) == 1).float()
_cas_label_fn = lambda label: label.float()

DATASETS = {
    "fets": dict(
        module="data_loader.fets_dataset",
        in_channels=4,
        out_channels=3,
        feature_size=48,
        region_names=("wt", "tc", "et"),
        loss_label_fn=convert_to_brats_regions,
        metric_label_fn=convert_to_brats_regions,
        metric_pred_fn=_fets_metric_pred_fn,
        loss_kwargs=dict(to_onehot_y=False, sigmoid=True, squared_pred=True),
        default_data_dir="data/FeTS2022",
        default_manifest="MICCAI_FeTS2022_TrainingData/partitioning_1.csv",
        default_num_clients=23,
        default_strategy="fedpidavg",
        include_cldice=False,
        sliding_window=False,
    ),
    "cas": dict(
        module="data_loader.cas_dataset",
        in_channels=1,
        out_channels=2,
        feature_size=48,
        region_names=("coronary",),
        loss_label_fn=lambda label: label,
        metric_label_fn=_cas_label_fn,
        metric_pred_fn=_cas_metric_pred_fn,
        loss_kwargs=dict(to_onehot_y=True, softmax=True, squared_pred=True),
        default_data_dir="data/CAS",
        default_manifest="imagecas_split.csv",
        default_num_clients=5,
        default_strategy="fedavg",
        include_cldice=True,
        sliding_window=True,
    ),
}


def get_dataset_config(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choices: {list(DATASETS)}")
    return DATASETS[name]


def resolve_manifest_path(project_root, data_dir, cfg):
    """FeTS's partitioning CSV historically lived either inside
    MICCAI_FeTS2022_TrainingData/ or directly under data_dir; keep that
    fallback. CAS's manifest lives directly under data_dir."""
    abs_data_dir = os.path.join(project_root, data_dir)
    manifest = os.path.join(abs_data_dir, cfg["default_manifest"])
    if not os.path.exists(manifest):
        manifest = os.path.join(abs_data_dir, os.path.basename(cfg["default_manifest"]))
    return abs_data_dir, manifest


def get_data_loader_module(name):
    """Returns the imported data_loader module (fets_dataset or cas_dataset)."""
    import importlib
    return importlib.import_module(get_dataset_config(name)["module"])
