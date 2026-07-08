"""Optuna HPO for centralized training — Swin UNETR (FeTS 2022 / ImageCAS) or
ViT-B/16 (Fed-ISIC2019 / NIH ChestX-ray14), dispatched by --dataset.

Usage:
    python train/hpo_centralized.py --dataset fets [--n_trials 20] [--hpo_epochs 15] [--data_dir data/FeTS2022]
    python train/hpo_centralized.py --dataset isic [--n_trials 20] [--hpo_epochs 15]
    python train/hpo_centralized.py --dataset nih  [--n_trials 20] [--hpo_epochs 15]

Results are persisted in optuna_centralized{_dataset}.db (SQLite). Interrupted studies can
be resumed by re-running with the same --study_name — Optuna skips completed trials.
"""
import argparse
import json
import os
import sys

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def _get_run_training(dataset):
    if dataset == "isic":
        from train.centralized_classification import run_training
    elif dataset == "nih":
        from train.centralized_multilabel import run_training
    else:
        from train.centralized_baseline import run_training
    return run_training


_METRIC_NAMES = {"isic": "val balanced accuracy", "nih": "val mean AUC-ROC"}


def build_objective(args):
    run_training = _get_run_training(args.dataset)
    metric_name = _METRIC_NAMES.get(args.dataset, "val mean Dice")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        print(
            f"\n[Trial {trial.number}] lr={lr:.2e}  weight_decay={weight_decay:.2e}",
            flush=True,
        )

        kwargs = dict(
            dataset=args.dataset,
            epochs=args.hpo_epochs,
            lr=lr,
            weight_decay=weight_decay,
            data_dir=args.data_dir,
            save_checkpoint=False,
            trial=trial,
            max_steps=args.hpo_max_steps,
        )
        # Segmentation batch_size is fixed at 1 (3D volume memory constraints);
        # classification (isic/nih) can use a real batch size.
        kwargs["batch_size"] = args.batch_size if args.dataset in ("isic", "nih") else 1
        # Training-set subsampling (isic/nih only — centralized_baseline.py's
        # run_training doesn't accept this kwarg): stacks with hpo_epochs/
        # max_steps to shrink per-trial wall time further. Val/test are never
        # subsampled, so trial metrics stay comparable.
        if args.dataset in ("isic", "nih") and args.hpo_subsample_frac is not None:
            kwargs["train_subsample_frac"] = args.hpo_subsample_frac

        val_metric = run_training(**kwargs)

        print(f"[Trial {trial.number}] Final {metric_name}: {val_metric:.4f}", flush=True)
        return val_metric

    return objective


# Warm-start: seed the very first trial with hyperparameters already known to
# work well for that exact dataset/model from a prior HPO run (ISIC/NIH only
# — fets/cas keep the original cold-start behavior). Only enqueued on a
# genuinely fresh study (len(study.trials) == 0) so re-running/resuming this
# script doesn't keep re-adding the seed trial.
_WARM_START_PARAMS = {
    "isic": {"lr": 2.0513382630874486e-05, "weight_decay": 2.9375384576328313e-06},  # best of isic's own completed centralized HPO
    "nih": {"lr": 1.0994335574766187e-05, "weight_decay": 0.0008123245085588687},    # best so far of nih's own centralized HPO
    "cas": {"lr": 1e-4, "weight_decay": 1e-5},                                       # MONAI SwinUNETR recommended defaults for a new dataset
}

# ISIC/NIH use a much smaller n_startup_trials than fets: with only ~10
# trials total, exempting 3 of them from pruning (the original default)
# wastes a large fraction of the time budget on trials that may be obviously
# bad within the first few epochs. CAS uses n_startup_trials=5 (half the
# budget) to allow sufficient exploration before pruning begins.
_PRUNER_KWARGS = {
    "isic": dict(n_startup_trials=2, n_warmup_steps=3, interval_steps=1),
    "nih": dict(n_startup_trials=2, n_warmup_steps=3, interval_steps=1),
    "cas": dict(n_startup_trials=2, n_warmup_steps=1, interval_steps=1),
}
_DEFAULT_PRUNER_KWARGS = dict(n_startup_trials=3, n_warmup_steps=3)


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for centralized training (SwinUNETR or ViT-B/16)")
    parser.add_argument("--dataset", type=str, default="fets", choices=["fets", "cas", "isic", "nih"])
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--hpo_epochs", type=int, default=15, help="Epochs per HPO trial")
    parser.add_argument("--batch_size", type=int, default=32, help="isic/nih only (fets/cas are fixed at 1)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset (default: per-dataset config)")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name (default: per-dataset)")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (default: per-dataset sqlite file)")
    parser.add_argument("--output", type=str, default=None, help="Path to save best hyperparameters as JSON (default: per-dataset)")
    parser.add_argument("--hpo_max_steps", type=int, default=0, help="Limit training to N gradient steps per epoch (0=unlimited; HPO speedup)")
    parser.add_argument("--hpo_subsample_frac", type=float, default=None,
                         help="isic/nih only: subsample the training split to this fraction, e.g. 0.4 (default: no subsampling)")
    args = parser.parse_args()

    suffix = "" if args.dataset == "fets" else f"_{args.dataset}"
    if args.study_name is None:
        args.study_name = f"centralized_hpo{suffix}"
    if args.storage is None:
        args.storage = f"sqlite:///{os.path.join(project_root, f'optuna_centralized{suffix}.db')}"
    if args.output is None:
        args.output = os.path.join(project_root, f"hpo_best_centralized{suffix}.json")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(**_PRUNER_KWARGS.get(args.dataset, _DEFAULT_PRUNER_KWARGS)),
        load_if_exists=True,
    )

    if args.dataset in _WARM_START_PARAMS and len(study.trials) == 0:
        study.enqueue_trial(_WARM_START_PARAMS[args.dataset])
        print(f"Warm-started study with known-good params: {_WARM_START_PARAMS[args.dataset]}", flush=True)

    already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = args.n_trials - already_done
    if remaining <= 0:
        print(f"Study already has {already_done} completed trials (target={args.n_trials}). Nothing to do.")
    else:
        print(f"Starting HPO: {remaining} trial(s) remaining (target={args.n_trials})", flush=True)
        study.optimize(build_objective(args), n_trials=remaining)

    _RESULT_KEYS = {
        "isic": ("best_val_balanced_accuracy", "Val Balanced Accuracy"),
        "nih": ("best_val_mean_auc_roc", "Val Mean AUC-ROC"),
    }
    metric_key, metric_label = _RESULT_KEYS.get(args.dataset, ("best_val_mean_dice", "Val mean Dice"))

    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best trial #{best.number}  {metric_label}: {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v:.6g}")
    print("=" * 60)

    result = {
        "study_name": args.study_name,
        "best_trial": best.number,
        metric_key: round(best.value, 4),
        "best_params": {k: float(v) for k, v in best.params.items()},
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "hpo_epochs": args.hpo_epochs,
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Best config saved to {args.output}")


if __name__ == "__main__":
    main()
