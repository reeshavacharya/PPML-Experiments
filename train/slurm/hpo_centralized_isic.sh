#!/bin/bash
#SBATCH --job-name=is_chp
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_out_hpo_centralized_isic.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_err_hpo_centralized_isic.log

ulimit -c 0   # disable core dumps — Bus error crashes were filling the NFS filesystem

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

echo "=== ISIC Centralized HPO starting on $(hostname) at $(date) ==="

# Sized for a ~2-3h ceiling: 10 trials x 3 epochs x ~5.5min/epoch ~= 2.75h
# worst case with the full training set (no pruning). --hpo_subsample_frac
# 0.4 (train on 40% of the training split only; val/test stay full-size)
# cuts that further to ~1-1.5h, leaving headroom to re-add trials/epochs
# later if search quality needs it. Warm-started + pruner-tuned in
# train/hpo_centralized.py. Uses a separate "_fast" study/output so it never
# collides with the already-completed 12-trial centralized_hpo_isic study
# (which used 10 epochs/trial and the full training set) — this script is
# always freely re-runnable on its own.
python3 -u train/hpo_centralized.py \
    --dataset isic \
    --n_trials 10 \
    --hpo_epochs 3 \
    --batch_size 32 \
    --study_name centralized_hpo_isic_fast \
    --storage sqlite:///optuna_centralized_isic_fast.db \
    --output hpo_best_centralized_isic_fast.json \
    --hpo_subsample_frac 0.4

echo "=== ISIC Centralized HPO finished at $(date) ==="
