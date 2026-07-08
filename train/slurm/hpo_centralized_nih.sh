#!/bin/bash
#SBATCH --job-name=nh_chp
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=mdc-1057-18-4
#SBATCH --time=08:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_out_hpo_centralized_nih.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_err_hpo_centralized_nih.log

ulimit -c 0   # disable core dumps — Bus error crashes were filling the NFS filesystem

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

echo "=== NIH Centralized HPO starting on $(hostname) at $(date) ==="

# Sized for a ~2-3h ceiling: 10 trials x 1 epoch x ~15min/epoch = 2.5h worst
# case with the full training set (no pruning) — NIH costs ~3x more per
# epoch than ISIC, so epochs/trial had to drop further to hit the same time
# budget. --hpo_subsample_frac 0.3 (more aggressive than ISIC's 0.4 for the
# same reason) cuts that further to under 1h, leaving headroom to re-add
# epochs/trials later if search quality needs it. Warm-started + pruner-
# tuned in train/hpo_centralized.py. Uses a separate "_fast" study/output so
# it never collides with the original (much slower, 10-epoch, full-data)
# centralized_hpo_nih study — this script is always freely re-runnable on
# its own.
python3 -u train/hpo_centralized.py \
    --dataset nih \
    --n_trials 10 \
    --hpo_epochs 1 \
    --batch_size 32 \
    --study_name centralized_hpo_nih_fast \
    --storage sqlite:///optuna_centralized_nih_fast.db \
    --output hpo_best_centralized_nih_fast.json \
    --hpo_subsample_frac 0.3

echo "=== NIH Centralized HPO finished at $(date) ==="
