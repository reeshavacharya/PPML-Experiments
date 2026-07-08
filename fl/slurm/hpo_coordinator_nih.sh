#!/bin/bash
#SBATCH --job-name=nh_hco
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_hpo_coordinator_nih.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_hpo_coordinator_nih.log

# This job submits and manages NIH FL HPO trials sequentially. Each trial
# is now ONE combined single-GPU SLURM job (fl/slurm/hpo_trial_nih.sh: server
# + all 5 clients sharing 1 GPU via the existing per-GPU flock in
# fl/client_nih.py), replacing the old layout of a separate CPU-only server
# job plus a 2-array-task client job (3+2) that needed 2 simultaneous GPUs —
# that was routinely stuck PENDING waiting on GPU availability.
#
# Sized for a ~2-3h compute ceiling: single-GPU serialization roughly doubles
# per-round time vs the old 2-GPU-parallel layout, but --hpo_subsample_frac
# 0.3 claws most of that back by training each client on 30% of its own data
# (NIH costs ~3x more per epoch than ISIC, so a more aggressive subsample
# fraction is used here). 3 trials x 5 rounds within the same ~2-3h ceiling.
# Clients unchanged (5) per experimental design. If it doesn't finish in
# time, re-run this script — Optuna resumes automatically from
# optuna_fl_nih_fast.db (load_if_exists=True).

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

echo "=== NIH FL HPO Coordinator starting on $(hostname) at $(date) ==="

python3 -u fl/hpo_coordinator.py \
    --dataset nih \
    --n_trials 3 \
    --hpo_rounds 5 \
    --num_clients 5 \
    --poll_interval 60 \
    --study_name fl_hpo_nih_fast \
    --storage sqlite:///optuna_fl_nih_fast.db \
    --hpo_subsample_frac 0.3

echo "=== NIH FL HPO Coordinator finished at $(date) ==="
