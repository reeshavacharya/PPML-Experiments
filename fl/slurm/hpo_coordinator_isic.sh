#!/bin/bash
#SBATCH --job-name=is_hco
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_hpo_coordinator_isic.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_hpo_coordinator_isic.log

# This job submits and manages ISIC FL HPO trials sequentially. Each trial
# is now ONE combined single-GPU SLURM job (fl/slurm/hpo_trial_isic.sh: server
# + all 6 clients sharing 1 GPU via the existing per-GPU flock in
# fl/client_isic.py), replacing the old layout of a separate CPU-only server
# job plus a 2-array-task client job that needed 2 simultaneous GPUs — that
# was routinely stuck PENDING waiting on GPU availability.
#
# Sized for a ~2-3h compute ceiling: single-GPU serialization roughly doubles
# per-round time vs the old 2-GPU-parallel layout (~13min -> ~26min/round),
# but --hpo_subsample_frac 0.4 claws most of that back (~10-11min/round) by
# training each client on 40% of its own data. 3 trials x 5 rounds x ~11min
# ~= 2.75h worst case. Clients unchanged (6) per experimental design. If it
# doesn't finish in time, re-run this script — Optuna resumes automatically
# from optuna_fl_isic_fast.db (load_if_exists=True).

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

echo "=== ISIC FL HPO Coordinator starting on $(hostname) at $(date) ==="

python3 -u fl/hpo_coordinator.py \
    --dataset isic \
    --n_trials 3 \
    --hpo_rounds 5 \
    --num_clients 6 \
    --poll_interval 60 \
    --study_name fl_hpo_isic_fast \
    --storage sqlite:///optuna_fl_isic_fast.db \
    --hpo_subsample_frac 0.4

echo "=== ISIC FL HPO Coordinator finished at $(date) ==="
