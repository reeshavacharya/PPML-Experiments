#!/bin/bash
#SBATCH --job-name=ft_hco
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_hpo_coordinator.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_hpo_coordinator.log

# This job submits and manages FL HPO trials sequentially.
# Each trial: server job + client array job → wait → report to Optuna.
#
# If all 20 trials don't finish within 96h, re-run this script —
# Optuna resumes automatically from the SQLite DB (load_if_exists=True).

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

echo "=== FL HPO Coordinator starting on $(hostname) at $(date) ==="

python3 -u fl/hpo_coordinator.py \
    --n_trials 10 \
    --hpo_rounds 8 \
    --num_clients 23 \
    --poll_interval 60

echo "=== FL HPO Coordinator finished at $(date) ==="
