#!/bin/bash
#SBATCH --job-name=cs_hco
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_hpo_coordinator_cas.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_hpo_coordinator_cas.log

# This job submits and manages CAS FL HPO trials sequentially.
# Each trial: server job (hpo_server_cas.sh) + client array job
# (hpo_client_cas.sh, 2+2+1 grouped) -> wait -> report to Optuna.
#
# Per-round timing from the FL smoke test (2+2+1 client grouping): ~35-40
# min/round, so 8 rounds/trial ~5h, 10 trials ~50h -- fits this 96h budget
# in one submission. If it doesn't, re-run this script -- Optuna resumes
# automatically from optuna_fl_cas.db (load_if_exists=True).

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

echo "=== CAS FL HPO Coordinator starting on $(hostname) at $(date) ==="

python3 -u fl/hpo_coordinator.py \
    --dataset cas \
    --n_trials 10 \
    --hpo_rounds 6 \
    --num_clients 5 \
    --poll_interval 60 \
    --hpo_max_steps 50

echo "=== CAS FL HPO Coordinator finished at $(date) ==="
