#!/bin/bash
#SBATCH --job-name=cs_hsv
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_hpo_server_cas_%j.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_hpo_server_cas_%j.log

# Required env vars (passed by fl/hpo_coordinator.py via --export):
#   TRIAL_ID, HP_CONFIG, RESULT_PATH, HPO_ROUNDS, NUM_CLIENTS (default 5)

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

: "${TRIAL_ID:?TRIAL_ID not set}"
: "${HP_CONFIG:?HP_CONFIG not set}"
: "${RESULT_PATH:?RESULT_PATH not set}"
: "${HPO_ROUNDS:=15}"
: "${NUM_CLIENTS:=5}"
: "${HPO_MAX_STEPS:=0}"

echo "=== CAS HPO Server Trial ${TRIAL_ID} starting on $(hostname) at $(date) ==="
echo "    HP config: ${HP_CONFIG}"
echo "    Rounds:    ${HPO_ROUNDS}"

python3 -u fl/server.py \
    --dataset cas \
    --strategy fedavg \
    --rounds "${HPO_ROUNDS}" \
    --num_clients "${NUM_CLIENTS}" \
    --server_address 0.0.0.0:8083 \
    --hp_config "${HP_CONFIG}" \
    --result_path "${RESULT_PATH}" \
    --hpo_max_steps "${HPO_MAX_STEPS}"

echo "=== CAS HPO Server Trial ${TRIAL_ID} finished at $(date) ==="
