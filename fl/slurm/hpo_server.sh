#!/bin/bash
#SBATCH --job-name=ft_hsv
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_hpo_server_%j.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_hpo_server_%j.log

# Required env vars (passed by hpo_coordinator.py via --export):
#   TRIAL_ID        - Optuna trial number
#   HP_CONFIG       - path to trial JSON config
#   RESULT_PATH     - path to write result JSON
#   HPO_ROUNDS      - number of FL rounds for this trial
#   NUM_CLIENTS     - number of clients (default 23)

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

: "${TRIAL_ID:?TRIAL_ID not set}"
: "${HP_CONFIG:?HP_CONFIG not set}"
: "${RESULT_PATH:?RESULT_PATH not set}"
: "${HPO_ROUNDS:=15}"
: "${NUM_CLIENTS:=23}"

echo "=== HPO Server Trial ${TRIAL_ID} starting on $(hostname) at $(date) ==="
echo "    HP config: ${HP_CONFIG}"
echo "    Rounds:    ${HPO_ROUNDS}"

python3 -u fl/server.py \
    --rounds "${HPO_ROUNDS}" \
    --num_clients "${NUM_CLIENTS}" \
    --server_address 0.0.0.0:8080 \
    --hp_config "${HP_CONFIG}" \
    --result_path "${RESULT_PATH}"

echo "=== HPO Server Trial ${TRIAL_ID} finished at $(date) ==="
