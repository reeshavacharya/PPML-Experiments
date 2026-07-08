#!/bin/bash
#SBATCH --job-name=cs_fsv
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_fedavg_server_cas.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_fedavg_server_cas.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

echo "=== FedAvg Server (CAS) starting on $(hostname) at $(date) ==="

# Auto-select best trial from fl/hpo_best_config_cas.json (written by
# hpo_coordinator_cas.sh when all 10 trials finish). HPO is still running;
# server uses built-in defaults (lr=3e-4, wd=1e-5) until the file is present.
HP_ARGS=()
if [ -f fl/hpo_best_config_cas.json ]; then
    BEST_TRIAL=$(python3 -c "import json; print(json.load(open('fl/hpo_best_config_cas.json'))['best_trial'])")
    HP_ARGS=(--hp_config "fl/hpo_configs/trial_cas_${BEST_TRIAL}.json")
    echo "HP config: fl/hpo_configs/trial_cas_${BEST_TRIAL}.json"
else
    echo "WARNING: fl/hpo_best_config_cas.json not found — using built-in defaults (lr=3e-4, wd=1e-5)."
fi

python3 -u fl/server.py \
    --dataset cas \
    --strategy fedavg \
    --rounds 100 \
    --server_address 0.0.0.0:8083 \
    --num_clients 5 \
    "${HP_ARGS[@]}"

echo "=== FedAvg Server (CAS) finished at $(date) ==="
