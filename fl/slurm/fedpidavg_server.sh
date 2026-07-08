#!/bin/bash
#SBATCH --job-name=ft_psv
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --exclude=mdc-1057-13-8,mdc-1057-18-3,mdc-1057-18-4
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_fedpidavg_server.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_fedpidavg_server.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

# Clear any stale address from a previous server run. fl/server.py doesn't
# write the fresh address until ~1min into startup (after model init), and
# the client wrapper's wait-loop only checks file *existence* — a client
# array task launched in that window can read a leftover address pointing
# at a dead server, exhaust its retries, and silently exit without training
# (this happened to institutions 1-6 on 2026-07-06).
rm -f fl/server_address.txt

echo "=== FedPIDAvg Server starting on $(hostname) at $(date) ==="

# Auto-select best trial from fl/hpo_best_config.json (written by hpo_coordinator.sh
# when all 10 trials finish). Falls back to trial 3 (best-so-far: val mean Dice 0.8037)
# while HPO is still running. After HPO finishes, no manual edit is needed.
if [ -f fl/hpo_best_config.json ]; then
    BEST_TRIAL=$(python3 -c "import json; print(json.load(open('fl/hpo_best_config.json'))['best_trial'])")
    HP_CONFIG="fl/hpo_configs/trial_${BEST_TRIAL}.json"
else
    HP_CONFIG="fl/hpo_configs/trial_3.json"
fi
echo "HP config: ${HP_CONFIG}"

python3 -u fl/server.py \
    --dataset fets \
    --rounds 100 \
    --server_address 0.0.0.0:8080 \
    --num_clients 23 \
    --hp_config "${HP_CONFIG}"

echo "=== FedPIDAvg Server finished at $(date) ==="
