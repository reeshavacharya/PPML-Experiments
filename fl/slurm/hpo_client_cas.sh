#!/bin/bash
#SBATCH --job-name=cs_hcl
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_hpo_client_cas_%a_%j.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_hpo_client_cas_%a_%j.log
#SBATCH --array=1-5

# 1 CAS client per SLURM task — all 5 run in parallel on separate GPUs.
# Required env vars (passed by fl/hpo_coordinator.py via --export):
#   TRIAL_ID, HPO_ROUNDS

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

: "${HPO_ROUNDS:=6}"
: "${TRIAL_ID:=unknown}"

ADDR_FILE="fl/server_address_cas.txt"
echo "Waiting for server address file..."
for i in $(seq 1 60); do
    [ -f "$ADDR_FILE" ] && break
    sleep 5
done
SERVER_ADDR=$(cat "$ADDR_FILE" | tr -d '[:space:]')
echo "Server address: ${SERVER_ADDR}:8083"

echo "=== CAS HPO Trial ${TRIAL_ID} — Client ${SLURM_ARRAY_TASK_ID} on $(hostname) GPU ${CUDA_VISIBLE_DEVICES} at $(date) ==="

python3 -u fl/client.py \
    --dataset cas \
    --client_id "${SLURM_ARRAY_TASK_ID}" \
    --server_address "${SERVER_ADDR}:8083" \
    --num_workers 2 \
    --num_rounds "${HPO_ROUNDS}"

echo "=== CAS HPO Trial ${TRIAL_ID} — Client ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
