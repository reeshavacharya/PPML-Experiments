#!/bin/bash
#SBATCH --job-name=is_htr
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --exclude=mdc-1057-18-4
#SBATCH --time=12:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_hpo_trial_isic_%j.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_hpo_trial_isic_%j.log

# Combined server + all-clients job for one ISIC FL HPO trial, sharing a
# single GPU — replaces the old 2-job layout (CPU-only server job +
# 2-array-task client job needing 2 GPUs at once). fl/client_isic.py already
# serializes actual GPU residency per client via a per-GPU flock
# (_acquire_gpu/_release_gpu), so VRAM was never additive across concurrent
# clients on one GPU — packing all 6 onto 1 GPU instead of 3+3 across 2
# doesn't add OOM risk, it just roughly doubles wall-clock per round (one
# lock instead of two running in parallel). That's a good trade here: FL HPO
# jobs were routinely stuck PENDING waiting for 2 simultaneous GPU nodes,
# and this frees an entire GPU per trial for FeTS/CAS.
#
# Required env vars (passed by fl/hpo_coordinator.py via --export): TRIAL_ID,
# HP_CONFIG, RESULT_PATH, HPO_ROUNDS, NUM_CLIENTS (default 6),
# HPO_SUBSAMPLE_FRAC (optional, empty = no subsampling)

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

: "${TRIAL_ID:?TRIAL_ID not set}"
: "${HP_CONFIG:?HP_CONFIG not set}"
: "${RESULT_PATH:?RESULT_PATH not set}"
: "${HPO_ROUNDS:=5}"
: "${NUM_CLIENTS:=6}"
: "${HPO_SUBSAMPLE_FRAC:=}"

echo "=== ISIC FL HPO Trial ${TRIAL_ID} (single-GPU) starting on $(hostname) at $(date) ==="

ADDR_FILE="fl/server_address_isic.txt"
rm -f "$ADDR_FILE"

CHILD_PIDS=()
trap 'echo "Caught signal — killing children: ${CHILD_PIDS[*]}"; kill "${CHILD_PIDS[@]}" 2>/dev/null; exit 1' TERM INT

python3 -u fl/server_isic.py \
    --rounds "${HPO_ROUNDS}" \
    --num_clients "${NUM_CLIENTS}" \
    --server_address 0.0.0.0:8081 \
    --hp_config "${HP_CONFIG}" \
    --result_path "${RESULT_PATH}" \
    > "fl/slurm/std_out_hpo_server_isic_trial_${TRIAL_ID}.log" 2>&1 &
CHILD_PIDS+=($!)
echo "Server launched (PID: $!)"

echo "Waiting for server address file..."
for i in $(seq 1 60); do
    [ -f "$ADDR_FILE" ] && break
    sleep 5
done
SERVER_ADDR=$(cat "$ADDR_FILE" | tr -d '[:space:]')
echo "Server address: ${SERVER_ADDR}:8081"

SUBSAMPLE_ARGS=()
if [ -n "$HPO_SUBSAMPLE_FRAC" ]; then
    SUBSAMPLE_ARGS=(--train_subsample_frac "$HPO_SUBSAMPLE_FRAC")
fi

for CID in $(seq 1 "${NUM_CLIENTS}"); do
    python3 -u fl/client_isic.py \
        --client_id "$CID" \
        --server_address "${SERVER_ADDR}:8081" \
        --num_workers 2 \
        --num_rounds "${HPO_ROUNDS}" \
        "${SUBSAMPLE_ARGS[@]}" \
        > "fl/slurm/std_out_hpo_client_isic_${CID}_trial_${TRIAL_ID}.log" 2>&1 &
    CHILD_PIDS+=($!)
    echo "  Client $CID launched (PID: $!)"
    sleep 2
done

echo "All processes launched (server + ${NUM_CLIENTS} clients) on 1 GPU. Waiting..."
wait "${CHILD_PIDS[@]}"
echo "=== ISIC FL HPO Trial ${TRIAL_ID} finished at $(date) ==="
