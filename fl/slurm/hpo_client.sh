#!/bin/bash
#SBATCH --job-name=ft_hcl
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_hpo_client_%a_%j.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_hpo_client_%a_%j.log
#SBATCH --array=1-4

# 4 tasks × 1 GPU each = 4 GPU slots.
# Two tasks can co-locate on a 2-GPU node (SLURM assigns different CUDA_VISIBLE_DEVICES,
# so the per-GPU lock in client.py serializes only clients within the same task).
#
# Resource sizing: mdc-1057-13-8 has 2 free GPUs but only 12 free CPUs / ~113 GB RAM.
# 4 CPUs + 50 GB per task lets 2 tasks fit there simultaneously (8 CPUs / 100 GB used).
#
# Required env vars (passed by hpo_coordinator.py via --export):
#   TRIAL_ID   - Optuna trial number (for logging)
#   HPO_ROUNDS - number of FL rounds for this trial

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

: "${HPO_ROUNDS:=8}"
: "${TRIAL_ID:=unknown}"

# Wait until server_address.txt is written by the server
ADDR_FILE="fl/server_address.txt"
echo "Waiting for server address file..."
for i in $(seq 1 60); do
    [ -f "$ADDR_FILE" ] && break
    sleep 5
done
SERVER_ADDR=$(cat "$ADDR_FILE" | tr -d '[:space:]')
echo "Server address: ${SERVER_ADDR}:8080"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Client groups — one GPU per task; clients within a group serialized by GPU lock.
case $SLURM_ARRAY_TASK_ID in
    1) CLIENTS="1 2 3 4 5 6"          ;;
    2) CLIENTS="7 8 9 10 11 12 13"    ;;
    3) CLIENTS="14 15 16 17 18 19 20" ;;
    4) CLIENTS="21 22 23"             ;;
esac

echo "=== HPO Trial ${TRIAL_ID} — Group ${SLURM_ARRAY_TASK_ID} on $(hostname) GPU ${CUDA_VISIBLE_DEVICES} at $(date) ==="
echo "    Clients: ${CLIENTS}"

# Explicit PID tracking + trap: `scancel` sends SIGTERM to this script's own
# process, but backgrounded children (&) aren't guaranteed to receive it too
# — an orphaned client.py can survive the cancellation and keep
# retrying/reconnecting to whatever server is later listening on the same
# port, silently corrupting a later trial's client count. Trap ensures they
# actually die with this job.
CHILD_PIDS=()
trap 'echo "Caught signal — killing child clients: ${CHILD_PIDS[*]}"; kill "${CHILD_PIDS[@]}" 2>/dev/null; exit 1' TERM INT

for CID in $CLIENTS; do
    python3 -u fl/client.py \
        --client_id "$CID" \
        --server_address "${SERVER_ADDR}:8080" \
        --num_workers 1 \
        --num_rounds "${HPO_ROUNDS}" \
        > "fl/slurm/std_out_hpo_client_${CID}_trial_${TRIAL_ID}.log" 2>&1 &
    CHILD_PIDS+=($!)
    echo "  Client ${CID} launched (PID: $!)"
    sleep 2
done

echo "All clients in group ${SLURM_ARRAY_TASK_ID} launched. Waiting..."
wait "${CHILD_PIDS[@]}"
echo "=== HPO Trial ${TRIAL_ID} — Group ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
