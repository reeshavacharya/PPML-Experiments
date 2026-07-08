#!/bin/bash
#SBATCH --job-name=cs_fcl
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_client_cas_%a.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_client_cas_%a.log
#SBATCH --array=1-3

# 5 CAS clients grouped into 3 concurrent-GPU tasks (2+2+1) rather than one
# GPU per client (--array=1-5): muma_2021 only has 4 GPU-bearing nodes x 2
# GPUs = 8 total, shared with other users and this project's own FeTS jobs,
# so 5 simultaneous GPU requests routinely queue on "(Resources)". Clients
# within a task share one GPU sequentially via client.py's existing per-GPU
# flock — same mechanism fedpidavg_client.sh uses for FeTS's 23-institution
# grouping. Correctness is unaffected, only wall-clock time per round.

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

case $SLURM_ARRAY_TASK_ID in
    1) CLIENTS="1 2" ;;
    2) CLIENTS="3 4" ;;
    3) CLIENTS="5"   ;;
esac

echo "=== CAS FedAvg — Group $SLURM_ARRAY_TASK_ID (clients: $CLIENTS) on $(hostname) at $(date) ==="

ADDR_FILE="fl/server_address_cas.txt"
while [ ! -f "$ADDR_FILE" ]; do
    sleep 5
done
SERVER_ADDR=$(cat "$ADDR_FILE" | tr -d '[:space:]')
echo "Server address: ${SERVER_ADDR}:8083"

# Explicit PID tracking + trap: `scancel` sends SIGTERM to this script's own
# process, but backgrounded children (&) aren't guaranteed to receive it too
# — an orphaned client.py can survive the cancellation and keep
# retrying/reconnecting to whatever server is later listening on the same
# port, silently corrupting a later run's client count. Trap ensures they
# actually die with this job.
CHILD_PIDS=()
trap 'echo "Caught signal — killing child clients: ${CHILD_PIDS[*]}"; kill "${CHILD_PIDS[@]}" 2>/dev/null; exit 1' TERM INT

for CID in $CLIENTS; do
    python3 -u fl/client.py \
        --dataset cas \
        --client_id $CID \
        --server_address "${SERVER_ADDR}:8083" \
        --num_workers 4 \
        --resume \
        > "fl/slurm/std_out_client_cas_${CID}.log" 2>&1 &
    CHILD_PIDS+=($!)
    echo "  Client $CID started (PID: $!)"
    sleep 2
done

echo "All clients in group $SLURM_ARRAY_TASK_ID launched. Waiting..."
wait "${CHILD_PIDS[@]}"

echo "=== CAS FedAvg — Group $SLURM_ARRAY_TASK_ID finished at $(date) ==="
