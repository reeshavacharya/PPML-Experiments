#!/bin/bash
#SBATCH --job-name=nh_fcl
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=mdc-1057-18-4
#SBATCH --time=96:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_client_nih_%a.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_client_nih_%a.log
#SBATCH --array=1-2

# 5 NIH clients grouped into 2 concurrent-GPU tasks (3+2) rather than one
# GPU per client (--array=1-5) — muma_2021 only has 4 usable GPU-bearing
# nodes (13-8, 18-1, 18-2, 18-3; 13-9..13-13 have no GPU gres despite being
# in the same partition, and 18-4 is drained), routinely shared with other
# jobs, so needing only 2 GPUs at once schedules far more reliably. Clients
# within a task share one GPU sequentially via client_nih.py's per-GPU flock.

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

case $SLURM_ARRAY_TASK_ID in
    1) CLIENTS="1 2 3" ;;
    2) CLIENTS="4 5"   ;;
esac

echo "=== NIH FedAvg — Group $SLURM_ARRAY_TASK_ID (clients: $CLIENTS) on $(hostname) at $(date) ==="

ADDR_FILE="fl/server_address_nih.txt"
while [ ! -f "$ADDR_FILE" ]; do
    sleep 5
done
SERVER_ADDR=$(cat "$ADDR_FILE" | tr -d '[:space:]')
echo "Server address: ${SERVER_ADDR}:8082"

# Explicit PID tracking + trap: `scancel` sends SIGTERM to this script's own
# process, but backgrounded children (&) aren't guaranteed to receive it too
# — an orphaned client_nih.py can survive the cancellation and keep
# retrying/reconnecting to whatever server is later listening on the same
# port, silently corrupting a later run's client count. Trap ensures they
# actually die with this job.
CHILD_PIDS=()
trap 'echo "Caught signal — killing child clients: ${CHILD_PIDS[*]}"; kill "${CHILD_PIDS[@]}" 2>/dev/null; exit 1' TERM INT

for CID in $CLIENTS; do
    python3 -u fl/client_nih.py \
        --client_id $CID \
        --server_address "${SERVER_ADDR}:8082" \
        --num_workers 4 \
        --resume \
        > "fl/slurm/std_out_client_nih_${CID}.log" 2>&1 &
    CHILD_PIDS+=($!)
    echo "  Client $CID started (PID: $!)"
    sleep 2
done

echo "All clients in group $SLURM_ARRAY_TASK_ID launched. Waiting..."
wait "${CHILD_PIDS[@]}"

echo "=== NIH FedAvg — Group $SLURM_ARRAY_TASK_ID finished at $(date) ==="
