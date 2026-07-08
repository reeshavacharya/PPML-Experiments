#!/bin/bash
#SBATCH --job-name=is_scl
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_smoke_client_isic_%a.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_smoke_client_isic_%a.log
#SBATCH --array=1-3

# 6 ISIC clients grouped into 3 concurrent-GPU tasks (2 each), same GPU-
# conservation reasoning as smoke_test_client_cas.sh. Clients within a task
# share one GPU sequentially via client_isic.py's per-GPU flock. Submit
# smoke_test_server_isic.sh first (or alongside — the client waits on
# fl/server_address_isic.txt).

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

case $SLURM_ARRAY_TASK_ID in
    1) CLIENTS="1 2" ;;
    2) CLIENTS="3 4" ;;
    3) CLIENTS="5 6" ;;
esac

echo "=== ISIC FL smoke test — Group $SLURM_ARRAY_TASK_ID (clients: $CLIENTS) on $(hostname) at $(date) ==="

ADDR_FILE="fl/server_address_isic.txt"
echo "Waiting for server address file..."
for i in $(seq 1 60); do
    [ -f "$ADDR_FILE" ] && break
    sleep 5
done
SERVER_ADDR=$(cat "$ADDR_FILE" | tr -d '[:space:]')
echo "Server address: ${SERVER_ADDR}:8081"

# Explicit PID tracking + trap: `scancel` sends SIGTERM to this script's own
# process, but backgrounded children (&) aren't guaranteed to receive it too
# — an orphaned client_isic.py can survive the cancellation and keep
# retrying/reconnecting to whatever server is later listening on the same
# port, silently corrupting a later run's client count. Trap ensures they
# actually die with this job.
CHILD_PIDS=()
trap 'echo "Caught signal — killing child clients: ${CHILD_PIDS[*]}"; kill "${CHILD_PIDS[@]}" 2>/dev/null; exit 1' TERM INT

for CID in $CLIENTS; do
    python3 -u fl/client_isic.py \
        --client_id $CID \
        --server_address "${SERVER_ADDR}:8081" \
        --num_workers 4 \
        --num_rounds 3 \
        --seed 9999 \
        > "fl/slurm/std_out_smoke_client_isic_${CID}.log" 2>&1 &
    CHILD_PIDS+=($!)
    echo "  Client $CID launched (PID: $!)"
    sleep 2
done

echo "All clients in group $SLURM_ARRAY_TASK_ID launched. Waiting..."
wait "${CHILD_PIDS[@]}"

echo "=== ISIC FL smoke test — Group $SLURM_ARRAY_TASK_ID finished at $(date) ==="
echo ""
echo "Check for: fl/client_isic_seed9999_{1..6}_metrics.csv — 3 fit rows + 3 val"
echo "rows each, Balanced Accuracy/Macro F1/Macro AUC populated on val rows,"
echo "resource columns populated on fit rows."
