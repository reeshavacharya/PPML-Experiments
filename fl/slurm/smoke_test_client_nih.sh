#!/bin/bash
#SBATCH --job-name=nh_scl
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=mdc-1057-18-4
#SBATCH --time=02:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_smoke_client_nih.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_smoke_client_nih.log

# All 5 clients serialized on a single GPU (no array) — unlike
# fedavg_client_nih.sh's 2-GPU grouping, this only needs ONE free GPU slot
# to schedule at all, which matters when muma_2021's whole 8-GPU capacity
# is already saturated by concurrent FeTS/ISIC HPO jobs. (snsm_itn19 was
# tried as a workaround for the GPU scarcity — ruled out: /work/r/reeshav
# isn't mounted there at all, a muma-group-specific storage export, so
# every non-muma partition is a dead end for this project regardless of
# QOS/GPU access.) Fine for a one-off 3-round smoke test; the real 50-round
# run keeps the 2-GPU split for throughput. Submit smoke_test_server_nih.sh
# first (or alongside — this waits on fl/server_address_nih.txt).

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

echo "=== NIH FL smoke test — all 5 clients on $(hostname) at $(date) ==="

ADDR_FILE="fl/server_address_nih.txt"
echo "Waiting for server address file..."
for i in $(seq 1 60); do
    [ -f "$ADDR_FILE" ] && break
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

for CID in 1 2 3 4 5; do
    python3 -u fl/client_nih.py \
        --client_id $CID \
        --server_address "${SERVER_ADDR}:8082" \
        --num_workers 4 \
        --num_rounds 3 \
        --seed 9999 \
        > "fl/slurm/std_out_smoke_client_nih_${CID}.log" 2>&1 &
    CHILD_PIDS+=($!)
    echo "  Client $CID started (PID: $!)"
    sleep 2
done

echo "All 5 clients launched. Waiting..."
wait "${CHILD_PIDS[@]}"

echo "=== NIH FL smoke test finished at $(date) ==="
echo ""
echo "Check for: fl/client_nih_seed9999_{1..5}_metrics.csv — 3 fit rows + 3 val"
echo "rows each, Mean AUC/Macro F1/per-label AUC populated on val rows,"
echo "resource columns populated on fit rows."
