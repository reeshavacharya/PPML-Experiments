#!/bin/bash
#SBATCH --job-name=cs_scl
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --exclude=mdc-1057-13-8,mdc-1057-18-1,mdc-1057-18-4
#SBATCH --time=02:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_smoke_client_cas_%a.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_smoke_client_cas_%a.log
#SBATCH --array=1-3

# 5 CAS clients grouped into 3 concurrent-GPU tasks (2+2+1) instead of 1
# client-per-GPU (--array=1-5) — muma_2021 only had 3 free GPUs total when
# this was written (1 on mdc-1057-18-2, 2 on mdc-1057-18-3; mdc-1057-13-8 and
# mdc-1057-18-1 were fully allocated, mdc-1057-18-4 has free GPUs but is
# DRAINed from a disk-usage issue). Clients within a task share one GPU
# sequentially via client.py's existing per-GPU flock (same mechanism
# fedpidavg_client.sh uses for FeTS's 23-institution grouping) — correctness
# is unaffected, it's just slower than fully parallel. --exclude blacklists
# the fully-allocated/drained nodes (a positive --nodelist of 2 hosts made
# sbatch mis-parse the node count as "-N 2-1" and reject the job outright);
# --gres=gpu:1 already filters out the GPU-less 13-x nodes on its own, so
# this leaves 18-2/18-3 as the only viable candidates without hardcoding
# them. Drop the --exclude once more GPUs free up cluster-wide.

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

case $SLURM_ARRAY_TASK_ID in
    1) CLIENTS="1 2" ;;
    2) CLIENTS="3 4" ;;
    3) CLIENTS="5"   ;;
esac

echo "=== CAS FL smoke test — Group $SLURM_ARRAY_TASK_ID (clients: $CLIENTS) on $(hostname) at $(date) ==="

ADDR_FILE="fl/server_address_cas.txt"
echo "Waiting for server address file..."
for i in $(seq 1 60); do
    [ -f "$ADDR_FILE" ] && break
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
        --num_rounds 3 \
        --seed 9999 \
        > "fl/slurm/std_out_smoke_client_cas_${CID}.log" 2>&1 &
    CHILD_PIDS+=($!)
    echo "  Client $CID launched (PID: $!)"
    sleep 2
done

echo "All clients in group $SLURM_ARRAY_TASK_ID launched. Waiting..."
wait "${CHILD_PIDS[@]}"

echo "=== CAS FL smoke test — Group $SLURM_ARRAY_TASK_ID finished at $(date) ==="
echo ""
echo "Check for: fl/client_cas_seed9999_{1..5}_metrics.csv — 3 fit rows + 3 val"
echo "rows each, Dice/HD95/clDice populated on val rows, resource columns on fit rows."
