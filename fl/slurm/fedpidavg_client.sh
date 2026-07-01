#!/bin/bash
#SBATCH --job-name=fpid_cli
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_group_%a.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_group_%a.log
#SBATCH --array=1-7

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

# Map each array task to a group of institutions
case $SLURM_ARRAY_TASK_ID in
    1) CLIENTS="1 2 3" ;;
    2) CLIENTS="4 5 6" ;;
    3) CLIENTS="7 8 9 10" ;;
    4) CLIENTS="11 12 13" ;;
    5) CLIENTS="14 15 16" ;;
    6) CLIENTS="17 18 19 20" ;;
    7) CLIENTS="21 22 23" ;;
esac

echo "=== Group $SLURM_ARRAY_TASK_ID (institutions: $CLIENTS) on $(hostname) at $(date) ==="

ADDR_FILE="fl/server_address.txt"
while [ ! -f "$ADDR_FILE" ]; do
    sleep 5
done
SERVER_ADDR=$(cat "$ADDR_FILE" | tr -d '[:space:]')
echo "Server address: ${SERVER_ADDR}:8080"

# Launch one Flower client per institution (GPU lock serializes access)
for CID in $CLIENTS; do
    python3 -u fl/client.py \
        --client_id $CID \
        --server_address "${SERVER_ADDR}:8080" \
        --num_workers 1 \
        --resume \
        > fl/slurm/std_out_client_${CID}.log 2>&1 &
    echo "  Client $CID started (PID: $!)"
    sleep 2
done

echo "All clients in group $SLURM_ARRAY_TASK_ID launched. Waiting..."
wait

echo "=== Group $SLURM_ARRAY_TASK_ID finished at $(date) ==="
