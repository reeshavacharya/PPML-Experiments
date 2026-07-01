#!/bin/bash
#SBATCH --job-name=fpid_g7fix
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --nodelist=mdc-1057-18-2
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_group_7_fix.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_group_7_fix.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

SERVER_ADDR=$(cat fl/server_address.txt | tr -d '[:space:]')
echo "=== Group 7 fix (institutions: 21 22 23) on $(hostname) at $(date) ==="
echo "Server: ${SERVER_ADDR}:8080"

for CID in 21 22 23; do
    python3 -u fl/client.py \
        --client_id $CID \
        --server_address "${SERVER_ADDR}:8080" \
        --num_workers 1 \
        --resume \
        > fl/slurm/std_out_client_${CID}.log 2>&1 &
    echo "  Client $CID started (PID: $!)"
    sleep 2
done

echo "All clients launched. Waiting..."
wait

echo "=== Group 7 fix finished at $(date) ==="
