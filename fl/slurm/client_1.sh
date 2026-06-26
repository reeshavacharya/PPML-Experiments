#!/bin/bash
#SBATCH --job-name=fl_client_1
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --nodelist=mdc-1057-18-3
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_client_1.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_client_1.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

ADDR_FILE="fl/server_address.txt"
echo "Waiting for server address file..."
while [ ! -f "$ADDR_FILE" ]; do
    sleep 5
done
SERVER_ADDR=$(cat "$ADDR_FILE" | tr -d '[:space:]')
echo "=== FL Client 1 connecting to ${SERVER_ADDR}:8080 from $(hostname) at $(date) ==="

srun python3 -u fl/client.py --client_id 1 --server_address "${SERVER_ADDR}:8080" --num_workers 2

echo "=== FL Client 1 finished at $(date) ==="