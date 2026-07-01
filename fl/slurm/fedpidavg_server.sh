#!/bin/bash
#SBATCH --job-name=fpid_server
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_fedpidavg_server.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_fedpidavg_server.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

echo "=== FedPIDAvg Server starting on $(hostname) at $(date) ==="

python3 -u fl/server.py \
    --rounds 100 \
    --server_address 0.0.0.0:8080 \
    --num_clients 23 \
    --alpha 0.45 --beta 0.45 --gamma 0.1 \
    --resume

echo "=== FedPIDAvg Server finished at $(date) ==="
