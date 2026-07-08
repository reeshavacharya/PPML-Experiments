#!/bin/bash
#SBATCH --job-name=nh_fsv
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_fedavg_server_nih.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_fedavg_server_nih.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

# Clear any stale address from a previous server run — see fedpidavg_server.sh
# for the race this guards against (client reads a leftover address before
# the new server overwrites it, then silently fails to train).
rm -f fl/server_address_nih.txt

echo "=== FedAvg Server (NIH) starting on $(hostname) at $(date) ==="

# Update --lr/--weight_decay from fl/hpo_best_config_nih.json once
# fl/slurm/hpo_coordinator_nih.sh has run (or pass --hp_config <path> instead).
python3 -u fl/server_nih.py \
    --rounds 50 \
    --server_address 0.0.0.0:8082 \
    --num_clients 5

echo "=== FedAvg Server (NIH) finished at $(date) ==="
