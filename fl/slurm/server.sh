#!/bin/bash
#SBATCH --job-name=ft_legsv
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_server.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_server.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

echo "=== FL Server starting on $(hostname) at $(date) ==="

srun python3 -u fl/server.py --rounds 100 --server_address 0.0.0.0:8080

echo "=== FL Server finished at $(date) ==="
