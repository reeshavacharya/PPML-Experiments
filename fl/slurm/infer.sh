#!/bin/bash
#SBATCH --job-name=fl_infer
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_infer.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_infer.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

echo "=== FL Inference starting on $(hostname) at $(date) ==="

python3 -u fl/inference.py

echo "=== FL Inference finished at $(date) ==="
