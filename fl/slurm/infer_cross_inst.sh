#!/bin/bash
#SBATCH --job-name=fl_infer_cross
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_infer_cross.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_infer_cross.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

echo "=== FL Cross-Institutional Inference starting on $(hostname) at $(date) ==="

python3 -u fl/inference_cross_inst.py

echo "=== FL Cross-Institutional Inference finished at $(date) ==="
