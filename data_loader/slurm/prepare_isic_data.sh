#!/bin/bash
#SBATCH --job-name=is_prp
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/data_loader/slurm/std_out_isic_prepare.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/data_loader/slurm/std_err_isic_prepare.log

# CPU-bound only (stratified split over the CSV) — no GPU requested on purpose.

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

srun python3 data_loader/isic_prepare.py --data_dir data/ISIC --seed 42
