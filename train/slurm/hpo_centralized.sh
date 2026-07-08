#!/bin/bash
#SBATCH --job-name=ft_chp
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_out_hpo_centralized.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_err_hpo_centralized.log

ulimit -c 0   # disable core dumps — Bus error crashes were filling the NFS filesystem

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

echo "=== Centralized HPO starting on $(hostname) at $(date) ==="

python3 -u train/hpo_centralized.py \
    --n_trials 12 \
    --hpo_epochs 8 \
    --data_dir data/FeTS2022

echo "=== Centralized HPO finished at $(date) ==="
