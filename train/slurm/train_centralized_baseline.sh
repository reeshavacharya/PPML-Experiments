#!/bin/bash
#SBATCH --job-name=ft_ctr
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_out_train_baseline.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_err_train_baseline.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
ulimit -c 0

# Best hyperparameters from centralized HPO (optuna, 12 trials, 8 epochs each)
# Best trial #0: val mean Dice 0.8754
srun python3 train/centralized_baseline.py \
    --lr 5.6115164153345e-05 \
    --weight_decay 0.0007114476009343421 \
    --epochs 100 \
    --num_workers 2