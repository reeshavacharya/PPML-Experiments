#!/bin/bash
#SBATCH --job-name=ft_cte
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/test/slurm/std_out_test_baseline.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/test/slurm/std_err_test_baseline.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
ulimit -c 0

srun python3 test/centralized_baseline.py --dataset fets --setting centralized