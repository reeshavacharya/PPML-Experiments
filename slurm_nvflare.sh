#!/bin/bash
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G              # Safely requests almost all of the 257GB available
#SBATCH --gres=gpu:2            # Uses the cluster's generic GPU tag
#SBATCH --time=48:00:00

#SBATCH --job-name=nvflare
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/std_out_nvflare.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/std_err_nvflare.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

# Execute the training pipeline
srun nvflare simulator nvflare_job -w /tmp/nvflare_workspace -n 3 -t 3