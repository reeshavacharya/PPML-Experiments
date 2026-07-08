#!/bin/bash
#SBATCH --job-name=is_tst
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=mdc-1057-18-4
#SBATCH --time=02:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/test/slurm/std_out_test_centralized_isic.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/test/slurm/std_err_test_centralized_isic.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

# Evaluates the centrally-trained ISIC checkpoint on the shared 4,650-image
# pooled test set. For the FL checkpoint on the same test set once available,
# pass --setting fl instead.
srun python3 test/centralized_classification.py --dataset isic --setting centralized
