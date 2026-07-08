#!/bin/bash
#SBATCH --job-name=nh_tst
#SBATCH --partition=cbcs
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/test/slurm/std_out_test_centralized_nih.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/test/slurm/std_err_test_centralized_nih.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

# Evaluates the centrally-trained NIH ChestX-ray14 checkpoint (the epoch-6
# peak, val_mean_auc 0.8287 — saved before the model started overfitting
# past epoch ~10) on the held-out test set.
srun python3 test/centralized_multilabel.py --dataset nih --setting centralized
