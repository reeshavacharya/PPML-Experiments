#!/bin/bash
#SBATCH --job-name=cs_chp
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_out_hpo_centralized_cas.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_err_hpo_centralized_cas.log

ulimit -c 0   # disable core dumps — Bus error crashes were filling the NFS filesystem

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

echo "=== CAS Centralized HPO starting on $(hostname) at $(date) ==="

python3 -u train/hpo_centralized.py \
    --dataset cas \
    --n_trials 10 \
    --hpo_epochs 6 \
    --hpo_max_steps 50

echo "=== CAS Centralized HPO finished at $(date) ==="
