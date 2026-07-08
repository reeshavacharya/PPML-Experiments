#!/bin/bash
#SBATCH --job-name=test_cas
#SBATCH --partition=cbcs
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/test/slurm/std_out_test_baseline_cas.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/test/slurm/std_err_test_baseline_cas.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

# Evaluates the centrally-trained CAS checkpoint on the shared 200-case test set.
# For the FL checkpoint on the same test set, pass --setting fl instead.
srun python3 test/centralized_baseline.py --dataset cas --setting centralized
