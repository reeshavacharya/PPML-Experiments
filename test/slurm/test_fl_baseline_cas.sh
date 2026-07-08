#!/bin/bash
#SBATCH --job-name=test_fl_cas
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/test/slurm/std_out_test_fl_baseline_cas.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/test/slurm/std_err_test_fl_baseline_cas.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

# Evaluates the FL best-round CAS checkpoint on the same shared 200-case test
# set used by test_centralized_baseline_cas.sh, for a like-for-like comparison.
srun python3 test/centralized_baseline.py --dataset cas --setting fl
