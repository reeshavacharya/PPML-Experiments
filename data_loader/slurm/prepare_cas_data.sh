#!/bin/bash
#SBATCH --job-name=cs_prp
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/data_loader/slurm/std_out_cas_prepare.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/data_loader/slurm/std_err_cas_prepare.log

# CPU/IO-bound only (zip join + extract) — no GPU requested on purpose.

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

srun python3 data_loader/cas_prepare.py extract
srun python3 data_loader/cas_prepare.py inspect --sample 30
