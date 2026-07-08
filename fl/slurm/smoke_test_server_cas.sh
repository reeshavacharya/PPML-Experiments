#!/bin/bash
#SBATCH --job-name=cs_ssv
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_smoke_server_cas.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_smoke_server_cas.log

# One-off sanity check before the real FedAvg CAS run (fl/slurm/fedavg_server_cas.sh)
# or the FL Optuna HPO (fl/hpo_coordinator.py --dataset cas). Only 3 rounds — this
# is the first time CASFedAvgStrategy and the 5-IID-client CAS partitioning have
# run against real data. --seed 9999 sandboxes the smoke test's checkpoint/resume/
# summary files away from the seed-42 filenames the real run will use, so nothing
# here needs to be cleaned up before the real run starts.

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

echo "=== CAS FL smoke test — Server starting on $(hostname) at $(date) ==="

srun python3 -u fl/server.py \
    --dataset cas \
    --strategy fedavg \
    --rounds 3 \
    --server_address 0.0.0.0:8083 \
    --num_clients 5 \
    --seed 9999

echo "=== CAS FL smoke test — Server finished at $(date) ==="
echo ""
echo "Check for:"
echo "  - fl/training_summary_cas_seed9999.json — best_val_dice > 0, algorithm=FedAvg"
echo "  - checkpoint/swin_unetr_fl_cas_seed9999_best.pth was written"
echo "  - Server log shows all 5 clients connecting and contributing to aggregation"
