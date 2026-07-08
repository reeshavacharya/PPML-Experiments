#!/bin/bash
#SBATCH --job-name=is_ssv
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_smoke_server_isic.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_smoke_server_isic.log

# One-off sanity check before the real FedAvg ISIC run (fl/slurm/fedavg_server_isic.sh)
# or the FL Optuna HPO (fl/hpo_coordinator.py --dataset isic). Only 3 rounds — this
# is the first time the generalized CASFedAvgStrategy (metric_key="val_balanced_acc")
# and the 6-natural-center ISIC partitioning have run against real data. --seed 9999
# sandboxes the smoke test's checkpoint/resume/summary files away from the seed-42
# filenames the real run will use, so nothing here needs to be cleaned up before the
# real run starts.

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e

echo "=== ISIC FL smoke test — Server starting on $(hostname) at $(date) ==="

srun python3 -u fl/server_isic.py \
    --rounds 3 \
    --server_address 0.0.0.0:8081 \
    --num_clients 6 \
    --seed 9999

echo "=== ISIC FL smoke test — Server finished at $(date) ==="
echo ""
echo "Check for:"
echo "  - fl/training_summary_isic_seed9999.json — best_val_balanced_acc > 0, algorithm=FedAvg"
echo "  - checkpoint/vit_isic_fl_seed9999_best.pth was written"
echo "  - Server log shows all 6 clients connecting and contributing to aggregation"
