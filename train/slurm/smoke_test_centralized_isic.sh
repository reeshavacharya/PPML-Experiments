#!/bin/bash
#SBATCH --job-name=is_sct
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_out_smoke_isic.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_err_smoke_isic.log

# One-off sanity check before committing to the 48h Optuna HPO job
# (train/slurm/hpo_centralized_isic.sh) or the full 100-epoch run
# (train/slurm/train_centralized_isic.sh). Only 1 epoch — this pipeline has
# never run against real data/GPU, so the goal is to catch runtime bugs
# cheaply: timm ViT-B/16 pretrained-weight download, the ShadeOfGray/
# ColorJitter transform chain on real dermoscopy JPEGs, FocalLoss shape
# handling on (B, 8) logits, and the metrics CSV writing the right number of
# columns. Requires data/ISIC/isic_split.csv to already exist — run
# data_loader/slurm/prepare_isic_data.sh first if it doesn't.

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e
ulimit -c 0

echo "=== ISIC smoke test starting on $(hostname) at $(date) ==="

srun python3 -u train/centralized_classification.py \
    --dataset isic \
    --epochs 1 \
    --batch_size 16 \
    --loss focal \
    --num_workers 8

echo "=== ISIC smoke test finished at $(date) ==="
echo ""
echo "Check for:"
echo "  - Loss not NaN across the epoch"
echo "  - Val Balanced Accuracy > 0 and not NaN (train/centralized_isic_metrics.csv)"
echo "  - Val Macro AUC OvR populated (not blank/NaN) in the same CSV"
echo "  - checkpoint/vit_isic_centralized_best.pth was written"
echo "  - Peak RAM / VRAM / GPU Util columns populated (resource monitoring works)"
echo "  - Wall-clock time for the epoch, to sanity-check the 100-epoch budget"
