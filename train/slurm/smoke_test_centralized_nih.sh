#!/bin/bash
#SBATCH --job-name=nh_sct
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=mdc-1057-18-4
#SBATCH --time=04:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_out_smoke_nih.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_err_smoke_nih.log

# One-off sanity check before committing to the 48h Optuna HPO job
# (train/slurm/hpo_centralized_nih.sh) or the full 50-epoch run
# (train/slurm/train_centralized_nih.sh). Only 1 epoch — this pipeline has
# never run against real data/GPU, so the goal is to catch runtime bugs
# cheaply: the patient-wise split manifest, the 12-folder image-index scan,
# BCEWithLogitsLoss(pos_weight=...) shape handling on (B, 14) logits, and the
# metrics CSV writing the right number of columns. Requires
# data/NIH-Chest/nih_split.csv to already exist — run
# data_loader/slurm/prepare_nih_data.sh first if it doesn't.

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e
ulimit -c 0

echo "=== NIH smoke test starting on $(hostname) at $(date) ==="

srun python3 -u train/centralized_multilabel.py \
    --dataset nih \
    --epochs 1 \
    --batch_size 16 \
    --loss bce \
    --num_workers 8

echo "=== NIH smoke test finished at $(date) ==="
echo ""
echo "Check for:"
echo "  - Loss not NaN across the epoch"
echo "  - Val Mean AUC ROC > 0 and not NaN (train/centralized_nih_metrics.csv)"
echo "  - Per-label AUC/Sensitivity columns populated (not blank/NaN) in the same CSV"
echo "  - checkpoint/vit_nih_centralized_best.pth was written"
echo "  - Peak RAM / VRAM / GPU Util columns populated (resource monitoring works)"
echo "  - Wall-clock time for the epoch, to sanity-check the 50-epoch budget"
