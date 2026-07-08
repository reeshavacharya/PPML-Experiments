#!/bin/bash
#SBATCH --job-name=cs_sct
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --exclude=mdc-1057-13-8,mdc-1057-18-1,mdc-1057-18-4
#SBATCH --time=08:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_out_smoke_cas.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_err_smoke_cas.log

# One-off sanity check before committing to the 72h Optuna HPO job
# (train/slurm/hpo_centralized_cas.sh) or the full 100-epoch run
# (train/slurm/train_centralized_cas.sh). Only 1 epoch — this has never run
# against real data/GPU, so the goal is to catch runtime bugs cheaply:
# shape mismatches in DiceCELoss(softmax=True, to_onehot_y=True), the
# CropForegroundd/ScaleIntensityRanged interaction on real CT intensities,
# SlidingWindowInferer on the actual resampled volume size, clDice on real
# predictions, and the metrics CSV writing the right number of columns.
#
# Node/resource notes (from nodes.txt/queue.txt/info.txt, muma_2021):
# mdc-1057-13-8 and mdc-1057-18-1 are typically fully GPU-allocated and
# mdc-1057-18-4 is DRAINed (disk >95% full) — excluded so this doesn't queue
# behind them; --gres=gpu:1 already rules out the GPU-less 13-9/10/11/12/13
# nodes on its own. cpus-per-task bumped 8->16 and --num_workers 4->8 below:
# the previous run (job 33335545, cancelled mid-epoch) processed only ~7/700
# training steps in ~2.5 min on 8 cpus/4 workers — that pace would need
# several hours per epoch, likely CPU-bound on Spacingd's resampling of full
# CT volumes rather than GPU compute, so more DataLoader workers should help
# hide it. Only 1 epoch here (not 2) to keep the smoke test itself bounded;
# if this still runs long, the planned 72h budget for the real 100-epoch job
# (train_centralized_cas.sh) needs revisiting once we have a real per-epoch
# timing from this run.

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e
ulimit -c 0

echo "=== CAS smoke test starting on $(hostname) at $(date) ==="

srun python3 -u train/centralized_baseline.py \
    --dataset cas \
    --epochs 1 \
    --num_workers 8

echo "=== CAS smoke test finished at $(date) ==="
echo ""
echo "Check for:"
echo "  - Loss not NaN across the epoch"
echo "  - Val Dice > 0 and not NaN (train/centralized_baseline_cas_metrics.csv)"
echo "  - HD95 / clDice columns populated (not blank) in the same CSV"
echo "  - checkpoint/swin_unetr_centralized_cas_best.pth was written"
echo "  - Peak RAM / VRAM / GPU Util columns populated (resource monitoring works)"
echo "  - Wall-clock time for the epoch, to sanity-check the 100-epoch budget"
