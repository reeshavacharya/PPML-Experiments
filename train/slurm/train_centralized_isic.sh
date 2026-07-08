#!/bin/bash
#SBATCH --job-name=is_ctr
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_out_train_baseline_isic.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_err_train_baseline_isic.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e
ulimit -c 0

# Read tuned lr/weight_decay from hpo_best_centralized_isic.json (written by
# train/hpo_centralized.py --dataset isic) if it exists, so this script always
# trains with HPO-tuned hyperparameters once available, with no manual
# copy-paste step. Falls back to centralized_classification.py's own argparse
# defaults (with a clear warning) if HPO hasn't completed yet.
HPO_JSON="hpo_best_centralized_isic.json"
EXTRA_ARGS=()
if [ -f "$HPO_JSON" ]; then
    LR=$(python3 -c "import json; print(json.load(open('$HPO_JSON'))['best_params']['lr'])")
    WD=$(python3 -c "import json; print(json.load(open('$HPO_JSON'))['best_params']['weight_decay'])")
    echo "Using HPO-tuned hyperparameters from $HPO_JSON: lr=$LR weight_decay=$WD"
    EXTRA_ARGS=(--lr "$LR" --weight_decay "$WD")
else
    echo "WARNING: $HPO_JSON not found — training with script defaults (NOT HPO-tuned)."
fi

srun python3 train/centralized_classification.py \
    --dataset isic \
    --epochs 100 \
    --batch_size 32 \
    --loss focal \
    --num_workers 4 \
    "${EXTRA_ARGS[@]}"
