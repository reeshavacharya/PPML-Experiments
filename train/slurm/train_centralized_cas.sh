#!/bin/bash
#SBATCH --job-name=cs_ctr
#SBATCH --partition=cbcs
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_out_train_baseline_cas.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/train/slurm/std_err_train_baseline_cas.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate
set -e
ulimit -c 0

# Extract lr/weight_decay from hpo_best_centralized_cas.json (written by
# train/slurm/hpo_centralized_cas.sh when all trials finish). Exits early
# if the file is missing so a partial run is never started without HPO results.
if [ ! -f hpo_best_centralized_cas.json ]; then
    echo "ERROR: hpo_best_centralized_cas.json not found. Run CAS centralized HPO first." >&2
    exit 1
fi
LR=$(python3 -c "import json; print(json.load(open('hpo_best_centralized_cas.json'))['best_params']['lr'])")
WD=$(python3 -c "import json; print(json.load(open('hpo_best_centralized_cas.json'))['best_params']['weight_decay'])")
echo "HPO best: lr=${LR}  weight_decay=${WD}"

srun python3 train/centralized_baseline.py \
    --dataset cas \
    --lr "${LR}" \
    --weight_decay "${WD}" \
    --epochs 100 \
    --num_workers 4 \
    --resume

# Only reached on a genuine, non-preempted completion (set -e above kills the
# script before this line if srun is terminated or errors) — chains testing
# immediately rather than waiting on manual follow-up.
echo "Training complete. Submitting test job..."
sbatch test/slurm/test_centralized_baseline_cas.sh
