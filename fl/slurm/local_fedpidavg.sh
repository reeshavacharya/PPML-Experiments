#!/bin/bash
#SBATCH --job-name=ft_all23
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_fedpidavg.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_fedpidavg.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

echo "=== FedPIDAvg run on $(hostname) at $(date) ==="
echo "GPU: $(nvidia-smi -L)"

# Clean stale lock file
rm -f fl/.gpu_lock

# Start server (no GPU needed)
CUDA_VISIBLE_DEVICES="" python3 -u fl/server.py \
    --rounds 100 \
    --server_address 127.0.0.1:8080 \
    --num_clients 23 \
    --alpha 0.45 --beta 0.45 --gamma 0.1 &
SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"
sleep 30

# Launch all 23 clients on the same GPU (file lock serializes GPU access)
for i in $(seq 1 23); do
    python3 -u fl/client.py \
        --client_id $i \
        --server_address 127.0.0.1:8080 \
        --num_workers 1 \
        > fl/slurm/std_out_client_${i}.log 2>&1 &
    echo "Client $i started (PID: $!)"
    sleep 2
done

echo "All 23 clients launched. Waiting for server to finish..."
wait $SERVER_PID
SERVER_EXIT=$?
echo "Server exited ($SERVER_EXIT). Cleaning up..."
wait

echo "=== FedPIDAvg run finished at $(date) ==="
