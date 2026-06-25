#!/bin/bash
#SBATCH --job-name=fl_local
#SBATCH --partition=muma_2021
#SBATCH --qos=muma21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --output=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_out_local.log
#SBATCH --error=/work/r/reeshav/PPML-Experiments-FL/fl/slurm/std_err_local.log

cd /work/r/reeshav/PPML-Experiments-FL
source .venv/bin/activate

echo "=== FL Local run on $(hostname) at $(date) ==="
echo "GPUs available: $(nvidia-smi -L)"

# Start server in background (no GPU needed)
CUDA_VISIBLE_DEVICES="" python3 -u fl/server.py --rounds 100 --server_address 127.0.0.1:8080 &
SERVER_PID=$!
echo "Server started (PID: $SERVER_PID), waiting for gRPC to initialize..."
sleep 30

# Each client gets its own GPU and reduced workers to avoid shared memory exhaustion
CUDA_VISIBLE_DEVICES=0 python3 -u fl/client.py --client_id 1 --server_address 127.0.0.1:8080 --num_workers 2 \
    > fl/slurm/std_out_client_1.log 2>&1 &
CLIENT1_PID=$!
echo "Client 1 started (PID: $CLIENT1_PID) on GPU 0"

CUDA_VISIBLE_DEVICES=1 python3 -u fl/client.py --client_id 2 --server_address 127.0.0.1:8080 --num_workers 2 \
    > fl/slurm/std_out_client_2.log 2>&1 &
CLIENT2_PID=$!
echo "Client 2 started (PID: $CLIENT2_PID) on GPU 1"

CUDA_VISIBLE_DEVICES=0 python3 -u fl/client.py --client_id 3 --server_address 127.0.0.1:8080 --num_workers 2 \
    > fl/slurm/std_out_client_3.log 2>&1 &
CLIENT3_PID=$!
echo "Client 3 started (PID: $CLIENT3_PID) on GPU 0"

echo "All processes launched. Waiting for server to finish..."
wait $SERVER_PID
SERVER_EXIT=$?
echo "Server exited with code $SERVER_EXIT. Cleaning up clients..."
wait $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID

echo "=== FL Local run finished at $(date) ==="
