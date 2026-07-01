#!/bin/bash
cd /work/r/reeshav/PPML-Experiments-FL

# Clean stale state
rm -f fl/server_address.txt
rm -f fl/.gpu_lock_*

# Submit server (1 CPU node) + 7 grouped client jobs (1 GPU each)
sbatch fl/slurm/fedpidavg_server.sh
sbatch fl/slurm/fedpidavg_client.sh

echo "Submitted: 1 server + 7 client group jobs (23 institutions total)"
