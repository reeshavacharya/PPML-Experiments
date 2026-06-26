#!/bin/bash
cd /work/r/reeshav/PPML-Experiments-FL

# Remove stale address file so clients wait for the new server
rm -f fl/server_address.txt

sbatch fl/slurm/server.sh
sbatch fl/slurm/client_1.sh
sbatch fl/slurm/client_2.sh
sbatch fl/slurm/client_3.sh
