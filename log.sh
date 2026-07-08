nohup bash -c '
while true; do
  squeue -u $USER > /work/r/reeshav/PPML-Experiments-FL/queue.txt
  sinfo -s > /work/r/reeshav/PPML-Experiments-FL/info.txt
  scontrol show node > /work/r/reeshav/PPML-Experiments-FL/nodes.txt
  scontrol show job > /work/r/reeshav/PPML-Experiments-FL/jobs.txt
  sleep 30
done
' > /dev/null 2>&1 &
disown