#!/bin/bash
# Poll the cluster and bring results home as they appear. Ends when the queue is empty AND
# the vanilla_ens array has been seen (so it cannot finish during the gap before that array
# is released by drive_cluster.sh).
cd /Users/dariussattari/Documents/thesis/Uncertainty-Aware-Hip-Exoskeleton
H=login.rc.fas.harvard.edu
seen_vens=0
for i in $(seq 1 400); do
  q=$(ssh -o BatchMode=yes $H "squeue -h -u \$USER -o '%j'" 2>/dev/null)
  echo "$q" | grep -q loto-vens && seen_vens=1
  n=$(echo "$q" | grep -c loto || true)
  echo "[$(date +%H:%M)] $n job(s) in queue, vanilla_ens seen=$seen_vens"
  bash ml/LOTO/slurm/collect.sh --no-doc 2>&1 | grep -E "Number of files transferred|PROBLEM|✗|checked [0-9]+ eval" || true
  if [ "$n" -eq 0 ] && [ "$seen_vens" -eq 1 ]; then
    echo "ALL CLUSTER JOBS FINISHED"
    break
  fi
  sleep 600
done
