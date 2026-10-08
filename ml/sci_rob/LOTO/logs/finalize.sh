#!/bin/bash
# Wait for the cluster queue to drain, then do the final collection and write the doc section.
cd /Users/dariussattari/Documents/thesis/Uncertainty-Aware-Hip-Exoskeleton
H=login.rc.fas.harvard.edu
echo "waiting for the queue to drain..."
empty=0
for i in $(seq 1 600); do
  n=$(ssh -o BatchMode=yes $H "squeue -h -u \$USER -o '%j' 2>/dev/null | grep -c loto" 2>/dev/null)
  n=${n:-1}
  if [ "$n" -eq 0 ]; then empty=$((empty+1)); else empty=0; fi
  [ "$empty" -ge 2 ] && break        # two consecutive empties: not a gap between arrays
  sleep 60
done
echo "[$(date +%H:%M)] queue drained -- final collection"
bash ml/sci_rob/LOTO/slurm/collect.sh 2>&1 | tail -40
echo
echo "=== FINAL ==="
.venv/bin/python ml/sci_rob/LOTO/summarize.py 2>&1 | sed -n '1,30p'
