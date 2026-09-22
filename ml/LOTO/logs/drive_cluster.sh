#!/bin/bash
# Wait for the smoke job, then release the nine vanilla-ensemble cells if it passed.
H=login.rc.fas.harvard.edu
SMOKE=47403524
LOG=~/Uncertainty-Aware-Hip-Exoskeleton/ml/LOTO/slurm/logs/smoke_${SMOKE}.out

echo "waiting on smoke $SMOKE ..."
while ssh $H "squeue -h -j $SMOKE -o %t" 2>/dev/null | grep -qE 'R|PD|CG'; do sleep 60; done

if ssh $H "grep -q 'SMOKE OK' $LOG" 2>/dev/null; then
  echo "SMOKE OK -- releasing vanilla_ens"
  ssh $H 'bash -lc "
    cd ~/Uncertainty-Aware-Hip-Exoskeleton
    export HIPEXO_REPO=\$HOME/Uncertainty-Aware-Hip-Exoskeleton
    export HIPEXO_LOTO_DATA=\$HIPEXO_REPO/data/processed_all
    sbatch --account=pslade_lab --partition=gpu \
      --export=ALL,HIPEXO_REPO=\$HIPEXO_REPO,HIPEXO_LOTO_DATA=\$HIPEXO_LOTO_DATA \
      ml/LOTO/slurm/vanilla_ens.sbatch"' 2>&1 | grep -v "post-quantum\|store now"
else
  echo "SMOKE DID NOT PASS -- vanilla_ens NOT submitted. Last lines:"
  ssh $H "tail -25 $LOG" 2>&1 | grep -v "post-quantum\|store now"
fi
