#!/bin/bash
# Bring finished moment estimator runs and logs back to the laptop. Run on the LAPTOP; repeatable. last.pt stays on the cluster.
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../../.."
mkdir -p ml/nature/estimator/runs ml/nature/estimator/slurm/logs
rsync -az --stats --exclude '*_smoke/' --exclude 'last.pt' "$HOST:~/$REMOTE/ml/nature/estimator/runs/" ml/nature/estimator/runs/ \
  | { grep -E "Number of (regular )?files transferred|Total transferred" || true; }
rsync -az "$HOST:~/$REMOTE/ml/nature/estimator/slurm/logs/" ml/nature/estimator/slurm/logs/ || true
echo "finished runs: $(find ml/nature/estimator/runs -name summary.json | wc -l | tr -d ' ')"
