#!/bin/bash
# Bring finished awareness runs home (everything but the forecast figures' PDFs). Run on the LAPTOP; repeatable.
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../../.."
mkdir -p ml/sci_rob/awareness/runs ml/sci_rob/awareness/slurm/logs
rsync -az --stats --exclude '*.pdf' "$HOST:~/$REMOTE/ml/sci_rob/awareness/runs/" ml/sci_rob/awareness/runs/ | grep -E "Number of files transferred|Total transferred"
rsync -az "$HOST:~/$REMOTE/ml/sci_rob/awareness/slurm/logs/" ml/sci_rob/awareness/slurm/logs/ || true
find ml/sci_rob/awareness/runs -maxdepth 3 -name 'eval_test.json' | sort
