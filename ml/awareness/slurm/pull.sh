#!/bin/bash
# Bring finished awareness runs home (everything but the forecast figures' PDFs). Run on the LAPTOP; repeatable.
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../.."
mkdir -p ml/awareness/runs ml/awareness/slurm/logs
rsync -az --stats --exclude '*.pdf' "$HOST:~/$REMOTE/ml/awareness/runs/" ml/awareness/runs/ | grep -E "Number of files transferred|Total transferred"
rsync -az "$HOST:~/$REMOTE/ml/awareness/slurm/logs/" ml/awareness/slurm/logs/ || true
find ml/awareness/runs -maxdepth 3 -name 'eval_test.json' | sort
