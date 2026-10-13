#!/bin/bash
# Bring finished autoencoder runs into the Nature model-exploration notebooks folder. Run on the LAPTOP; repeatable.
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../../.."
DEST=notebooks/nature/model_exploration/runs/ae
mkdir -p "$DEST" ml/nature/awareness/slurm/logs
rsync -az --stats --exclude '*_smoke/' --exclude 'last.pt' "$HOST:~/$REMOTE/ml/nature/awareness/runs/ae/" "$DEST/" \
  | { grep -E "Number of (regular )?files transferred|Total transferred" || true; }
rsync -az "$HOST:~/$REMOTE/ml/nature/awareness/slurm/logs/" ml/nature/awareness/slurm/logs/ || true
find "$DEST" -name summary.json | sort
