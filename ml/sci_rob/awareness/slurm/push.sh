#!/bin/bash
# Stage the awareness package to the cluster (code only). Run on the LAPTOP.
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../../.."
ssh "$HOST" "mkdir -p ~/$REMOTE/ml/sci_rob/awareness/slurm/logs"
rsync -az --stats --include '*.py' --include 'slurm/' --include 'slurm/*.sbatch' --include 'slurm/*.sh' --exclude '*' \
  ml/sci_rob/awareness/ "$HOST:~/$REMOTE/ml/sci_rob/awareness/" | grep -E "Number of files transferred|Total transferred"
rsync -az --stats --relative --exclude '__pycache__' ml/common ml/sci_rob/VAE/vae_train.py ml/sci_rob/VAE/vae_paths.py "$HOST:~/$REMOTE/" | grep -E "Number of files transferred"
echo "pushed to $HOST:~/$REMOTE/ml/sci_rob/awareness"
