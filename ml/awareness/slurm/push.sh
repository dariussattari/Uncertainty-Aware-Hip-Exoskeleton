#!/bin/bash
# Stage the awareness package to the cluster (code only). Run on the LAPTOP.
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../.."
ssh "$HOST" "mkdir -p ~/$REMOTE/ml/awareness/slurm/logs"
rsync -az --stats --include '*.py' --include 'slurm/' --include 'slurm/*.sbatch' --include 'slurm/*.sh' --exclude '*' \
  ml/awareness/ "$HOST:~/$REMOTE/ml/awareness/" | grep -E "Number of files transferred|Total transferred"
rsync -az --stats ml/VAE/vae_train.py ml/VAE/vae_model.py ml/VAE/vae_paths.py "$HOST:~/$REMOTE/ml/VAE/" | grep -E "Number of files transferred"
echo "pushed to $HOST:~/$REMOTE/ml/awareness"
