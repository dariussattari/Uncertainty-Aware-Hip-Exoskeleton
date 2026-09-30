#!/bin/bash
# Stage the VAE package to the cluster. Run on the LAPTOP from anywhere.
#
#     bash ml/VAE/slurm/push.sh
#
# Only Python and SLURM files go up: no runs, no logs. ml/vanilla and ml/LOTO must already be
# on the cluster (ml/LOTO/slurm/push.sh); the VAE imports them.
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../.."
ssh "$HOST" "mkdir -p ~/$REMOTE/ml/VAE/slurm/logs"
rsync -az --stats \
  --include '*.py' --include 'slurm/' --include 'slurm/*.sbatch' --include 'slurm/*.sh' \
  --exclude '*' \
  ml/VAE/ "$HOST:~/$REMOTE/ml/VAE/" \
  | grep -E "Number of files transferred|Total transferred"
echo "pushed to $HOST:~/$REMOTE/ml/VAE"
