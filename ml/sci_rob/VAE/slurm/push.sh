#!/bin/bash
# Stage the VAE package to the cluster. Run on the LAPTOP from anywhere.
#
#     bash ml/sci_rob/VAE/slurm/push.sh
#
# Only Python and SLURM files go up: no runs, no logs. ml/sci_rob/vanilla and ml/sci_rob/LOTO must already be
# on the cluster (ml/sci_rob/LOTO/slurm/push.sh); the VAE imports them.
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../../.."
ssh "$HOST" "mkdir -p ~/$REMOTE/ml/sci_rob/VAE/slurm/logs"
rsync -az --stats \
  --include '*.py' --include 'slurm/' --include 'slurm/*.sbatch' --include 'slurm/*.sh' \
  --exclude '*' \
  ml/sci_rob/VAE/ "$HOST:~/$REMOTE/ml/sci_rob/VAE/" \
  | grep -E "Number of files transferred|Total transferred"
rsync -az --stats --relative --exclude '__pycache__' ml/common "$HOST:~/$REMOTE/" | grep -E "Number of files transferred"
echo "pushed to $HOST:~/$REMOTE/ml/sci_rob/VAE and ml/common"
