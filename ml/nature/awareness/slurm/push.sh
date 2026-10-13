#!/bin/bash
# Stage the Nature awareness code and its arrays to the cluster. Run on the LAPTOP from anywhere in the repo.
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../../.."
git describe --always --dirty --abbrev=7 > ml/nature/awareness/GIT_SHA          # the code version every run logs
ssh "$HOST" "mkdir -p ~/$REMOTE/ml/nature/awareness/slurm/logs ~/hipexo_data/nature_ae"
rsync -az --relative --exclude '__pycache__' --exclude 'runs/' --exclude 'logs/' \
  ml/common ml/nature/awareness "$HOST:~/$REMOTE/" --stats | { grep -E "Number of (regular )?files transferred" || true; }
rsync -az --stats data/processed_nature/ae/ "$HOST:~/hipexo_data/nature_ae/" | { grep -E "Number of (regular )?files transferred|Total transferred" || true; }
echo "pushed code ($(cat ml/nature/awareness/GIT_SHA)) and arrays to $HOST"
