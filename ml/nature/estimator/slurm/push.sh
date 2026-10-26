#!/bin/bash
# Stage the moment estimator code and its arrays to the cluster. Run on the LAPTOP from anywhere in the repo.
# The sbatch sources ml/nature/awareness/slurm/env.sh, so that folder goes too.
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../../.."
git describe --always --dirty --abbrev=7 > ml/nature/estimator/GIT_SHA          # the code version every run logs
ssh "$HOST" "mkdir -p ~/$REMOTE/ml/nature/estimator/slurm/logs ~/hipexo_data/nature_moment"
rsync -az --relative --exclude '__pycache__' --exclude 'runs/' --exclude 'logs/' \
  ml/common ml/nature/estimator ml/nature/awareness/slurm/env.sh "$HOST:~/$REMOTE/" --stats | { grep -E "Number of (regular )?files transferred" || true; }
rsync -az --stats data/processed_nature/moment/ "$HOST:~/hipexo_data/nature_moment/" | { grep -E "Number of (regular )?files transferred|Total transferred" || true; }
echo "pushed code ($(cat ml/nature/estimator/GIT_SHA)) and arrays to $HOST"
