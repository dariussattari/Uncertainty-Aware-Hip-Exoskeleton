#!/bin/bash
# Stage code and arrays to the cluster. Run on your LAPTOP, from the repo root.
#
#     bash ml/LOTO/slurm/push.sh                 # code + data (first time, ~9.9 GB)
#     bash ml/LOTO/slurm/push.sh --code-only     # after editing code
#
# rsync, not scp, so a re-push only sends what changed and an interrupted transfer resumes.
set -euo pipefail
# NOTE: macOS ships rsync 2.6.9, which has no --info flag. Use --stats/-v only.
HOST="${HIPEXO_HOST:-dsattari@login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
DATA_REMOTE="${HIPEXO_DATA_REMOTE:-hipexo_data}"
cd "$(dirname "$0")/../../.."

echo "==> code to $HOST:~/$REMOTE"
rsync -az --stats --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'runs/' --exclude '.venv' \
  --exclude 'ml/LOTO/slurm/logs/' \
  ml/vanilla ml/prob_ensemble ml/LOTO "$HOST:~/$REMOTE/ml/"

if [[ "${1:-}" != "--code-only" ]]; then
  echo "==> arrays to $HOST:~/$DATA_REMOTE  (~9.9 GB; resumable, rerun if interrupted)"
  rsync -a --stats --partial \
    data/processed_all "$HOST:~/$DATA_REMOTE/"
fi

echo "==> linking ~/$REMOTE/data -> ~/$DATA_REMOTE so run.py finds the repo root"
ssh "$HOST" "mkdir -p ~/$REMOTE/data && ln -sfn ~/$DATA_REMOTE/processed_all ~/$REMOTE/data/processed_all && ls -l ~/$REMOTE/data/"

cat <<TXT

on the cluster, once:
  cd ~/$REMOTE
  \$EDITOR ml/LOTO/slurm/env.sh          # set HIPEXO_ACCOUNT
  bash  ml/LOTO/slurm/setup.sh
  sbatch ml/LOTO/slurm/smoke.sbatch      # ~15 min, proves CUDA + data + one real cell
  bash  ml/LOTO/slurm/submit.sh          # the matrix
TXT
