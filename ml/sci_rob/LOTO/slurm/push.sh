#!/bin/bash
# Stage code and arrays to the cluster. Run on your LAPTOP, from the repo root.
#
#     bash ml/sci_rob/LOTO/slurm/push.sh                 # code + data (first time, ~9.9 GB)
#     bash ml/sci_rob/LOTO/slurm/push.sh --code-only     # after editing code
#
# rsync, not scp, so a re-push only sends what changed and an interrupted transfer resumes.
set -euo pipefail
# NOTE: macOS ships rsync 2.6.9, which has no --info flag. Use --stats/-v only.
HOST="${HIPEXO_HOST:-dsattari@login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
DATA_REMOTE="${HIPEXO_DATA_REMOTE:-hipexo_data}"
cd "$(dirname "$0")/../../../.."

echo "==> code to $HOST:~/$REMOTE"
# --relative keeps ml/common and ml/sci_rob/<package> nested on the remote side exactly as here.
rsync -az --stats --delete --relative \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'runs/' --exclude '.venv' \
  --exclude 'slurm/logs/' \
  ml/common ml/sci_rob/vanilla ml/sci_rob/prob_ensemble ml/sci_rob/LOTO "$HOST:~/$REMOTE/"

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
  \$EDITOR ml/sci_rob/LOTO/slurm/env.sh          # set HIPEXO_ACCOUNT
  bash  ml/sci_rob/LOTO/slurm/setup.sh
  sbatch ml/sci_rob/LOTO/slurm/smoke.sbatch      # ~15 min, proves CUDA + data + one real cell
  bash  ml/sci_rob/LOTO/slurm/submit.sh          # the matrix
TXT
