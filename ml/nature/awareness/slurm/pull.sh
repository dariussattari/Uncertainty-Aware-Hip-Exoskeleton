#!/bin/bash
# Bring finished autoencoder and VAE runs into the Nature model-exploration notebooks folder. Run on the LAPTOP; repeatable.
# --copy-dirlinks: run folders on the cluster may be symlinks into lab storage (slurm/move_runs.sh); copy what they hold.
# runs/ae -> notebooks/nature/model_exploration/runs/ae (notebooks 01, 05); runs/vae -> .../runs/vae (notebook 02, and 05);
# runs/ensemble -> .../runs/ensemble (notebook 10, the gait-phase ensemble baseline).
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../../.."
mkdir -p ml/nature/awareness/slurm/logs
for family in ae vae ensemble; do
  DEST="notebooks/nature/model_exploration/runs/$family"
  if ssh "$HOST" "test -d ~/$REMOTE/ml/nature/awareness/runs/$family"; then
    mkdir -p "$DEST"
    echo "runs/$family:"
    rsync -az --copy-dirlinks --stats --exclude '*_smoke/' --exclude 'last.pt' "$HOST:~/$REMOTE/ml/nature/awareness/runs/$family/" "$DEST/" \
      | { grep -E "Number of (regular )?files transferred|Total transferred" || true; }
  else
    echo "runs/$family: none on the cluster yet"
  fi
done
# the training-class ablation (train_class.sbatch) -> notebooks/nature/model_exploration/runs_train_class (notebook 07); kept
# outside runs/ so the ID-safe notebooks never pick these runs up
if ssh "$HOST" "test -d ~/$REMOTE/ml/nature/awareness/runs_train_class"; then
  mkdir -p notebooks/nature/model_exploration/runs_train_class
  echo "runs_train_class:"
  rsync -az --copy-dirlinks --stats --exclude '*_smoke/' --exclude 'last.pt' "$HOST:~/$REMOTE/ml/nature/awareness/runs_train_class/" \
    notebooks/nature/model_exploration/runs_train_class/ | { grep -E "Number of (regular )?files transferred|Total transferred" || true; }
fi
rsync -az "$HOST:~/$REMOTE/ml/nature/awareness/slurm/logs/" ml/nature/awareness/slurm/logs/ || true
echo "finished runs: ID-safe $(find notebooks/nature/model_exploration/runs -name summary.json | wc -l | tr -d ' '), " \
     "training-class ablation $(find notebooks/nature/model_exploration/runs_train_class -name summary.json 2>/dev/null | wc -l | tr -d ' ')"
