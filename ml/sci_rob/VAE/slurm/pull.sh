#!/bin/bash
# Bring finished VAE runs home. Run on the LAPTOP; safe to repeat while cells are still running.
#
#     bash ml/sci_rob/VAE/slurm/pull.sh
#
# Smoke runs (*_smoke) and SLURM logs are pulled too, into their usual ignored places, so a
# failed cell can be read locally. Checkpoints are small for this model and are included.
set -euo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../../.."
mkdir -p ml/sci_rob/VAE/runs ml/sci_rob/VAE/slurm/logs
rsync -az --stats "$HOST:~/$REMOTE/ml/sci_rob/VAE/runs/" ml/sci_rob/VAE/runs/ \
  | grep -E "Number of files transferred|Total transferred"
rsync -az "$HOST:~/$REMOTE/ml/sci_rob/VAE/slurm/logs/" ml/sci_rob/VAE/slurm/logs/ || true
.venv/bin/python ml/sci_rob/VAE/vae_train.py --list
