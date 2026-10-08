#!/bin/bash
#SBATCH --job-name=aw-encoder
#SBATCH --output=ml/sci_rob/awareness/slurm/logs/encoder_%j.out
#SBATCH --error=ml/sci_rob/awareness/slurm/logs/encoder_%j.out
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Stages A and B of the awareness layer for one holdout: train the deterministic encoder
# (about an hour at the AE's settings), encode every window, fit the regime probes.
set -euo pipefail
source "$HIPEXO_REPO/ml/sci_rob/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
: "${AW_HOLDOUT:?set AW_HOLDOUT (a name from ml/sci_rob/LOTO/splits.HOLDOUTS)}"
python ml/sci_rob/awareness/encoder.py --holdout "$AW_HOLDOUT" ${AW_ENCODER_ARGS:-}
python ml/sci_rob/awareness/regime.py  --holdout "$AW_HOLDOUT"
echo "ENCODER+REGIME OK for $AW_HOLDOUT"
