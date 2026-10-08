#!/bin/bash
#SBATCH --job-name=aw-forecast
#SBATCH --output=ml/sci_rob/awareness/slurm/logs/forecast_%A_%a.out
#SBATCH --error=ml/sci_rob/awareness/slurm/logs/forecast_%A_%a.out
#SBATCH --array=0-1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Stage C: the 200 ms hip-angle forecast ensemble under the holdout.
#   0 -> plain           16 sensor channels
#   1 -> regime          16 channels + the regime probe's class probabilities (needs stage B)
# AW_FOLDS trims the leave-one-subject-out epoch selection (default 1 fold: the fast first pass).
set -euo pipefail
source "$HIPEXO_REPO/ml/sci_rob/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
: "${AW_HOLDOUT:?set AW_HOLDOUT}"
VARIANTS=(plain regime)
V="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
python ml/sci_rob/awareness/forecast.py --holdout "$AW_HOLDOUT" --variant "$V" -- --folds "${AW_FOLDS:-1}" ${AW_FORECAST_ARGS:-}
echo "FORECAST $V OK for $AW_HOLDOUT"
