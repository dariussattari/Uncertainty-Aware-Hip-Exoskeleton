#!/bin/bash
#SBATCH --job-name=aw-horizon
#SBATCH --output=ml/sci_rob/awareness/slurm/logs/horizon_%A_%a.out
#SBATCH --error=ml/sci_rob/awareness/slurm/logs/horizon_%A_%a.out
#SBATCH --array=0-1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Horizon sweep for the plain forecast ensemble (AW_HORIZONS, samples at 200 Hz; default "100 200"
# = 500 ms and 1 s; the 200 ms run already exists as forecast_plain).
set -euo pipefail
source "$HIPEXO_REPO/ml/sci_rob/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
: "${AW_HOLDOUT:?set AW_HOLDOUT}"
read -r -a HS <<< "${AW_HORIZONS:-100 200}"
H="${HS[$SLURM_ARRAY_TASK_ID]}"
python ml/sci_rob/awareness/forecast.py --holdout "$AW_HOLDOUT" --variant plain --horizon "$H" -- --folds "${AW_FOLDS:-1}" ${AW_FORECAST_ARGS:-}
echo "HORIZON $H OK for $AW_HOLDOUT"
