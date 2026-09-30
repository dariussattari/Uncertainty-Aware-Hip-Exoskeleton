#!/bin/bash
#SBATCH --job-name=aw-experts
#SBATCH --output=ml/awareness/slurm/logs/experts_%A_%a.out
#SBATCH --error=ml/awareness/slurm/logs/experts_%A_%a.out
#SBATCH --array=0-1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Stage C': one forecast ensemble per regime present in training (AW_REGIMES, space-separated,
# default "level ascend"), each trained on that regime's windows only. Scoring across experts
# runs on the laptop afterwards (experts.py score).
set -euo pipefail
source "$HIPEXO_REPO/ml/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
: "${AW_HOLDOUT:?set AW_HOLDOUT}"
read -r -a REGIMES <<< "${AW_REGIMES:-level ascend}"
R="${REGIMES[$SLURM_ARRAY_TASK_ID]}"
python ml/awareness/experts.py train --holdout "$AW_HOLDOUT" --regime "$R" -- --folds "${AW_FOLDS:-1}" ${AW_FORECAST_ARGS:-}
echo "EXPERT $R OK for $AW_HOLDOUT"
