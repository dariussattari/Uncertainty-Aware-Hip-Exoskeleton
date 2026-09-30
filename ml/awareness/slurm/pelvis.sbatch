#!/bin/bash
#SBATCH --job-name=aw-pelvis
#SBATCH --output=ml/awareness/slurm/logs/pelvis_%j.out
#SBATCH --error=ml/awareness/slurm/logs/pelvis_%j.out
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# The pelvis-IMU ablation: stages A and B of the awareness layer on the 22-channel arrays,
# written beside the 16-channel runs as encoder_<tag>/ and regime_<tag>/.
set -euo pipefail
: "${HIPEXO_PELVIS_DATA:?set HIPEXO_PELVIS_DATA (directory holding the 22-channel AE_GAN/)}"
export HIPEXO_LOTO_DATA="$HIPEXO_PELVIS_DATA"
source "$HIPEXO_REPO/ml/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
: "${AW_HOLDOUT:?set AW_HOLDOUT (a name from ml/LOTO/splits.HOLDOUTS)}"
TAG="${AW_TAG:-pelvis}"
python ml/awareness/encoder.py --holdout "$AW_HOLDOUT" --stage "encoder_$TAG" --data-root "$HIPEXO_LOTO_DATA" ${AW_ENCODER_ARGS:-}
python ml/awareness/regime.py  --holdout "$AW_HOLDOUT" --encoder-stage "encoder_$TAG" --stage "regime_$TAG"
echo "PELVIS ABLATION OK for $AW_HOLDOUT ($TAG)"
