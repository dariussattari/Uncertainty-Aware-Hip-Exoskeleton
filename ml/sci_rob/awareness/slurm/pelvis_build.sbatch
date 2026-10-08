#!/bin/bash
#SBATCH --job-name=aw-pelvis-build
#SBATCH --output=ml/sci_rob/awareness/slurm/logs/pelvis_build_%j.out
#SBATCH --error=ml/sci_rob/awareness/slurm/logs/pelvis_build_%j.out
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Cut the 22-channel (16 + pelvis IMU) mode-complete arrays from the raw trials on the cluster.
# CPU only; the GPU job of the ablation depends on it. Idempotent: existing splits are kept.
set -euo pipefail
export HIPEXO_STAGE=0
source "$HIPEXO_REPO/ml/sci_rob/LOTO/slurm/env.sh"
: "${HIPEXO_PELVIS_DATA:?set HIPEXO_PELVIS_DATA (where AE_GAN/ with 22 channels is written)}"
python ml/sci_rob/LOTO/build_data.py --channels pelvis --family AE_GAN --out-root "$HIPEXO_PELVIS_DATA"
du -sh "$HIPEXO_PELVIS_DATA/AE_GAN"
echo "PELVIS BUILD OK -> $HIPEXO_PELVIS_DATA"
