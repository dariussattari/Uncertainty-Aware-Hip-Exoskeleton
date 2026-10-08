#!/bin/bash
#SBATCH --job-name=loto-single
#SBATCH --output=ml/sci_rob/LOTO/slurm/logs/single_%A_%a.out
#SBATCH --error=ml/sci_rob/LOTO/slurm/logs/single_%A_%a.out
#SBATCH --array=0-5
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# The two single-stage models: autoencoder and GAN, each for all three held-out tasks.
#
#   index 0 1 2 -> vanilla/AE   for SA ST TR
#   index 3 4 5 -> vanilla/GAN  for SA ST TR
#
# The autoencoder is minutes; the GAN's 500 fixed epochs are ~3 h on this laptop's GPU and
# less here. Six independent cells, so the column costs one cell's wall clock instead of six.
set -euo pipefail
source "$HIPEXO_REPO/ml/sci_rob/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
cell "$SLURM_ARRAY_TASK_ID" "${LOTO_SINGLE[@]}"
echo "=== $CELL_MODEL, holding out $CELL_TASK ==="
python ml/sci_rob/LOTO/run.py --held-out "$CELL_TASK" --model "$CELL_MODEL"
