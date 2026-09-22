#!/bin/bash
#SBATCH --job-name=loto-probz
#SBATCH --output=ml/LOTO/slurm/logs/probz_%A_%a.out
#SBATCH --error=ml/LOTO/slurm/logs/probz_%A_%a.out
#SBATCH --array=0-2
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Stage 2 of the probabilistic ensemble, one task per array index: aggregate the twelve folds,
# retrain on all participants for the epoch count they chose, calibrate thresholds, evaluate,
# plot. Submitted with --dependency=afterok on the fold array.
set -euo pipefail
source "$HIPEXO_REPO/ml/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
TASK="${LOTO_TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== prob_ensemble stage 2, holding out $TASK ==="
for stage in collect final eval figures; do
  echo "--- $stage ---"
  python ml/LOTO/run.py --held-out "$TASK" --model prob_ensemble --prob-stage "$stage"
done
