#!/bin/bash
#SBATCH --job-name=loto-probf
#SBATCH --output=ml/LOTO/slurm/logs/probf_%A_%a.out
#SBATCH --error=ml/LOTO/slurm/logs/probf_%A_%a.out
#SBATCH --array=0-35
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Stage 1 of the probabilistic ensemble: 3 held-out tasks x 12 leave-one-subject-out folds.
#
#   index = task_index * 12 + fold_index
#
# This one DOES have a per-fold entry point, so all 36 folds run concurrently. Each writes only
# its own loso/fold_NN_SUBJECT.json, so there is no shared-file race, and a fold whose file
# already exists exits immediately -- which makes --requeue on a preemptible partition nearly
# free.
set -euo pipefail
source "$HIPEXO_REPO/ml/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
TASK="${LOTO_TASKS[$(( SLURM_ARRAY_TASK_ID / 12 ))]}"
FOLD=$(( SLURM_ARRAY_TASK_ID % 12 ))
echo "=== prob_ensemble fold $FOLD, holding out $TASK ==="
python ml/LOTO/run.py --held-out "$TASK" --model prob_ensemble --prob-stage fold --fold-index "$FOLD"
