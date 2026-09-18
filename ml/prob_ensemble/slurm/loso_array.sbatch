#!/bin/bash
#SBATCH --job-name=hipexo-loso
#SBATCH --output=logs/loso_%A_%a.out
#SBATCH --error=logs/loso_%A_%a.out
#SBATCH --array=0-11
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Stage 1: the twelve leave-one-subject-out folds, one per array task.
#
# This is the reason for running on a cluster at all. Sequentially these folds took about five
# hours; as an array they take the slowest single fold. Each task writes only its own
# loso/fold_NN_SUBJECT.json, so there is no shared-file race, and a task whose file already
# exists exits immediately -- which makes requeue-on-preemption free.
set -euo pipefail
mkdir -p logs
source "$(dirname "$0")/env.sh"
stage_data_to_tmp

python train.py fold \
  --index "$SLURM_ARRAY_TASK_ID" \
  --out "${HIPEXO_OUT:-ml/prob_ensemble/runs/prob_forecast}" \
  --max-epochs "${HIPEXO_MAX_EPOCHS:-60}" \
  --patience "${HIPEXO_PATIENCE:-10}" \
  --beta "${HIPEXO_BETA:-0.5}" \
  --warmup-epochs "${HIPEXO_WARMUP:-3}"
