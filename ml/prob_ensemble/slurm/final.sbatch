#!/bin/bash
#SBATCH --job-name=hipexo-final
#SBATCH --output=logs/final_%j.out
#SBATCH --error=logs/final_%j.out
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Stage 2: aggregate the folds, retrain on all twelve participants, calibrate, evaluate.
# Submitted with --dependency=afterok on the array, so it starts only if every fold succeeded.
set -euo pipefail
mkdir -p logs
source "$(dirname "$0")/env.sh"
stage_data_to_tmp

OUT="${HIPEXO_OUT:-ml/prob_ensemble/runs/prob_forecast}"

python train.py collect --out "$OUT"
python train.py final   --out "$OUT"
python evaluate.py      --out "$OUT" --split test
python evaluate.py      --out "$OUT" --split val

echo
echo "Done. Copy the results back with (from your laptop):"
echo "  scp -r ${USER}@boslogin.rc.fas.harvard.edu:${HIPEXO_REPO}/${OUT} ./ml/prob_ensemble/runs/"
