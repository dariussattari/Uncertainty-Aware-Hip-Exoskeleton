#!/bin/bash
#SBATCH --job-name=loto-vens
#SBATCH --output=ml/sci_rob/LOTO/slurm/logs/vens_%A_%a.out
#SBATCH --error=ml/sci_rob/LOTO/slurm/logs/vens_%A_%a.out
#SBATCH --array=0-8
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# The three vanilla ensembles: gait phase, summed correlation, 200 ms forecast -- each for all
# three held-out tasks.
#
#   0 1 2 -> Ensemble_GP        SA ST TR
#   3 4 5 -> Ensemble_synthetic SA ST TR
#   6 7 8 -> Ensemble_forecast  SA ST TR
#
# Unlike ml/sci_rob/prob_ensemble, ml/sci_rob/vanilla's two-stage protocol runs its twelve leave-one-subject-out
# folds INSIDE one process -- train_paper_protocol() owns the loop and there is no per-fold
# entry point. So a cell is one job, not an array, and the parallelism here is across the nine
# cells rather than within them. Nine jobs at ~6-9 h each still finish in one cell's wall clock,
# which is the part that matters; splitting the folds out would be a refactor of vanilla for a
# second-order gain.
set -euo pipefail
source "$HIPEXO_REPO/ml/sci_rob/LOTO/slurm/env.sh"
cell "$SLURM_ARRAY_TASK_ID" "${LOTO_VANILLA_ENS[@]}"
# Only the gait-phase ensemble reads the labelled `ensemble` arrays; the other two read AE_GAN.
if [[ "$CELL_MODEL" == *Ensemble_GP ]]; then
  export HIPEXO_STAGE_FAMILY="ensemble"
else
  export HIPEXO_STAGE_FAMILY="AE_GAN"
fi
stage_data_to_tmp
echo "=== $CELL_MODEL, holding out $CELL_TASK ==="
python ml/sci_rob/LOTO/run.py --held-out "$CELL_TASK" --model "$CELL_MODEL"
