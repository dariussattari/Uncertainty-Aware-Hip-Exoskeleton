#!/bin/bash
# Submit the full protocol as a dependent chain: twelve folds in parallel, then stage 2.
#
#     bash slurm/submit.sh
#     bash slurm/submit.sh --requeue        # cheaper, preemptible partition
#
# Stage 2 runs under --dependency=afterok, so it starts only if every fold succeeded. If some
# fold fails, fix it, re-run just that task (`sbatch --array=<n> slurm/loso_array.sbatch`),
# then submit final.sbatch on its own -- completed folds are skipped, not repeated.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh >/dev/null

PART="$HIPEXO_PART"
EXTRA=""
if [[ "${1:-}" == "--requeue" ]]; then
  PART="gpu_requeue"
  EXTRA="--requeue"
  echo "using the preemptible partition: every stage is idempotent, so a preempted fold"
  echo "costs at most its own runtime."
fi

COMMON="--account=$HIPEXO_ACCOUNT --partition=$PART $EXTRA"
mkdir -p slurm/logs

jid_loso=$(sbatch --parsable $COMMON slurm/loso_array.sbatch)
echo "stage 1 (12 folds, array) : $jid_loso"

jid_final=$(sbatch --parsable $COMMON --dependency=afterok:$jid_loso slurm/final.sbatch)
echo "stage 2 (collect + final) : $jid_final   [waits on $jid_loso]"

echo
echo "watch:    squeue -u $USER"
echo "logs:     tail -f ml/prob_ensemble/slurm/logs/loso_${jid_loso}_0.out"
echo "progress: ls ml/prob_ensemble/runs/prob_forecast/loso/    # one file per finished fold"
