#!/bin/bash
# Submit the whole leave-one-task-out matrix. Run from the repo root on a LOGIN node.
#
#     bash ml/LOTO/slurm/submit.sh              # everything
#     bash ml/LOTO/slurm/submit.sh single       # just the AE + GAN array
#     bash ml/LOTO/slurm/submit.sh --requeue    # preemptible partition, cheaper, may restart
#
# 18 cells in four arrays:
#   single       6 jobs   AE and GAN x 3 tasks
#   vanilla_ens  9 jobs   the three vanilla ensembles x 3 tasks  (long: 12 folds in-process)
#   prob_folds  36 jobs   prob_ensemble stage 1, one fold each
#   prob_final   3 jobs   prob_ensemble stage 2, afterok on the folds
set -euo pipefail
cd "${HIPEXO_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
source ml/LOTO/slurm/env.sh >/dev/null

WHICH="all"; PART="$HIPEXO_PART"; EXTRA=""
for arg in "$@"; do
  case "$arg" in
    --requeue) PART="gpu_requeue"; EXTRA="--requeue" ;;
    *) WHICH="$arg" ;;
  esac
done
COMMON=(--account "$HIPEXO_ACCOUNT" --partition "$PART" ${EXTRA:+$EXTRA}
        --export=ALL,HIPEXO_REPO="$HIPEXO_REPO",HIPEXO_LOTO_DATA="$HIPEXO_LOTO_DATA")
mkdir -p ml/LOTO/slurm/logs

sub() { sbatch "${COMMON[@]}" "$@" | awk '{print $NF}'; }

if [[ "$WHICH" == "all" || "$WHICH" == "single" ]]; then
  J1=$(sub ml/LOTO/slurm/single.sbatch)
  echo "single      array $J1   (6 cells: AE + GAN x SA ST TR)"
fi
if [[ "$WHICH" == "all" || "$WHICH" == "vanilla_ens" ]]; then
  J2=$(sub ml/LOTO/slurm/vanilla_ens.sbatch)
  echo "vanilla_ens array $J2   (9 cells, ~6-9 h each, all concurrent)"
fi
if [[ "$WHICH" == "all" || "$WHICH" == "prob" ]]; then
  J3=$(sub ml/LOTO/slurm/prob_folds.sbatch)
  echo "prob_folds  array $J3   (36 folds)"
  J4=$(sbatch "${COMMON[@]}" --dependency="afterok:$J3" ml/LOTO/slurm/prob_final.sbatch | awk '{print $NF}')
  echo "prob_final  array $J4   (3 cells, waits on $J3)"
fi

cat <<TXT

watch:     squeue -u \$USER
progress:  python ml/LOTO/run.py --list
check:     python ml/LOTO/verify.py
collect:   python ml/LOTO/summarize.py
pull back: bash ml/LOTO/slurm/pull.sh       # run this one on your LAPTOP
TXT
