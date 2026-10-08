#!/bin/bash
# Submit the awareness stages for one holdout. Run on a LOGIN node from the repo root.
#
#     bash ml/sci_rob/awareness/slurm/submit.sh DOWN               # encoder+regime, plain forecast and the
#                                                          # regime experts now, conditioned forecast after B
#     bash ml/sci_rob/awareness/slurm/submit.sh DOWN --only experts  # just the expert array
#     bash ml/sci_rob/awareness/slurm/submit.sh DOWN --folds 12    # full epoch-selection protocol
#     bash ml/sci_rob/awareness/slurm/submit.sh DOWN --only pelvis   # 22-channel ablation: build arrays (CPU),
#                                                          # smoke on gpu_test, then the full encoder+regime
set -euo pipefail
cd "${HIPEXO_REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
source ml/sci_rob/LOTO/slurm/env.sh >/dev/null
HOLDOUT="${1:?usage: submit.sh <holdout> [--folds N] [--requeue]}"; shift
PART="$HIPEXO_PART"; EXTRA=""; FOLDS=1
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --requeue) PART="gpu_requeue"; EXTRA="--requeue" ;;
    --folds) FOLDS="$2"; shift ;;
    --only) ONLY="$2"; shift ;;
  esac
  shift
done
mkdir -p ml/sci_rob/awareness/slurm/logs
COMMON=(--account "$HIPEXO_ACCOUNT" --partition "$PART" ${EXTRA:+$EXTRA}
        --export=ALL,HIPEXO_REPO="$HIPEXO_REPO",HIPEXO_LOTO_DATA="$HIPEXO_LOTO_DATA",AW_HOLDOUT="$HOLDOUT",AW_FOLDS="$FOLDS")
if [[ "$ONLY" == "pelvis" ]]; then
  PD="${HIPEXO_PELVIS_DATA:-$HOME/hipexo_data/processed_all_pelvis}"
  PCOMMON=(--account "$HIPEXO_ACCOUNT" ${EXTRA:+$EXTRA}
           --export=ALL,HIPEXO_REPO="$HIPEXO_REPO",HIPEXO_LOTO_DATA="$HIPEXO_LOTO_DATA",HIPEXO_PELVIS_DATA="$PD",AW_HOLDOUT="$HOLDOUT")
  JB=$(sbatch "${PCOMMON[@]}" --partition shared ml/sci_rob/awareness/slurm/pelvis_build.sbatch | awk '{print $NF}')
  echo "pelvis arrays       job $JB   (CPU, from the raw trials -> $PD)"
  JS=$(sbatch "${PCOMMON[@]}" --partition "$HIPEXO_PART_TEST" --time=01:00:00 --dependency="afterok:$JB" \
       --export=ALL,HIPEXO_REPO="$HIPEXO_REPO",HIPEXO_LOTO_DATA="$HIPEXO_LOTO_DATA",HIPEXO_PELVIS_DATA="$PD",AW_HOLDOUT="$HOLDOUT",AW_TAG=pelvis_smoke,AW_ENCODER_ARGS=--smoke \
       ml/sci_rob/awareness/slurm/pelvis.sbatch | awk '{print $NF}')
  echo "pelvis smoke        job $JS   (waits on $JB)"
  JF=$(sbatch "${PCOMMON[@]}" --partition "$PART" --dependency="afterok:$JS" ml/sci_rob/awareness/slurm/pelvis.sbatch | awk '{print $NF}')
  echo "pelvis encoder      job $JF   (waits on $JS)"
  echo "watch: squeue -u \$USER"; exit 0
fi
if [[ "$ONLY" == "horizons" ]]; then
  J5=$(sbatch "${COMMON[@]}" ml/sci_rob/awareness/slurm/horizons.sbatch | awk '{print $NF}')
  echo "horizon sweep       job $J5"; exit 0
fi
if [[ "$ONLY" == "experts" ]]; then
  J4=$(sbatch "${COMMON[@]}" ml/sci_rob/awareness/slurm/experts.sbatch | awk '{print $NF}')
  echo "regime experts      job $J4"; exit 0
fi
J1=$(sbatch "${COMMON[@]}" ml/sci_rob/awareness/slurm/encoder.sbatch | awk '{print $NF}')
echo "encoder+regime      job $J1"
J2=$(sbatch "${COMMON[@]}" --array=0 ml/sci_rob/awareness/slurm/forecast.sbatch | awk '{print $NF}')
echo "forecast plain      job $J2   (folds=$FOLDS)"
J3=$(sbatch "${COMMON[@]}" --array=1 --dependency="afterok:$J1" ml/sci_rob/awareness/slurm/forecast.sbatch | awk '{print $NF}')
echo "forecast regime     job $J3   (waits on $J1)"
J4=$(sbatch "${COMMON[@]}" ml/sci_rob/awareness/slurm/experts.sbatch | awk '{print $NF}')
echo "regime experts      job $J4   (array over the regimes present in training)"
echo "watch: squeue -u \$USER"
