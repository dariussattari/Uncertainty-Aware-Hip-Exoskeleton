#!/bin/bash
# Submit the Nature autoencoder array. Run on a LOGIN node.
#
#     bash ml/nature/awareness/slurm/submit.sh --smoke          # two-epoch wiring check of task 0 on gpu_test
#     bash ml/nature/awareness/slurm/submit.sh --array 0        # one task first: check `jobstats` (CLAUDE.md)
#     bash ml/nature/awareness/slurm/submit.sh --array 1-5      # then the rest
#     bash ml/nature/awareness/slurm/submit.sh --inputs raw_3s_spectral --array 0   # another input setting (train_ae.INPUTS)
#     bash ml/nature/awareness/slurm/submit.sh --inputs raw_3s_spectral --array 3-5 --args "--sparsity-weight 0.1"
#     bash ml/nature/awareness/slurm/submit.sh --array 6-8                        # the gaussian variant, seeds 0-2
set -euo pipefail
cd "${HIPEXO_REPO:-$HOME/Uncertainty-Aware-Hip-Exoskeleton}"
export HIPEXO_ACCOUNT="${HIPEXO_ACCOUNT:-pslade_lab}" HIPEXO_REPO="${HIPEXO_REPO:-$PWD}"
mkdir -p ml/nature/awareness/slurm/logs
ARRAY="0-5"; PART=gpu; EXTRA=(); EXPORT="ALL,HIPEXO_REPO=$HIPEXO_REPO"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) PART=gpu_test; ARRAY=0; EXTRA=(--time=00:20:00); EXPORT="$EXPORT,AE_SMOKE=1" ;;
    --array) ARRAY="$2"; shift ;;
    --inputs) EXPORT="$EXPORT,AE_INPUTS=$2"; shift ;;
    --args) export AE_ARGS="$2"; shift ;;               # extra train_ae.py options, e.g. --sparsity-weight 0.1 --latent 8x4
  esac
  shift
done
sbatch --account "$HIPEXO_ACCOUNT" --partition "$PART" --array="$ARRAY" "${EXTRA[@]}" --export="$EXPORT" \
  ml/nature/awareness/slurm/train_ae.sbatch
