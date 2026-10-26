#!/bin/bash
# Submit the Nature autoencoder array, with --vae the VAE grid (train_vae.py), with --ensemble the gait-phase ensemble
# baseline (train_ensemble.py). Run on a LOGIN node.
#
#     bash ml/nature/awareness/slurm/submit.sh --smoke          # two-epoch wiring check of task 0 on gpu_test
#     bash ml/nature/awareness/slurm/submit.sh --array 0        # one task first: check `jobstats` (CLAUDE.md)
#     bash ml/nature/awareness/slurm/submit.sh --array 1-5      # then the rest
#     bash ml/nature/awareness/slurm/submit.sh --inputs raw_3s_spectral --array 0   # another input setting (train_ae.INPUTS)
#     bash ml/nature/awareness/slurm/submit.sh --inputs raw_3s_spectral --array 3-5 --args "--sparsity-weight 0.1"
#     bash ml/nature/awareness/slurm/submit.sh --array 6-8                        # the gaussian variant, seeds 0-2
#     bash ml/nature/awareness/slurm/submit.sh --vae --smoke                      # VAE: two-epoch wiring check of cell 0 on gpu_test
#     bash ml/nature/awareness/slurm/submit.sh --vae --array 0                    # VAE: the standard VAE, seed 0; then jobstats
#     bash ml/nature/awareness/slurm/submit.sh --vae --array 1-29                 # VAE: the rest of the grid (train_vae.py --list)
#     bash ml/nature/awareness/slurm/submit.sh --vae --inputs raw_3s_spectral --array 0-14   # VAE on 3 s + spectral
#     bash ml/nature/awareness/slurm/submit.sh --vae --inputs raw_3s --args "--arch lstm" --array 0-11   # LSTM-VAE (or transformer)
#     bash ml/nature/awareness/slurm/submit.sh --train-class --array 33,84         # training-class ablation: first tasks, then jobstats
#     bash ml/nature/awareness/slurm/submit.sh --train-class                       # every line of train_class_tasks.txt
#     bash ml/nature/awareness/slurm/submit.sh --train-class --tasks tcn_tasks.txt # another task list (train_class_tasks.py --arch tcn)
#     bash ml/nature/awareness/slurm/submit.sh --ensemble --smoke                 # gait-phase ensemble baseline: wiring check on gpu_test
#     bash ml/nature/awareness/slurm/submit.sh --ensemble --array 0               # seed 0 first, then jobstats
#     bash ml/nature/awareness/slurm/submit.sh --ensemble --array 1-2             # the other seeds
#     bash ml/nature/awareness/slurm/submit.sh --ensemble --args "--protocol loso" --time 08:00:00   # the paper's two-stage protocol
set -euo pipefail
cd "${HIPEXO_REPO:-$HOME/Uncertainty-Aware-Hip-Exoskeleton}"
export HIPEXO_ACCOUNT="${HIPEXO_ACCOUNT:-pslade_lab}" HIPEXO_REPO="${HIPEXO_REPO:-$PWD}"
mkdir -p ml/nature/awareness/slurm/logs
ARRAY=""; DEFAULT_ARRAY="0-5"; TASK_FILE=""; SCRIPT=train_ae.sbatch; PART=gpu; EXTRA=(); EXPORT="ALL,HIPEXO_REPO=$HIPEXO_REPO"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vae) SCRIPT=train_vae.sbatch; DEFAULT_ARRAY="0-29" ;;
    --train-class) SCRIPT=train_class.sbatch ;;
    --ensemble) SCRIPT=train_ensemble.sbatch; DEFAULT_ARRAY="0-2" ;;
    --tasks) TASK_FILE="$2"; EXPORT="$EXPORT,TASK_FILE=$2"; shift ;;
    --time) EXTRA+=(--time="$2"); shift ;;               # override the script's time limit
    --smoke) PART=gpu_test; ARRAY=0; EXTRA=(--time=00:20:00); EXPORT="$EXPORT,AE_SMOKE=1" ;;
    --array) ARRAY="$2"; shift ;;
    --inputs) EXPORT="$EXPORT,AE_INPUTS=$2"; shift ;;
    --args) export AE_ARGS="$2"; shift ;;               # extra train_ae.py options, e.g. --sparsity-weight 0.1 --latent 8x4
  esac
  shift
done
if [[ "$SCRIPT" == train_class.sbatch ]]; then
  DEFAULT_ARRAY="0-$(( $(wc -l < "ml/nature/awareness/slurm/${TASK_FILE:-train_class_tasks.txt}") - 1 ))"
fi
sbatch --account "$HIPEXO_ACCOUNT" --partition "$PART" --array="${ARRAY:-$DEFAULT_ARRAY}" ${EXTRA[@]+"${EXTRA[@]}"} --export="$EXPORT" \
  "ml/nature/awareness/slurm/$SCRIPT"
