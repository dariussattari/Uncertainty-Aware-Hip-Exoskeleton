#!/bin/bash
#SBATCH --job-name=nature-ensemble
#SBATCH --output=ml/nature/awareness/slurm/logs/ensemble_%A_%a.out
#SBATCH --error=ml/nature/awareness/slurm/logs/ensemble_%A_%a.out
#SBATCH --array=0-2                 # seeds 0-2 of the gait-phase ensemble (train_ensemble.py); submit.sh --ensemble
#SBATCH --time=01:00:00             # val protocol; the paper's LOSO protocol (--args "--protocol loso") needs about 14x: submit.sh --time 08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # one single-threaded process; the GPU is the bottleneck (CLAUDE.md rule 1)
#SBATCH --gres=gpu:1
#SBATCH --mem=3G                    # the autoencoders' arrays and loader (train_ae.sbatch) plus the 150 MB target array; re-check with jobstats on task 0
# The ankle paper's gait-phase ensemble (seven TCNs) under bf16 autocast, the baseline the awareness models are judged
# against. Before the full array: --array 0, then jobstats (CLAUDE.md rule 4).
set -euo pipefail
source "$HIPEXO_REPO/ml/nature/awareness/slurm/env.sh"
i="${SLURM_ARRAY_TASK_ID:?run as an array}"
python ml/nature/awareness/train_ensemble.py --seed "$i" --data-dir "$HIPEXO_NATURE_AE_DATA" \
  --out-root ml/nature/awareness/runs/ensemble ${AE_SMOKE:+--smoke} ${AE_ARGS:-}
echo "NATURE ENSEMBLE OK: seed $i ${AE_ARGS:-}"
