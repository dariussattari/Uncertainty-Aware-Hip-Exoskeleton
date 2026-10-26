#!/bin/bash
#SBATCH --job-name=nature-moment
#SBATCH --output=ml/nature/estimator/slurm/logs/moment_%A_%a.out
#SBATCH --error=ml/nature/estimator/slurm/logs/moment_%A_%a.out
#SBATCH --array=0-2                 # seeds 0-2 of the paper's moment TCN (train_moment.py)
#SBATCH --time=03:00:00             # job 55133795 (A100, bf16): 7.1 ms per step, 0.8 min per training epoch; 100-epoch ceiling ~2.5 h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # one single-threaded process; the GPU is the bottleneck (CLAUDE.md rule 1)
#SBATCH --gres=gpu:1
#SBATCH --mem=6G                    # inputs.npy 3.8 GB + targets and estimates 0.6 GB read whole (~4.5 GB load peak); jobstats on
                                    # job 55134594: 3.1 GB sampled mid-run, CPU 48% of 1 core, GPU 60%
# The Nature paper's joint moment estimator (Model A). Order (CLAUDE.md rules 4 and 6):
#   sbatch ... --array=0 --export=ALL,MOMENT_ARGS=--profile  train_moment.sbatch   # step time and top kernels
#   sbatch ... --array=0 train_moment.sbatch, then jobstats; then --array=1-2
set -euo pipefail
source "$HIPEXO_REPO/ml/nature/awareness/slurm/env.sh"
export HIPEXO_NATURE_MOMENT_DATA="${HIPEXO_NATURE_MOMENT_DATA:-$HOME/hipexo_data/nature_moment}"
i="${SLURM_ARRAY_TASK_ID:?run as an array}"
python ml/nature/estimator/train_moment.py --seed "$i" --data-dir "$HIPEXO_NATURE_MOMENT_DATA" \
  --out-root ml/nature/estimator/runs ${MOMENT_ARGS:-}
echo "NATURE MOMENT OK: seed $i ${MOMENT_ARGS:-}"
