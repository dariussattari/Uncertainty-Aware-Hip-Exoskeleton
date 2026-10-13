#!/bin/bash
#SBATCH --job-name=nature-ae
#SBATCH --output=ml/nature/awareness/slurm/logs/ae_%A_%a.out
#SBATCH --error=ml/nature/awareness/slurm/logs/ae_%A_%a.out
#SBATCH --array=0-5                 # 0-2 vanilla seeds 0-2, 3-5 sparse seeds 0-2, 6-8 gaussian seeds 0-2 (submit.sh --array)
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # one single-threaded process; the GPU is the bottleneck (measured below)
#SBATCH --gres=gpu:1
#SBATCH --mem=3G                    # load-time peak ~2.2 GB (signals 1.2 GB + stats sample + CUDA host) + 25%; sampled MaxRSS 1.0 GB
# One autoencoder per task, trained under bf16 autocast (train_ae.get_autocast: 23 ms per step on an A100 against 551 ms
# in fp32, where cuDNN picks FFT convolutions). Each process keeps its GPU busy, so packing six onto one GPU only queued
# them (52 s per epoch, job 51341641).
set -euo pipefail
source "$HIPEXO_REPO/ml/nature/awareness/slurm/env.sh"
VARIANTS=(vanilla vanilla vanilla sparse sparse sparse gaussian gaussian gaussian)
SEEDS=(0 1 2 0 1 2 0 1 2)
i="${SLURM_ARRAY_TASK_ID:?run as an array}"
python ml/nature/awareness/train_ae.py --variant "${VARIANTS[$i]}" --seed "${SEEDS[$i]}" \
  --inputs "${AE_INPUTS:-raw_1s}" --data-dir "$HIPEXO_NATURE_AE_DATA" --out-root ml/nature/awareness/runs/ae ${AE_SMOKE:+--smoke} ${AE_ARGS:-}
echo "NATURE AE OK: ${VARIANTS[$i]} seed ${SEEDS[$i]} inputs ${AE_INPUTS:-raw_1s}"
