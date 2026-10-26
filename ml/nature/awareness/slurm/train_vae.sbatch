#!/bin/bash
#SBATCH --job-name=nature-vae
#SBATCH --output=ml/nature/awareness/slurm/logs/vae_%A_%a.out
#SBATCH --error=ml/nature/awareness/slurm/logs/vae_%A_%a.out
#SBATCH --array=0-29                # train_vae.py --list: 0-14 fixed decoder variance, 15-29 learned; beta 1, 8, 32, 64, 128 x seeds 0-2
                                    # with --inputs raw_3s_spectral (submit.sh): 0-14, train_vae.py --inputs raw_3s_spectral --list
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # one single-threaded process, as train_ae.sbatch (CLAUDE.md rule 1)
#SBATCH --gres=gpu:1
#SBATCH --mem=3G                    # train_ae.sbatch's measured request (same arrays and loader); re-check with jobstats on task 0
# One VAE per task under bf16 autocast, the AE's settings (train_vae.py). Before the full array: --array 0, then jobstats.
set -euo pipefail
source "$HIPEXO_REPO/ml/nature/awareness/slurm/env.sh"
i="${SLURM_ARRAY_TASK_ID:?run as an array}"
python ml/nature/awareness/train_vae.py --index "$i" --inputs "${AE_INPUTS:-raw_1s}" --data-dir "$HIPEXO_NATURE_AE_DATA" --out-root ml/nature/awareness/runs/vae ${AE_SMOKE:+--smoke} ${AE_ARGS:-}
echo "NATURE VAE OK: task $i inputs ${AE_INPUTS:-raw_1s} ${AE_ARGS:-}"
