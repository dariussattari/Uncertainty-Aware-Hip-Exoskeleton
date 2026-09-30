#!/bin/bash
#SBATCH --job-name=vae-sweep
#SBATCH --output=ml/VAE/slurm/logs/sweep_%A_%a.out
#SBATCH --error=ml/VAE/slurm/logs/sweep_%A_%a.out
#SBATCH --array=0-6
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# The beta sweep and controls on the original protocol, one cell per array task
# (see GROUPS['sweep'] in ml/VAE/vae_train.py):
#
#   0 1 2 3 4 -> beta 0.25, 2, 8, 16, 32        (beta 1 and 4 come from the matrix)
#   5         -> ae_control       beta=0, no sampling: the AE objective at the VAE settings
#   6         -> ae_control_aehp  beta=0, no sampling, at the AE's lr / batch / epoch budget
#
# Summarize afterwards with: python ml/VAE/vae_sweep.py --protocol original --write
set -euo pipefail
source "$HIPEXO_REPO/ml/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
python ml/VAE/vae_train.py --group sweep --index "$SLURM_ARRAY_TASK_ID" ${VAE_ARGS:-}
