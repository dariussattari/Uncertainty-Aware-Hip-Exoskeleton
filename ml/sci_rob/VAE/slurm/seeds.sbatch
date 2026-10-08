#!/bin/bash
#SBATCH --job-name=vae-seeds
#SBATCH --output=ml/sci_rob/VAE/slurm/logs/seeds_%A_%a.out
#SBATCH --error=ml/sci_rob/VAE/slurm/logs/seeds_%A_%a.out
#SBATCH --array=1-2
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Repeat one (variant, protocol) cell with seeds 1 and 2 (seed 0 already exists), so that a
# difference of a few hundredths AUROC can be told from seed noise before any claim is made.
#
#   VAE_VARIANT=beta4 VAE_PROTOCOL=original sbatch ... ml/sci_rob/VAE/slurm/seeds.sbatch
#   (submit.sh seeds <variant> [protocol] sets both)
set -euo pipefail
source "$HIPEXO_REPO/ml/sci_rob/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
: "${VAE_VARIANT:?set VAE_VARIANT (a name from VARIANTS in ml/sci_rob/VAE/vae_train.py)}"
python ml/sci_rob/VAE/vae_train.py --variant "$VAE_VARIANT" --protocol "${VAE_PROTOCOL:-original}" \
    --seed "$SLURM_ARRAY_TASK_ID" ${VAE_ARGS:-}
