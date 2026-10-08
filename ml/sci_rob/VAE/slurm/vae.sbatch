#!/bin/bash
#SBATCH --job-name=vae-cell
#SBATCH --output=ml/sci_rob/VAE/slurm/logs/vae_%A_%a.out
#SBATCH --error=ml/sci_rob/VAE/slurm/logs/vae_%A_%a.out
#SBATCH --array=0-7
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# The VAE matrix for 08_vae_variants.ipynb: two variants x four protocols, one cell per task.
#
#   0 1 2 3 -> standard (beta=1)  original SA ST TR
#   4 5 6 7 -> beta4    (beta=4)  original SA ST TR
#
# "original" is the participant-split protocol (LG/RA/RD in-distribution), rebuilt from the
# mode-complete arrays because the cluster holds no data/processed arrays; SA/ST/TR use the
# same ml/sci_rob/LOTO/splits.py builder as every other LOTO cell. Each cell trains, fits LOF on
# training means, evaluates val/test and writes ml/sci_rob/VAE/runs/<protocol>_<variant>_seed0/.
set -euo pipefail
source "$HIPEXO_REPO/ml/sci_rob/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
python ml/sci_rob/VAE/vae_train.py --index "$SLURM_ARRAY_TASK_ID" ${VAE_ARGS:-}
