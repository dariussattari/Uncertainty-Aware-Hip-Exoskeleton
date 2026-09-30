#!/bin/bash
#SBATCH --job-name=aw-regime
#SBATCH --output=ml/awareness/slurm/logs/regime_%j.out
#SBATCH --error=ml/awareness/slurm/logs/regime_%j.out
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Stage B alone (the probes on an already-encoded run); no data staging needed.
set -euo pipefail
source "$HIPEXO_REPO/ml/LOTO/slurm/env.sh"
: "${AW_HOLDOUT:?set AW_HOLDOUT}"
python ml/awareness/regime.py --holdout "$AW_HOLDOUT"
echo "REGIME OK for $AW_HOLDOUT"
