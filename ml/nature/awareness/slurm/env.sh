#!/bin/bash
# Cluster environment for the Nature awareness jobs (Harvard FASRC Cannon). Every sbatch script sources it.
export HIPEXO_ACCOUNT="${HIPEXO_ACCOUNT:-pslade_lab}"
export HIPEXO_REPO="${HIPEXO_REPO:-$HOME/Uncertainty-Aware-Hip-Exoskeleton}"
export HIPEXO_NATURE_AE_DATA="${HIPEXO_NATURE_AE_DATA:-$HOME/hipexo_data/nature_ae}"
export HIPEXO_VENV="${HIPEXO_VENV:-$HOME/.venvs/hipexo}"

module purge
module load python/3.10.13-fasrc01 2>/dev/null || module load python
module load cuda/12.4.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01 2>/dev/null || module load cudnn
source "$HIPEXO_VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1                  # one thread per training process (CLAUDE.md, cluster rules)
export MKL_NUM_THREADS=1
cd "$HIPEXO_REPO" || { echo "no $HIPEXO_REPO" >&2; exit 1; }
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | cpus ${SLURM_CPUS_PER_TASK:-?} | $(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"
