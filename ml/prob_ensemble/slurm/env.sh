#!/bin/bash
# Cluster environment for the probabilistic ensemble. Edit the three EDIT ME lines, then
# never touch this again -- every sbatch script sources it.
#
# Written for Harvard FASRC (Cannon). Module names and partitions DO drift between
# maintenance windows, so verify them once with:
#
#     module avail python
#     module avail cuda
#     sacctmgr show assoc where user=$USER format=account,partition%30
#
# ---------------------------------------------------------------------------- EDIT ME
export HIPEXO_ACCOUNT="CHANGE_ME"            # your lab's SLURM account, e.g. someone_lab
export HIPEXO_REPO="$HOME/Uncertainty-Aware-Hip-Exoskeleton"
export HIPEXO_DATA="$HOME/hipexo_data"       # dir containing AE_GAN/ and ood/  (~4.3 GB)
# ------------------------------------------------------------------------- END EDIT ME

# Partitions. gpu_requeue is much cheaper and usually starts sooner, and every stage here is
# idempotent and checkpointed, so preemption costs at most one fold. gpu_test caps at 1 hour
# and is the right place for --smoke.
export HIPEXO_PART="${HIPEXO_PART:-gpu}"
export HIPEXO_PART_TEST="${HIPEXO_PART_TEST:-gpu_test}"

module purge
# python: verify with `module avail python` -- this name is a guess and the only unverified one
module load python/3.10.13-fasrc01 2>/dev/null || module load python
# cuda 12.4.1 and this cudnn were both confirmed present on Cannon (helmod-rocky8/Core).
# The cluster default is cuda/13.3.1, which has no matching stable torch wheel -- pin 12.4.1
# to match the cu124 wheel that setup.sh installs.
module load cuda/12.4.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01 2>/dev/null || module load cudnn

# The venv is built once by setup.sh and reused. Kept outside the repo so a git operation
# cannot disturb a running job.
export HIPEXO_VENV="${HIPEXO_VENV:-$HOME/.venvs/hipexo}"
if [[ -f "$HIPEXO_VENV/bin/activate" ]]; then
  source "$HIPEXO_VENV/bin/activate"
else
  echo "WARNING: no venv at $HIPEXO_VENV -- run slurm/setup.sh first" >&2
fi

export PYTHONUNBUFFERED=1
# The memmapped arrays are read by one process per task; extra BLAS threads only cause
# contention when several array tasks share a node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Optional: stage the 4.3 GB of arrays onto node-local disk. Worth it when the data sits on a
# home or networked filesystem, because training memory-maps it and reads it every epoch.
# Skip it on fast parallel scratch (/n/netscratch), where it just wastes a copy.
stage_data_to_tmp() {
  if [[ "${HIPEXO_STAGE:-0}" != "1" ]]; then return 0; fi
  local dest="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
  mkdir -p "$dest"
  echo "staging $HIPEXO_DATA -> $dest"
  local t0=$SECONDS
  cp -r "$HIPEXO_DATA"/AE_GAN "$HIPEXO_DATA"/ood "$dest"/ || { echo "staging failed" >&2; return 1; }
  export HIPEXO_DATA="$dest"
  echo "staged in $((SECONDS - t0))s; HIPEXO_DATA=$HIPEXO_DATA"
}

cd "$HIPEXO_REPO/ml/prob_ensemble" || { echo "no $HIPEXO_REPO/ml/prob_ensemble" >&2; exit 1; }

echo "--- environment ---"
echo "host      $(hostname)"
echo "job       ${SLURM_JOB_ID:-none}  array task ${SLURM_ARRAY_TASK_ID:-none}"
echo "python    $(command -v python)"
echo "data      $HIPEXO_DATA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "gpu       none visible"
echo "-------------------"
