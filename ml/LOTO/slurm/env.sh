#!/bin/bash
# Cluster environment for the leave-one-task-out matrix. Every sbatch script sources it.
#
# Written for Harvard FASRC (Cannon). Module names and partitions drift between maintenance
# windows; verify once with `module avail python`, `module avail cuda`, and
# `sacctmgr show assoc where user=$USER format=account,partition%30`.
#
# ---------------------------------------------------------------------------- EDIT ME
export HIPEXO_ACCOUNT="${HIPEXO_ACCOUNT:-pslade_lab}"   # verified from sacctmgr
export HIPEXO_REPO="${HIPEXO_REPO:-$HOME/Uncertainty-Aware-Hip-Exoskeleton}"
export HIPEXO_LOTO_DATA="${HIPEXO_LOTO_DATA:-$HOME/hipexo_data/processed_all}"
# ------------------------------------------------------------------------- END EDIT ME

export HIPEXO_PART="${HIPEXO_PART:-gpu}"
export HIPEXO_PART_TEST="${HIPEXO_PART_TEST:-gpu_test}"

module purge
module load python/3.10.13-fasrc01 2>/dev/null || module load python
module load cuda/12.4.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01 2>/dev/null || module load cudnn

# Shared with ml/prob_ensemble -- the dependencies are identical, so one venv serves both.
export HIPEXO_VENV="${HIPEXO_VENV:-$HOME/.venvs/hipexo}"
if [[ -f "$HIPEXO_VENV/bin/activate" ]]; then
  source "$HIPEXO_VENV/bin/activate"
else
  echo "WARNING: no venv at $HIPEXO_VENV -- run slurm/setup.sh first" >&2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$USER"

# run.py locates the repo by walking up for a directory named `data`. push.sh creates
# $HIPEXO_REPO/data as a symlink to the arrays, so that works here unchanged; splits.py then
# reads $HIPEXO_LOTO_DATA, which may point somewhere else entirely (scratch, node-local).
cd "$HIPEXO_REPO" || { echo "no $HIPEXO_REPO" >&2; exit 1; }

echo "--- environment ---"
echo "host      $(hostname)"
echo "job       ${SLURM_JOB_ID:-none}  array task ${SLURM_ARRAY_TASK_ID:-none}"
echo "python    $(command -v python)"
echo "repo      $HIPEXO_REPO"
echo "data      $HIPEXO_LOTO_DATA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "gpu       none visible"
echo "-------------------"

# ---------------------------------------------------------------------------------------
# Stage the arrays onto node-local disk. NOT optional, and measured: training memory-maps the
# windows and gathers ~1,000 random rows per batch, which on $HOME (NFS) ran an autoencoder
# epoch in 221.8 s against 8 s on a laptop SSD -- 28x slower on an A100, with the GPU idle
# the whole time. The cost is one sequential 9.9 GB copy per job; the saving is every epoch
# after it.
stage_data_to_tmp() {
  [[ "${HIPEXO_STAGE:-1}" == "1" ]] || { echo "staging disabled"; return 0; }
  local dest="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/hipexo.$$"
  local src="$HIPEXO_LOTO_DATA"
  mkdir -p "$dest" || { echo "cannot create $dest -- training from $src" >&2; return 0; }
  local need avail
  need=$(du -sk "$src" | cut -f1)
  avail=$(df -k --output=avail "$dest" 2>/dev/null | tail -1 || echo 0)
  if [[ "${avail:-0}" -lt "$((need + 1048576))" ]]; then
    echo "only ${avail}K free at $dest for ${need}K of arrays -- training from $src instead" >&2
    return 0
  fi
  # Copy only the family this job reads. AE, GAN and every forecast/correlation model read
  # AE_GAN/; only Ensemble_GP reads ensemble/. Staging both doubles the copy and, across a
  # 36-task array, pulls several hundred GB through NFS home for no reason.
  local fams="${HIPEXO_STAGE_FAMILY:-AE_GAN ensemble}"
  echo "staging [$fams] $src -> $dest"
  local t0=$SECONDS
  for f in $fams; do
    [[ -d "$src/$f" ]] || continue
    cp -r "$src/$f" "$dest"/ || { echo "staging failed -- training from $src" >&2; return 0; }
  done
  export HIPEXO_LOTO_DATA="$dest"
  echo "staged $(du -sh "$dest" | cut -f1) in $((SECONDS - t0))s; HIPEXO_LOTO_DATA=$HIPEXO_LOTO_DATA"
}

# The matrix, flattened so an array index maps to one (model, held-out task) cell.
# Order matters only for readability; the arrays below index into these.
export LOTO_TASKS=(SA ST TR)
export LOTO_SINGLE=(vanilla/AE vanilla/GAN)                       # single-stage: 6 cells
export LOTO_VANILLA_ENS=(vanilla/Ensemble_GP vanilla/Ensemble_synthetic vanilla/Ensemble_forecast)

# cell <flat-index> <space-separated model list> -> sets $CELL_MODEL and $CELL_TASK.
# Takes the list by value rather than by nameref (`local -n`), which needs bash >= 4.3 and is
# not worth depending on for four lines.
cell() {
  local idx=$1; shift
  local models=( "$@" )
  local n=${#LOTO_TASKS[@]}
  CELL_MODEL="${models[$(( idx / n ))]}"
  CELL_TASK="${LOTO_TASKS[$(( idx % n ))]}"
}
