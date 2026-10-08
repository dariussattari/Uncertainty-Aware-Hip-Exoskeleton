#!/bin/bash
# One-time environment build on the cluster. Run on a LOGIN node -- it only pip-installs.
#
#     bash ml/sci_rob/LOTO/slurm/setup.sh
#
# Shares the venv with ml/sci_rob/prob_ensemble; if that one already exists this is a no-op beyond
# confirming the imports.
set -euo pipefail
source "$(dirname "$0")/env.sh" || true

if [[ ! -f "$HIPEXO_VENV/bin/activate" ]]; then
  python -m venv "$HIPEXO_VENV"
  source "$HIPEXO_VENV/bin/activate"
  python -m pip install --upgrade pip wheel
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -r "$HIPEXO_REPO/ml/sci_rob/prob_ensemble/requirements.txt"
else
  source "$HIPEXO_VENV/bin/activate"
  echo "reusing the existing venv at $HIPEXO_VENV"
  # still apply requirements: a venv built for an earlier stage can predate a dependency,
  # and a missing one only surfaces hours later when the figures stage runs
  python -m pip install -q -r "$HIPEXO_REPO/ml/sci_rob/prob_ensemble/requirements.txt"
fi

python - <<'PY'
import torch, numpy, pandas, sklearn, matplotlib, pyarrow
print(f"torch {torch.__version__}  cuda available {torch.cuda.is_available()}")
print("numpy, pandas, sklearn, matplotlib, pyarrow all import")
print("  (no GPU from a login node is normal -- smoke.sbatch checks it inside an allocation)")
PY
echo; echo "venv ready at $HIPEXO_VENV"
echo "next: sbatch ml/sci_rob/LOTO/slurm/smoke.sbatch"
