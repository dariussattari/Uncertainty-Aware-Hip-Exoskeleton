#!/bin/bash
# One-time environment build on the cluster. Run on a LOGIN node -- it only pip-installs.
#
#     bash slurm/setup.sh
#
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh || true          # env.sh warns about the missing venv; that is expected here

python -m venv "$HIPEXO_VENV"
source "$HIPEXO_VENV/bin/activate"
python -m pip install --upgrade pip wheel

# CUDA 12.4 wheels. If `module load cuda` gave you a different major version, change the index
# URL to match (cu121, cu124, cu126 ...); a mismatch shows up as torch.cuda.is_available()
# returning False, which smoke.sbatch checks for explicitly.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r ../requirements.txt

python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device 0: {torch.cuda.get_device_name(0)}")
else:
    print("  NOTE: no GPU visible from a login node -- that is normal. smoke.sbatch verifies")
    print("        CUDA from inside an allocation, which is where it matters.")
PY
echo
echo "venv ready at $HIPEXO_VENV"
echo "next: sbatch slurm/smoke.sbatch"
