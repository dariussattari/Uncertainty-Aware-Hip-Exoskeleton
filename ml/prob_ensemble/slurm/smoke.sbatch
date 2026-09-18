#!/bin/bash
#SBATCH --job-name=hipexo-smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.out
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Verifies the whole chain cheaply before you spend real GPU hours: CUDA is visible, the data
# is readable, one fold trains, collect works on a single fold, stage 2 runs, eval runs.
set -euo pipefail
mkdir -p logs
source "$(dirname "$0")/env.sh"
stage_data_to_tmp

OUT=ml/prob_ensemble/runs/_smoke

python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("FAIL: torch cannot see a GPU inside the allocation. Check that the cuda module "
             "major version matches the torch wheel installed by setup.sh.")
print(f"OK: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
PY

python data.py
python train.py fold --index 0 --out "$OUT" --max-epochs 4 --warmup-epochs 1 --patience 2 --force
python train.py collect --out "$OUT" --allow-partial
python train.py final --out "$OUT" --epochs 2
python evaluate.py --out "$OUT" --split test

echo
echo "SMOKE PASSED. Numbers are meaningless (4 epochs, 1 fold) -- the wiring is what was tested."
echo "Next: bash slurm/submit.sh"
