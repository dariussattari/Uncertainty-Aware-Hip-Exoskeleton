#!/bin/bash
#SBATCH --job-name=loto-smoke
#SBATCH --output=ml/sci_rob/LOTO/slurm/logs/smoke_%j.out
#SBATCH --error=ml/sci_rob/LOTO/slurm/logs/smoke_%j.out
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Cheap end-to-end proof before committing 50+ GPU-hours: CUDA is visible, the arrays resolve,
# the task holdout produces the expected window counts, and one real cell trains and evaluates.
set -euo pipefail
source "$HIPEXO_REPO/ml/sci_rob/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN ensemble"
stage_data_to_tmp

python - <<'PY'
import torch, sys
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("FAIL: no CUDA inside the allocation. The torch wheel does not match the loaded "
             "cuda module -- rerun slurm/setup.sh with the matching --index-url.")
print(f"  {torch.cuda.get_device_name(0)}")
PY

echo; echo "=== splits resolve, all three holdouts ==="
python ml/sci_rob/LOTO/splits.py --family autoencoder

echo; echo "=== one real cell, end to end: the autoencoder, TR held out ==="
python ml/sci_rob/LOTO/run.py --held-out TR --model vanilla/AE --force
python ml/sci_rob/LOTO/verify.py

# Two-epoch wiring checks for the models whose real cells cost 6-9 h each. Cheap insurance:
# a registry or data-factory mistake shows up here in minutes instead of after a day of
# queued GPU time. Ensemble_GP and Ensemble_synthetic were smoked locally; forecast was not.
echo; echo "=== two-epoch wiring checks for the long models ==="
for m in vanilla/Ensemble_forecast vanilla/Ensemble_GP; do
  echo "--- $m ---"
  python ml/sci_rob/LOTO/run.py --held-out TR --model "$m" --stage smoke || {
    echo "SMOKE FAILED for $m -- do NOT submit vanilla_ens.sbatch"; exit 1; }
done

echo; echo "SMOKE OK -- safe to run submit.sh"
