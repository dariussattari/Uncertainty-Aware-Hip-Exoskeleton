#!/bin/bash
#SBATCH --job-name=vae-smoke
#SBATCH --output=ml/VAE/slurm/logs/smoke_%j.out
#SBATCH --error=ml/VAE/slurm/logs/smoke_%j.out
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Cheap proof before the matrix: CUDA is visible, both data paths resolve (the rebuilt original
# protocol and one LOTO holdout), and two short cells train, score and write every artifact
# the notebook reads. Smoke runs land in *_smoke directories and are never shown as results.
set -euo pipefail
source "$HIPEXO_REPO/ml/LOTO/slurm/env.sh"
export HIPEXO_STAGE_FAMILY="AE_GAN"
stage_data_to_tmp
python - <<'PY'
import torch, sys
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("FAIL: no CUDA inside the allocation")
print(" ", torch.cuda.get_device_name(0))
PY
python ml/VAE/vae_train.py --variant standard --protocol original --smoke --force
python ml/VAE/vae_train.py --variant beta4 --protocol SA --smoke --force
python - <<'PY'
import json, numpy as np, pandas as pd
from pathlib import Path
root = Path('ml/VAE/runs')
for name in ('original_standard_seed0_smoke', 'SA_beta4_seed0_smoke'):
    out = root / name
    z = np.load(out / 'latents_test.npz'); meta = pd.read_parquet(out / 'latents_test.parquet')
    ev = json.loads((out / 'eval_test.json').read_text())
    assert set(z.files) >= {'Z', 'logvar', 'labels', 'smooth_latent_lof'}, z.files
    assert len(meta) == len(z['Z']) == len(z['logvar']), 'latent/meta length mismatch'
    assert (out / 'train_reference_mu.npy').exists() and (out / 'final.pt').exists()
    print(f"{name}: {len(meta):,} test windows, OOD {int(z['labels'].sum()):,}, "
          f"modes {sorted(meta['mode'].unique())}, LOF AUROC {ev['scores']['latent_lof']['auroc']:.3f} (smoke, not a result)")
PY
echo; echo "SMOKE OK -- safe to run submit.sh"
