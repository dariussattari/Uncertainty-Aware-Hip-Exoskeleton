#!/bin/bash
# Bring finished runs back. Run on your LAPTOP, from the repo root. Safe to run repeatedly
# while the cluster is still working -- it only copies what has appeared since last time.
set -euo pipefail
# NOTE: macOS ships rsync 2.6.9, which has no --info flag. Use --stats/-v only.
HOST="${HIPEXO_HOST:-dsattari@login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../../.."

# Results only: checkpoints are big and are not needed to write up a result. Add
# --include '*final.pt' if you want the weights too.
rsync -az --stats \
  --include '*/' \
  --include 'eval_*.json' --include 'report.json' --include 'summary.md' \
  --include '*.png' --include '*.pdf' --include '*.csv' \
  --include 'loso_summary.json' --include 'loso/***' \
  --exclude '*' \
  "$HOST:~/$REMOTE/ml/sci_rob/LOTO/" ml/sci_rob/LOTO/

echo
python ml/sci_rob/LOTO/verify.py
python ml/sci_rob/LOTO/summarize.py --quiet
echo "next: python ml/sci_rob/LOTO/write_update.py"
