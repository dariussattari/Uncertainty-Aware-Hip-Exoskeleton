#!/bin/bash
# Bring the cluster's results home and rebuild every downstream artefact. Run on the LAPTOP.
#
#     bash ml/sci_rob/LOTO/slurm/collect.sh            # pull, verify, summarise, write the doc section
#     bash ml/sci_rob/LOTO/slurm/collect.sh --no-doc   # stop before touching Live_Update.docx
#
# Safe to run repeatedly while the cluster is still working: it copies only what has appeared
# since last time, and every step downstream is idempotent. Running it early gives a partial
# table rather than an error.
#
# NOTE: macOS ships rsync 2.6.9, which has no --info flag. Use --stats/-v only.
set -uo pipefail
HOST="${HIPEXO_HOST:-login.rc.fas.harvard.edu}"
REMOTE="${HIPEXO_REMOTE:-Uncertainty-Aware-Hip-Exoskeleton}"
cd "$(dirname "$0")/../../../.."
PY=.venv/bin/python

echo "==> pulling results from $HOST"
# Results only. Checkpoints are large and nothing downstream reads them; add
# --include '*final.pt' if the weights are wanted too.
rsync -az --stats \
  --include '*/' \
  --include 'eval_*.json' --include 'report.json' --include 'summary.md' \
  --include '*.png' --include '*.pdf' --include '*.csv' \
  --include 'loso_summary.json' --include 'loso/***' \
  --include 'scores_*.npz' --include 'latents_*.npz' --include '*.parquet' \
  --exclude '*' \
  "$HOST:~/$REMOTE/ml/sci_rob/LOTO/" ml/sci_rob/LOTO/ 2>&1 | grep -E "Number of files transferred|Total transferred" || true

echo
echo "==> verifying every leaf against its own definition"
$PY ml/sci_rob/LOTO/verify.py || echo "  (verification reported problems -- read them before using the numbers)"

echo
echo "==> collecting"
$PY ml/sci_rob/LOTO/summarize.py --quiet
$PY ml/sci_rob/LOTO/run.py --list | tail -n 12

if [[ "${1:-}" != "--no-doc" ]]; then
  echo
  echo "==> writing the section into Live_Update.docx"
  $PY ml/sci_rob/LOTO/write_update.py
fi

echo
echo "done. table: ml/sci_rob/LOTO/summary_test.md   figures: ml/sci_rob/LOTO/{auroc,flagrate}_test.png"
