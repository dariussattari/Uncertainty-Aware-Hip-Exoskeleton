#!/bin/bash
# Build the mode-complete arrays ON THE CLUSTER, from Zenodo, instead of uploading them.
#
#     bash ml/LOTO/slurm/fetch_data.sh
#
# Measured: Zenodo -> cluster runs at ~24 MB/s, against ~0.3 MB/s for laptop -> cluster. The
# 9.2 GB of processed arrays would take over eight hours to upload; the 20 raw archives they
# are derived from take about three minutes to pull here, and rebuilding them is four more.
# So the cheap direction is to move the recipe, not the result.
#
# Only participants AB15-AB34 are fetched -- those are the twenty in the study pool (phase 3/4,
# custom hip hardware). The other fourteen are Samsung GEMS or early-phase and are filtered out
# by the inventory before any window is cut, so downloading them would be pure waste.
#
# The participant split is fixed by a seed over the INVENTORY, not over what is on disk, so a
# partial download cannot silently change which participants train and which are held out.
set -euo pipefail

REPO="${HIPEXO_REPO:-$HOME/Uncertainty-Aware-Hip-Exoskeleton}"
RAW="$REPO/data/molinaro-raw"
REC=10849318
PAR="${PAR:-5}"

cd "$REPO"
mkdir -p "$RAW"

fetch_one() {
  local i="$1" raw="$2" rec="$3"
  if [[ -d "$raw/AB$i" ]]; then echo "skip AB$i (already unpacked)"; return 0; fi
  local zip="$raw/AB$i.zip"
  # --retry-all-errors is not in the cluster's curl (7.x); --retry alone covers transient
  # network and 5xx failures, which is what matters here.
  curl -L --fail -C - --retry 10 --retry-delay 5 \
       --connect-timeout 20 --speed-limit 50000 --speed-time 60 \
       -s -o "$zip" "https://zenodo.org/records/${rec}/files/AB${i}.zip?download=1" \
    || { echo "FAIL download AB$i"; return 1; }
  unzip -qq -o "$zip" -d "$raw" || { echo "FAIL unzip AB$i"; return 1; }
  rm -f "$zip"                       # the archive is dead weight once unpacked
  echo "ok   AB$i"
}
export -f fetch_one

echo "=== fetching AB15-AB34 from Zenodo (${PAR} at a time) ==="
t0=$SECONDS
seq -w 15 34 | xargs -P "$PAR" -I{} bash -c 'fetch_one "$@"' _ {} "$RAW" "$REC"
echo "--- $((SECONDS - t0))s, $(du -sh "$RAW" | cut -f1) on disk ---"

n=$(ls -d "$RAW"/AB* 2>/dev/null | wc -l)
if [[ "$n" -ne 20 ]]; then
  echo "expected 20 participant directories, found $n -- rerun to fill the gaps" >&2
  exit 1
fi

echo
echo "=== rebuilding the mode-complete arrays ==="
# Any partial upload must go first: build_data.py skips a split whose .npy already exists, so a
# truncated file left behind would be silently accepted and then trained on.
# `find ... | head -1 | grep -q .` looks like a presence test but is not one here: head
# closes the pipe, find dies on SIGPIPE, and `set -o pipefail` then makes the whole pipeline
# non-zero, so the clearing silently never ran. Delete unconditionally instead -- the arrays
# are derived data, rebuilt in four minutes, and there is no version of this worth guessing at.
echo "clearing any previous arrays (they are rebuilt below, not recovered)"
rm -rf "$REPO/data/processed_all"/AE_GAN "$REPO/data/processed_all"/ensemble
python ml/LOTO/build_data.py

echo
echo "=== window counts, to compare against the laptop ==="
python - <<'PY'
import pandas as pd
from pathlib import Path
root = Path("data/processed_all")
for fam in ("AE_GAN", "ensemble"):
    for split in ("train", "val", "test"):
        p = root / fam / f"{split}_meta.parquet"
        if not p.exists():
            print(f"{fam}/{split}: MISSING"); continue
        m = pd.read_parquet(p, columns=["mode"])
        by = m["mode"].value_counts().to_dict()
        print(f"{fam:7s} {split:5s} {len(m):7,}  " +
              "  ".join(f"{k}={by.get(k, 0):6,}" for k in
                        ("LG", "RA", "RD", "SA", "SD", "TR", "ST")))
PY
echo
echo "DATA READY -- next: sbatch ml/LOTO/slurm/smoke.sbatch"
