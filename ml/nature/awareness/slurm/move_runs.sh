#!/bin/bash
# Move run folders out of the home quota to the lab's holylabs storage, leaving a symlink, so every script keeps working:
#     sbatch ... move_runs.sh runs runs_train_class/id_ood_safe      (paths relative to ml/nature/awareness)
# Each folder is copied, the copy checked file by file (size and time) against the source, and only then removed.
set -euo pipefail
SRC_ROOT="$HOME/Uncertainty-Aware-Hip-Exoskeleton/ml/nature/awareness"
DST_ROOT="${HIPEXO_RUNS_STORE:-/n/holylabs/LABS/pslade_lab/Lab/$USER/hipexo_runs}"
for rel in "$@"; do
  src="$SRC_ROOT/$rel"; dst="$DST_ROOT/$rel"
  if [[ -L "$src" ]]; then echo "$rel: already a symlink -> $(readlink "$src")"; continue; fi
  mkdir -p "$dst"
  rsync -a "$src/" "$dst/"
  diff_lines=$(rsync -a --dry-run --itemize-changes "$src/" "$dst/" | wc -l)
  n_src=$(find "$src" -type f | wc -l); n_dst=$(find "$dst" -type f | wc -l)
  if [[ "$diff_lines" -ne 0 || "$n_src" -ne "$n_dst" ]]; then echo "$rel: copy differs ($diff_lines changes, $n_src vs $n_dst files); source kept" >&2; exit 1; fi
  rm -rf "$src"
  ln -s "$dst" "$src"
  echo "$rel: $n_dst files moved to $dst ($(du -sh "$dst" | cut -f1))"
done
df -h "$HOME" | tail -1
