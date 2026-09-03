#!/usr/bin/env bash
# Download the Molinaro et al. (Sci. Robot. 2024, adi8852) hip exo dataset from Zenodo.
# Resumable, retrying, stall-aware, parallel. Safe to re-run: completed files are skipped.
set -uo pipefail

DEST="${1:-data/molinaro-raw}"
REC=10849318
PAR=4          # parallel transfers; raise cautiously, Zenodo throttles aggressive clients
mkdir -p "$DEST"

fetch() {
  local i="$1" dest="$2"
  curl -L --fail \
       -C -                              `# resume partial files` \
       --retry 15 --retry-delay 5 --retry-all-errors \
       --connect-timeout 20 \
       --speed-limit 50000 --speed-time 30 `# abort if <50KB/s for 30s, then retry` \
       -o "${dest}/AB${i}.zip" \
       "https://zenodo.org/records/${REC}/files/AB${i}.zip?download=1" \
    && echo "ok   AB${i}" || echo "FAIL AB${i}"
}
export -f fetch
export REC

seq -w 1 34 | xargs -P "$PAR" -I{} bash -c 'fetch "$@"' _ {} "$DEST"

echo
echo "=== verifying zip integrity ==="
fail=0
for f in "$DEST"/AB*.zip; do
  if unzip -tqq "$f" >/dev/null 2>&1; then
    printf 'ok      %s\n' "$(basename "$f")"
  else
    printf 'CORRUPT %s  <- delete and re-run\n' "$(basename "$f")"; fail=1
  fi
done
[ "$fail" -eq 0 ] && echo "all archives valid"
