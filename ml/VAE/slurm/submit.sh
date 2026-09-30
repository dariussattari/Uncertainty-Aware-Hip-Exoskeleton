#!/bin/bash
# Submit VAE jobs. Run on a LOGIN node from the repo root.
#
#     bash ml/VAE/slurm/submit.sh smoke                      # ~15 min on gpu_test
#     bash ml/VAE/slurm/submit.sh matrix                     # 8 cells: standard/beta4 x 4 protocols
#     bash ml/VAE/slurm/submit.sh sweep                      # 7 cells: beta sweep + two beta=0 controls
#     bash ml/VAE/slurm/submit.sh seeds beta4 [original]     # seeds 1-2 of one cell
#     append --requeue to any of them for the preemptible partition
set -euo pipefail
cd "${HIPEXO_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
source ml/LOTO/slurm/env.sh >/dev/null
WHICH="${1:-smoke}"; shift || true
PART="$HIPEXO_PART"; EXTRA=""; ARGS=()
for arg in "$@"; do
  case "$arg" in
    --requeue) PART="gpu_requeue"; EXTRA="--requeue" ;;
    *) ARGS+=("$arg") ;;
  esac
done
[[ "$WHICH" == "smoke" ]] && PART="$HIPEXO_PART_TEST"
mkdir -p ml/VAE/slurm/logs
COMMON=(--account "$HIPEXO_ACCOUNT" --partition "$PART" ${EXTRA:+$EXTRA}
        --export=ALL,HIPEXO_REPO="$HIPEXO_REPO",HIPEXO_LOTO_DATA="$HIPEXO_LOTO_DATA")
case "$WHICH" in
  smoke)  sbatch "${COMMON[@]}" ml/VAE/slurm/smoke.sbatch ;;
  matrix) sbatch "${COMMON[@]}" ml/VAE/slurm/vae.sbatch ;;
  sweep)  sbatch "${COMMON[@]}" ml/VAE/slurm/sweep.sbatch ;;
  seeds)
    [[ ${#ARGS[@]} -ge 1 ]] || { echo "usage: submit.sh seeds <variant> [protocol]" >&2; exit 2; }
    VAE_VARIANT="${ARGS[0]}" VAE_PROTOCOL="${ARGS[1]:-original}" \
      sbatch "${COMMON[@]}" --export=ALL,HIPEXO_REPO="$HIPEXO_REPO",HIPEXO_LOTO_DATA="$HIPEXO_LOTO_DATA",VAE_VARIANT="${ARGS[0]}",VAE_PROTOCOL="${ARGS[1]:-original}" \
      ml/VAE/slurm/seeds.sbatch ;;
  *) echo "usage: submit.sh smoke|matrix|sweep|seeds <variant> [protocol] [--requeue]" >&2; exit 2 ;;
esac
echo "watch: squeue -u \$USER    list: python ml/VAE/vae_train.py --list"
