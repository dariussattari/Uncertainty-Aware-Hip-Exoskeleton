#!/usr/bin/env bash
# Copy what the Jetson runs into deployment/bundle/, keeping repo-relative paths so the ml/ imports resolve unchanged.
# The list follows deployment/deployability_status.md ("What goes on the Jetson"); edit both together.
# Usage:  deployment/bundle.sh && rsync -a --delete --filter='P *.engine' --filter='P moment.onnx' --filter='P moment.json' \
#             deployment/bundle/ jetson:~/hip-exo/        (P = keep the engines built on the Jetson)
# On the Jetson:  cd ~/hip-exo && sha256sum -c SHA256SUMS
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=deployment/bundle
VAE=notebooks/nature/model_exploration/runs/vae
TCN=notebooks/nature/model_exploration/runs_train_class/id_ood_safe/vae
EST=ml/nature/estimator/runs

FILES=(
    # Model definitions and causal feature / IMU code
    ml/common/models/__init__.py
    ml/common/models/moment_tcn.py      # B1 joint-moment estimator
    ml/common/models/seq_vae.py         # LSTM-VAE (novelty) and TCN-VAE (regime latent)
    ml/common/features.py               # spectral features for the TCN-VAE input
    ml/common/signals.py                # IMU frame solver for per-wearer calibration
    ml/nature/estimator/imu_fit.py      # IMU rotation / lever-arm model (E41)
    # Awareness layer: novelty scores, regime classifier, u cutoffs (fitted by deployment/export_awareness.py)
    deployment/awareness.py
    deployment/test_awareness.py
    # Joint-moment estimator: ONNX -> TensorRT engine, built and checked on the Jetson
    deployment/moment_engine.py
    deployment/moment_model.py
    deployment/check_moment_real.py
    # Guide
    deployment/deployability_status.md
)
for s in 0 1 2; do
    # best.pt holds weights + input standardization (mean, std) + config, so no data files are needed
    FILES+=("$EST/hip_thigh_aug_s$s/best.pt" "$EST/hip_thigh_aug_s$s/config.json")
    FILES+=("$VAE/lstm_vae_raw_3s_b1_fixed_s$s/best.pt" "$VAE/lstm_vae_raw_3s_b1_fixed_s$s/config.json")
    FILES+=("$TCN/tcn_vae_raw_3s_spectral_b128_fixed_s$s/best.pt" "$TCN/tcn_vae_raw_3s_spectral_b128_fixed_s$s/config.json")
    FILES+=("deployment/artifacts/awareness_s$s.npz")
done

missing=()
for f in "${FILES[@]}"; do
    [ -f "$f" ] || missing+=("$f")
done
if [ ${#missing[@]} -gt 0 ]; then
    printf 'missing (pull the runs from the cluster, or run deployment/export_awareness.py):\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"
rsync -aR "${FILES[@]}" "$OUT/"
{
    echo "git_sha $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- "${FILES[@]}" 2>/dev/null || echo -dirty)"
    echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/VERSION"
(cd "$OUT" && find . -type f ! -name SHA256SUMS | sort | xargs shasum -a 256 > SHA256SUMS)
du -sh "$OUT"
