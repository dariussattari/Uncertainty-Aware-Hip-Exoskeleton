#!/usr/bin/env bash
# Send one trained joint-moment estimator run to the Jetson and build its TensorRT engine there.
# Copies only that run's best.pt + config.json and the code that loads it, then runs deployment/moment_engine.py on the
# Jetson, which exports ONNX, builds the engine and fails unless it matches PyTorch (see that file's docstring).
# Usage:  deployment/send_moment.sh [run_dir] [extra moment_engine.py args, e.g. --fp16]
#         default run: hip_thigh_aug_s2, B1's best seed on validation (R^2 0.807; E41)
# Env:    JETSON (ssh host, default "jetson"), REMOTE (repo copy on the Jetson, default ~/hip-exo),
#         VENV (default ~/venvs/hipexo)
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${1:-ml/nature/estimator/runs/hip_thigh_aug_s2}
shift $(( $# > 0 ? 1 : 0 ))
JETSON=${JETSON:-jetson}
REMOTE=${REMOTE:-hip-exo}
VENV=${VENV:-venvs/hipexo}

for f in "$RUN/best.pt" "$RUN/config.json"; do
    [ -f "$f" ] || { echo "missing $f (pull the run from the cluster first)" >&2; exit 1; }
done
rsync -aR deployment/moment_engine.py ml/common/models/__init__.py ml/common/models/moment_tcn.py \
    "$RUN/best.pt" "$RUN/config.json" "$JETSON:$REMOTE/"
ssh "$JETSON" "cd $REMOTE && . ~/$VENV/bin/activate && python deployment/moment_engine.py $RUN $*"
echo "engine and moment.json are in $JETSON:$REMOTE/$RUN/"
