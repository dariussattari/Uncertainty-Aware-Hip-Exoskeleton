#!/usr/bin/env bash
# Unattended device-test battery on the Jetson, then shutdown. Run from ~/hip-exo, detached:
#   setsid nohup bash deployment/overnight.sh > logs/overnight/run.log 2>&1 < /dev/null &
# SMOKE=1 runs every step on a few trials / seconds into logs/overnight_smoke (outputs to *_smoke folders), no shutdown.
#
#   T1  final 16 participants, calibrated, sealed (predictions only)       replay_runner.py --out jetson_final
#   T4  per-window latency budget of the whole controller (stand-ins)      latency_budget.py
#   T2  jitter study on dev: baseline, gc frozen, pinned core, both        replay_runner.py --out jitter_*
#   T3  fp16 engine: build, real-data check, dev replay                     moment_engine.py --fp16, check_moment_real.py
#   T5  soak at true 100 Hz pacing (12 h), tegrastats throughout           soak_test.py
# Each step logs to its own file and a failure does not stop the next. Shuts down only if the sudoers rule for
# /usr/sbin/shutdown exists (sudo -n works); otherwise leaves the board idling.
set -uo pipefail
cd "$(dirname "$0")/.."
. ~/venvs/hipexo/bin/activate

RUN=ml/nature/estimator/runs/hip_thigh_aug_s2
SMOKE=${SMOKE:-0}
SOAK_HOURS=${SOAK_HOURS:-12}
CPU=3                                                     # a core away from cpu0, which takes most interrupts
if [ "$SMOKE" = 1 ]; then
    LOG=logs/overnight_smoke; SUFFIX=_smoke; LIMIT="--limit-trials 2"; SUBJ_FINAL=AB16; SUBJ_DEV=AB15; SOAK_HOURS=0.005
else
    LOG=logs/overnight; SUFFIX=; LIMIT=; SUBJ_FINAL=final; SUBJ_DEV=dev
fi
mkdir -p "$LOG"
SUMMARY=$LOG/summary.txt

note() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }
step() {                                                  # step <name> <command...>: run, log, record exit status
    local name=$1; shift
    note "start $name: $*"
    local t0=$SECONDS
    "$@" > "$LOG/$name.log" 2>&1
    local rc=$?
    note "end   $name: exit $rc after $(( (SECONDS - t0) / 60 )) min; last lines:"
    grep -v -i warning "$LOG/$name.log" | tail -4 | sed 's/^/        /' | tee -a "$SUMMARY"
}

note "overnight battery (SMOKE=$SMOKE) on $(hostname), git $(cat VERSION 2>/dev/null || echo 'n/a'), soak $SOAK_HOURS h"
CAN_SHUTDOWN=0
sudo -n -l /usr/sbin/shutdown > /dev/null 2>&1 && CAN_SHUTDOWN=1
note "auto shutdown available: $CAN_SHUTDOWN"
tegrastats --interval 1000 --logfile "$LOG/tegrastats.log" &
TEGRA=$!

step T1_final_sealed python -u deployment/replay_runner.py $RUN --subjects $SUBJ_FINAL --variants calibrated --out jetson_final$SUFFIX $LIMIT
step T4_latency_budget python -u deployment/latency_budget.py $RUN
step T2_jitter_baseline python -u deployment/replay_runner.py $RUN --subjects $SUBJ_DEV --variants calibrated --out jitter_baseline$SUFFIX $LIMIT
step T2_jitter_gcfreeze python -u deployment/replay_runner.py $RUN --subjects $SUBJ_DEV --variants calibrated --out jitter_gcfreeze$SUFFIX $LIMIT --gc-freeze
step T2_jitter_cpu python -u deployment/replay_runner.py $RUN --subjects $SUBJ_DEV --variants calibrated --out jitter_cpu$SUFFIX $LIMIT --cpu $CPU
step T2_jitter_both python -u deployment/replay_runner.py $RUN --subjects $SUBJ_DEV --variants calibrated --out jitter_both$SUFFIX $LIMIT --gc-freeze --cpu $CPU
step T3_fp16_build python -u deployment/moment_engine.py $RUN --fp16
step T3_fp16_real_check bash -c "cd deployment && python -u check_moment_real.py ../$RUN --engine moment_fp16.engine"
step T3_fp16_replay python -u deployment/replay_runner.py $RUN --subjects $SUBJ_DEV --variants calibrated --engine moment_fp16.engine --out jetson_fp16$SUFFIX $LIMIT
step T5_soak python -u deployment/soak_test.py $RUN --hours $SOAK_HOURS

kill $TEGRA 2> /dev/null
note "battery complete; outputs: data/replay/scirob/predictions/{jetson_final,jitter_*,jetson_fp16}$SUFFIX, data/replay/scirob/soak, data/replay/scirob/latency_budget.json, $LOG/"
sync
if [ "$SMOKE" != 1 ] && [ "$CAN_SHUTDOWN" = 1 ]; then
    note "shutting down in 1 minute"
    sudo -n /usr/sbin/shutdown -h +1
else
    note "no shutdown (smoke run, or no sudoers rule); the board stays on"
fi
