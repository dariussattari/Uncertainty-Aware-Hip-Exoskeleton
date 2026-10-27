"""On the Jetson: stream the replay data at the device's true 100 Hz pacing for hours, timing every tick.

    python deployment/soak_test.py ml/nature/estimator/runs/hip_thigh_aug_s2 --hours 12 [--gc-freeze] [--cpu N]

Each tick (10 ms, absolute schedule) pushes the next two 200 Hz samples of both legs into the ring buffers and, once a
full window exists, runs one two-leg engine call, as ``replay_runner.py`` does; then sleeps until the next tick. Cycles
through every participant file (calibrated variant) until ``--hours`` have passed. Per tick: step time, wake-up
lateness (how late the tick started), and whether the step ended after the next tick's start (a missed deadline).
Every minute: resident memory. Saves data/replay/scirob/soak/soak.npz every 10 minutes (atomically), so an interrupted
soak keeps what it measured. Temperatures and power come from tegrastats, logged alongside by overnight.sh.
"""
import argparse
import gc
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moment_engine as me  # noqa: E402
import moment_model as mm  # noqa: E402
from replay_runner import DATA, LegBuffers  # noqa: E402

TICK_S = 0.01                                            # 100 Hz
SAMPLES_PER_TICK = 2                                     # 200 Hz data
SAVE_EVERY_S = 600


def get_rss_mb() -> float:
    with open('/proc/self/status') as f:
        line = next(l for l in f if l.startswith('VmRSS'))
    return int(line.split()[1]) / 1024


def save(out: Path, **arrays: np.ndarray) -> None:
    tmp = out.with_name(f'{out.stem}.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, out)


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split('\n')[0])
    parser.add_argument('run', type = Path)
    parser.add_argument('--hours', type = float, default = 12.0)
    parser.add_argument('--engine', default = 'moment.engine')
    parser.add_argument('--gc-freeze', action = 'store_true')
    parser.add_argument('--cpu', type = int, default = None)
    args = parser.parse_args()
    channels = mm.load_run(args.run)[1]['channels']
    sign = mm.get_left_sign(channels)[:, None]                                  # (C, 1)
    runner = me.EngineRunner(args.run / args.engine)
    files = sorted(DATA.glob('AB*.npz'))
    streams = [np.load(f)['x_calibrated'] for f in files]                     # (N, 2, C) each, trials concatenated
    out_dir = DATA / 'soak'
    out_dir.mkdir(parents = True, exist_ok = True)
    out = out_dir / 'soak.npz'
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    if args.gc_freeze:
        gc.collect()
        gc.freeze()
        gc.disable()
    n_ticks = int(args.hours * 3600 / TICK_S)
    step_ms = np.full(n_ticks, np.nan, np.float32)
    late_ms = np.full(n_ticks, np.nan, np.float32)
    missed = np.zeros(n_ticks, bool)
    rss = []
    buffers = LegBuffers(runner.buffers['window'].shape[2], len(channels))
    stream, pos = 0, 0
    start = time.perf_counter()
    last_save = start
    print(f'soak: {args.hours} h, {n_ticks:,} ticks, {len(files)} participant files, gc frozen {args.gc_freeze}, '
          f'cpu {args.cpu}', flush = True)
    for k in range(n_ticks):
        deadline = start + k * TICK_S
        now = time.perf_counter()
        if deadline > now:
            time.sleep(deadline - now)
        t0 = time.perf_counter()
        late_ms[k] = 1e3 * (t0 - deadline)
        for _ in range(SAMPLES_PER_TICK):                                       # the next two samples, wrapping files
            if pos >= len(streams[stream]):
                stream, pos = (stream + 1) % len(streams), 0
                buffers = LegBuffers(buffers.window, len(channels))             # a new recording starts empty
            buffers.push(streams[stream][pos])
            pos += 1
        if buffers.window_ready():
            w = buffers.get_window().copy()
            w[0] *= sign
            runner(torch.from_numpy(w).cuda())
        t1 = time.perf_counter()
        step_ms[k] = 1e3 * (t1 - t0)
        missed[k] = t1 > deadline + TICK_S
        if k % 6000 == 0:                                                        # once a minute
            rss.append((k, get_rss_mb()))
        if t1 - last_save > SAVE_EVERY_S:
            save(out, step_ms = step_ms[:k + 1], late_ms = late_ms[:k + 1], missed = missed[:k + 1], rss = np.array(rss),
                 tick_s = TICK_S, gc_frozen = args.gc_freeze, complete = False)
            last_save = t1
            print(f'{(t1 - start) / 3600:.2f} h: {k + 1:,} ticks, missed {int(missed[:k + 1].sum())}, '
                  f'step p99 {np.nanpercentile(step_ms[:k + 1], 99):.2f} ms, rss {rss[-1][1]:.0f} MB', flush = True)
    save(out, step_ms = step_ms, late_ms = late_ms, missed = missed, rss = np.array(rss), tick_s = TICK_S,
         gc_frozen = args.gc_freeze, complete = True)
    print(f'soak done: {n_ticks:,} ticks, missed {int(missed.sum())}, step median {np.nanmedian(step_ms):.2f} ms, '
          f'p99 {np.nanpercentile(step_ms, 99):.2f} ms, max {np.nanmax(step_ms):.1f} ms', flush = True)


if __name__ == '__main__':
    main()
