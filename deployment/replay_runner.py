"""On the Jetson: stream Sci-Rob trials sample by sample through the moment engine, as the device will (stage A).

    python deployment/replay_runner.py ml/nature/estimator/runs/hip_thigh_aug_s2 [--subjects dev|all|AB15,...]

Per participant file of ``scirob_replay.py`` (data/replay/scirob/, copied from the Mac) and variant: each sample of
each trial is pushed into a 249-sample ring buffer per leg; every ``STRIDE``-th sample (100 Hz) whose buffer holds 249
finite samples, the two windows are mirrored (left leg, from moment.json) and run through the TensorRT engine in one
call. The engine is the one ``moment_engine.py`` built and checked; channels, window and mirroring come from its
moment.json. Writes data/replay/scirob/predictions/jetson/<subject>.npz: ``pred_<variant>`` (n, 2) N m/kg (NaN where no
prediction), and per call ``engine_ms`` (engine only) and ``step_ms`` (window assembly, copy in, engine, copy out).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moment_engine as me  # noqa: E402
import moment_model as mm  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / 'data' / 'replay' / 'scirob'
VARIANTS = ('nominal', 'calibrated')
DEV = ('AB15', 'AB19', 'AB25', 'AB29')                  # as scirob_replay.DEV (that module needs the Mac's caches)
STRIDE = 2                                               # as replay_reference.STRIDE: 100 Hz at 200 Hz


class LegBuffers:
    """Both legs' last ``window`` samples, written twice into a (2 window) array so the current window is always one
    contiguous slice; counts consecutive finite samples so incomplete windows are skipped."""

    def __init__(self, window: int, n_channels: int) -> None:
        self.window = window
        self.data = np.zeros((2 * window, 2, n_channels), np.float32)
        self.pos = 0
        self.finite_run = 0

    def push(self, sample: np.ndarray) -> None:
        """One (2, C) sample, legs (left, right)."""
        self.data[self.pos] = sample
        self.data[self.pos + self.window] = sample
        self.pos = (self.pos + 1) % self.window
        self.finite_run = self.finite_run + 1 if np.isfinite(sample).all() else 0

    def window_ready(self) -> bool:
        return self.finite_run >= self.window

    def get_window(self) -> np.ndarray:
        """(2, C, W): legs, channels, time oldest first."""
        return self.data[self.pos:self.pos + self.window].transpose(1, 2, 0)


def get_subjects(choice: str) -> list[str]:
    files = sorted(DATA.glob('AB*.npz'))
    if choice == 'dev':
        result = [f.stem for f in files if f.stem in DEV]
    elif choice == 'all':
        result = [f.stem for f in files]
    else:
        result = choice.split(',')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split('\n')[0])
    parser.add_argument('run', type = Path)
    parser.add_argument('--subjects', default = 'dev', help = "'dev', 'all' or a comma-separated list")
    parser.add_argument('--engine', default = 'moment.engine')
    args = parser.parse_args()
    spec = json.loads((args.run / 'moment.json').read_text())
    if spec['engine'] != args.engine:
        raise ValueError(f"moment.json describes {spec['engine']}, not {args.engine}: rebuild with moment_engine.py")
    channels, window = spec['input']['channels'], spec['input']['window_samples']
    sign = mm.get_left_sign(channels)[:, None]                                  # (C, 1)
    if sign.ravel().tolist() != [-1.0 if c in spec['input']['left_leg_negate'] else 1.0 for c in channels]:
        raise ValueError('mirroring rule disagrees with moment.json')
    runner = me.EngineRunner(args.run / args.engine)
    out = DATA / 'predictions' / 'jetson'
    out.mkdir(parents = True, exist_ok = True)
    for subject in get_subjects(args.subjects):
        data = np.load(DATA / f'{subject}.npz')
        if [str(c) for c in data['channels']] != channels:
            raise ValueError(f'{subject}: replay channels {data["channels"]} do not match the engine {channels}')
        result, engine_ms, step_ms = {}, [], []
        t0 = time.perf_counter()
        for variant in VARIANTS:
            x_all = data[f'x_{variant}']                                         # (N, 2, C), legs unmirrored
            pred = np.full((len(x_all), 2), np.nan, np.float32)
            for start, length in zip(data['start'], data['length']):
                buffers = LegBuffers(window, len(channels))
                for t in range(length):
                    buffers.push(x_all[start + t])
                    if t >= window - 1 and (t - (window - 1)) % STRIDE == 0 and buffers.window_ready():
                        s0 = time.perf_counter()
                        w = buffers.get_window().copy()                          # (2, C, W)
                        w[0] *= sign                                             # mirror the left leg
                        gpu = torch.from_numpy(w).cuda(non_blocking = False)
                        e0 = time.perf_counter()
                        y = runner(gpu)                                          # (2, 1), synchronized
                        e1 = time.perf_counter()
                        pred[start + t] = y[:, 0].cpu().numpy()
                        step_ms.append(1e3 * (time.perf_counter() - s0))
                        engine_ms.append(1e3 * (e1 - e0))
            result[f'pred_{variant}'] = pred
        np.savez(out / f'{subject}.npz', **result, engine_ms = np.array(engine_ms, np.float32),
                 step_ms = np.array(step_ms, np.float32), stride = STRIDE, engine = args.engine)
        print(f'{subject}: {len(step_ms):,} calls in {time.perf_counter() - t0:.0f} s; step median '
              f'{np.median(step_ms):.2f} ms, p99 {np.percentile(step_ms, 99):.2f} ms', flush = True)


if __name__ == '__main__':
    main()
