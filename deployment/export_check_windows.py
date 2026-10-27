"""Sample real validation windows for a moment-estimator run, for the on-device check (check_moment_real.py).

    .venv/bin/python deployment/export_check_windows.py ml/nature/estimator/runs/hip_thigh_aug_s2

Draws ``--n`` windows of the validation participants (all 26 tasks, a defined hip target) from the run's
``preds_val.npz`` and writes ``<run>/check_val.npz``: the raw input windows (N, C, 249) exactly as training fed them
(``train_moment.get_batch``, before standardization), the hip target, the cluster's saved prediction (bf16 autocast) and
the Nature study's own deployed estimator for the same windows, handled as ``score_moment.py`` does: hip sign flipped to
OpenSim flexion-positive, and only on Phase 3 recordings (where the final model ran; NaN elsewhere). Needs data/processed_nature/moment. Validation
participants only: the test participants stay untouched.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data' / 'processed_nature' / 'moment'
sys.path.insert(0, str(REPO / 'ml' / 'nature' / 'estimator'))
from score_moment import DEPLOYED_SIGN  # noqa: E402  (hip extension-positive -> OpenSim flexion-positive)


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split('\n')[0])
    parser.add_argument('run', type = Path)
    parser.add_argument('--n', type = int, default = 2048, help = 'windows (even: the engine runs two per call)')
    parser.add_argument('--seed', type = int, default = 0)
    args = parser.parse_args()
    if args.n % 2:
        raise ValueError('--n must be even')

    meta = json.loads((DATA / 'meta.json').read_text())
    config = json.loads((args.run / 'config.json').read_text())
    channels = config['channels']
    preds = np.load(args.run / 'preds_val.npz')
    if preds['target'].shape[1] != 1:
        raise ValueError('expected a hip-only run (one target column)')
    defined = np.flatnonzero(np.isfinite(preds['target'][:, 0]))
    pick = np.sort(np.random.default_rng(args.seed).choice(defined, args.n, replace = False))   # indices into preds_val
    rows = preds['rows'][pick]                                                  # rows of windows.parquet

    windows = pd.read_parquet(DATA / 'windows.parquet')
    if not (windows['split'].to_numpy()[rows] == 'val').all():                # rows are positional, as in training
        raise ValueError('preds_val rows are not all validation windows')
    starts, legs = windows['start'].to_numpy(np.int64)[rows], windows['leg'].to_numpy(np.int64)[rows]
    inputs = np.load(DATA / 'inputs.npy', mmap_mode = 'r')                      # (n_samples, 2, 25), raw units
    cols = [meta['channels'].index(c) for c in channels]
    window = int(meta['window'])
    x = np.stack([inputs[s:s + window, leg][:, cols] for s, leg in zip(starts, legs)])   # (N, W, C)
    x = np.ascontiguousarray(x.transpose(0, 2, 1), dtype = np.float32)        # (N, W, C) -> (N, C, W)
    if not np.isfinite(x).all():
        raise FloatingPointError('non-finite input windows')

    out = args.run / 'check_val.npz'
    np.savez(out, x = x, target = preds['target'][pick, 0], cluster_pred = preds['pred'][pick, 0],
             paper_estimate = np.where(windows['phase'].to_numpy()[rows] == 'phase3', DEPLOYED_SIGN[0] * preds['estimate'][pick, 0], np.nan),
             rows = rows, legs = legs,
             tasks = windows['task'].to_numpy()[rows].astype(str), data_version = np.array(config['data_version']))   # plain str, no pickles
    print(f'wrote {out}: {args.n} validation windows, {x.nbytes / 1e6:.0f} MB, '
          f'{len(set(windows["task"].astype(str).to_numpy()[rows]))} tasks')


if __name__ == '__main__':
    main()
