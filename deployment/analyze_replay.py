"""Score the Sci-Rob replay (stage A): B1 vs inverse dynamics vs the estimator that ran on the exo.

    .venv/bin/python deployment/analyze_replay.py [--source jetson|reference] [--subjects dev|final|all]

Rules, fixed before scoring:
  - B1's prediction at sample t estimates the hip moment at t - 10 (trained 50 ms behind its window); the online
    estimate ``trq_est`` is scored at t. Both against the inverse-dynamics ``moment`` (NaN off the force plates).
  - Every number compares the estimators on the same samples: where B1, trq_est and the moment all exist.
  - Sci-Rob's moment / trq_est sign convention is set from the dev participants only: the sign that makes B1's
    correlation with the moment positive (B1 is OpenSim flexion-positive). Its lag scan is printed for the record.
  - Dev participants (scirob_replay.DEV) inform decisions; ``--subjects final`` scores the other 16, once.
With both sources present, also reports how far the Jetson's predictions are from the PyTorch reference.
Writes data/replay/scirob/predictions/<source>_scores.csv.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moment_model as mm  # noqa: E402
import scirob_replay as sr  # noqa: E402

PRED = sr.OUT / 'predictions'
LAG = mm.HIP_DELAY_SAMPLES


def get_subjects(choice: str) -> list[str]:
    files = sorted(f.stem for f in sr.OUT.glob('AB*.npz'))
    return {'dev': [s for s in files if s in sr.DEV], 'final': [s for s in files if s not in sr.DEV], 'all': files}[choice]


def get_rows(subject: str, source: str) -> pd.DataFrame:
    """One row per (sample, leg) with a prediction: B1 per variant, trq_est at t, moment at t and at t - LAG, mode."""
    data, pred = np.load(sr.OUT / f'{subject}.npz'), np.load(PRED / source / f'{subject}.npz')
    n = len(data['moment'])
    trial = np.repeat(np.arange(len(data['start'])), data['length'])            # (n,) trial index per sample
    t_in_trial = np.arange(n) - np.repeat(data['start'], data['length'])
    lagged = np.full((n, 2), np.nan, np.float32)
    ok = t_in_trial >= LAG                                                     # the moment LAG samples earlier, same trial
    lagged[ok] = data['moment'][np.flatnonzero(ok) - LAG]
    frames = []
    missing = np.full(n, np.nan, np.float32)
    get = lambda v, leg: pred[f'pred_{v}'][:, leg] if f'pred_{v}' in pred.files else missing   # noqa: E731  (calibrated-only runs)
    for leg, side in enumerate(('l', 'r')):
        has = np.isfinite(get('calibrated', leg))
        idx = np.flatnonzero(has)
        frames.append(pd.DataFrame({
            'subject': subject, 'leg': side, 'trial': data['trial'][trial[idx]], 'mode': data['mode'][trial[idx]],
            'b1_nominal': get('nominal', leg)[idx], 'b1_calibrated': get('calibrated', leg)[idx],
            'trq_est': data['trq_est'][idx, leg], 'moment_t': data['moment'][idx, leg], 'moment_lag': lagged[idx, leg]}))
    return pd.concat(frames, ignore_index = True)


def get_sign(rows: pd.DataFrame) -> float:
    """+1 or -1: Sci-Rob's moment convention relative to B1's, from dev rows only."""
    ok = rows[['b1_calibrated', 'moment_lag']].notna().all(axis = 1)
    r = np.corrcoef(rows.loc[ok, 'b1_calibrated'], rows.loc[ok, 'moment_lag'])[0, 1]
    if abs(r) < 0.3:
        raise ValueError(f'B1 barely correlates with the moment on dev (r = {r:.2f}): no convention can be read off')
    return float(np.sign(r))


def get_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {'r2': float(1 - np.sum(err ** 2) / np.sum((target - target.mean()) ** 2)), 'rmse': float(np.sqrt(np.mean(err ** 2))),
            'r': float(np.corrcoef(pred, target)[0, 1])}


def score(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group: n and R^2 / RMSE / r of each estimator on the samples where all three exist."""
    b1 = [c for c in ('b1_nominal', 'b1_calibrated') if rows[c].notna().any()]          # calibrated-only runs have no nominal
    common = rows[[*b1, 'trq_est', 'moment_t', 'moment_lag']].notna().all(axis = 1)
    out = []
    for key, g in rows[common].groupby(by):
        entry = dict(zip(by, key if isinstance(key, tuple) else (key,)), n = len(g))
        for name, pred, target in [(f'B1 {c[3:]}', c, 'moment_lag') for c in b1] + [('online trq_est', 'trq_est', 'moment_t')]:
            entry.update({f'{name} {k}': v for k, v in get_metrics(g[pred].to_numpy(), g[target].to_numpy()).items()})
        out.append(entry)
    return pd.DataFrame(out)


def get_lag_scan(subject_files: list[str], sign: float) -> pd.DataFrame:
    """r of B1 (calibrated) and trq_est with the moment shifted by -30..30 samples, pooled over dev trials; B1's r times
    ``sign`` (trq_est already shares the moment's convention)."""
    result = []
    for lag in range(-30, 31, 5):
        rs = {'lag_samples': lag}
        for name, col in (('B1 calibrated', 'pred_calibrated'), ('online trq_est', None)):
            preds, targets = [], []
            for subject in subject_files:
                data = np.load(sr.OUT / f'{subject}.npz')
                p = np.load(PRED / 'reference' / f'{subject}.npz')[col] if col else data['trq_est']
                for start, length in zip(data['start'], data['length']):
                    a = p[start:start + length]
                    m = data['moment'][start:start + length]
                    a, m = (a[lag:], m[:length - lag]) if lag >= 0 else (a[:length + lag], m[-lag:])   # pred(t) vs moment(t - lag)
                    ok = np.isfinite(a) & np.isfinite(m)
                    preds.append(a[ok])
                    targets.append(m[ok])
            rs[name] = (sign if col else 1.0) * float(np.corrcoef(np.concatenate(preds), np.concatenate(targets))[0, 1])
        result.append(rs)
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split('\n')[0])
    parser.add_argument('--source', default = 'jetson', help = "predictions folder: jetson, reference, jetson_final, jetson_fp16, ...")
    parser.add_argument('--subjects', default = 'dev', choices = ('dev', 'final', 'all'))
    args = parser.parse_args()
    pd.set_option('display.width', 250)
    dev = [s for s in get_subjects('dev') if (PRED / 'reference' / f'{s}.npz').exists()]
    sign = get_sign(pd.concat([get_rows(s, 'reference') for s in dev]))
    print(f'Sci-Rob moment/trq_est sign relative to B1 (from dev {dev}): {sign:+.0f}')
    print('lag scan on dev (r of prediction(t) with moment(t - lag)):')
    print(get_lag_scan(dev, sign).round(3).to_string(index = False))

    subjects = get_subjects(args.subjects)
    rows = pd.concat([get_rows(s, args.source) for s in subjects], ignore_index = True)
    rows[['trq_est', 'moment_t', 'moment_lag']] *= sign
    rows['pooled'] = 'all'
    for title, by in (('pooled', ['pooled']), ('per mode', ['mode']), ('per participant', ['subject'])):
        table = score(rows, by)
        cols = [c for c in table if c in by + ['n'] or c.endswith(('r2', 'rmse'))]
        print(f'\n{args.source}, {args.subjects} participants, {title}:')
        print(table[cols].round(3).to_string(index = False))
    by_leg = score(rows, ['leg'])
    print('\nper leg:')
    print(by_leg[[c for c in by_leg if c in ('leg', 'n') or c.endswith('r2')]].round(3).to_string(index = False))
    score(rows, ['subject', 'mode']).to_csv(PRED / f'{args.source}_{args.subjects}_scores.csv', index = False)

    if args.source.startswith('jetson') or args.source.startswith('jitter'):
        diffs = []
        for s in subjects:
            ref_file = PRED / 'reference' / f'{s}.npz'
            if ref_file.exists():
                ref, jet = np.load(ref_file), np.load(PRED / args.source / f'{s}.npz')
                for v in [v for v in sr.VARIANTS if f'pred_{v}' in jet.files]:
                    a, b = ref[f'pred_{v}'], jet[f'pred_{v}']
                    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                        raise AssertionError(f'{s} {v}: the Jetson predicted on different samples than the reference')
                    diffs.append(np.abs(a - b)[np.isfinite(a)])
                timing = jet['step_ms']
                print(f'{s}: Jetson step median {np.median(timing):.2f} ms, p99 {np.percentile(timing, 99):.2f} ms, '
                      f'max {timing.max():.1f} ms; engine median {np.median(jet["engine_ms"]):.2f} ms')
        if diffs:
            d = np.concatenate(diffs)
            print(f'Jetson vs PyTorch reference over {len(d):,} predictions: max {d.max():.2e}, 99.9th pct '
                  f'{np.percentile(d, 99.9):.2e} N m/kg')


if __name__ == '__main__':
    main()
