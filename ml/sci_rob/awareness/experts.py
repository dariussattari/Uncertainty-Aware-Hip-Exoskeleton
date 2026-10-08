"""Stage C': regime experts, so that the regime input cannot be ignored.

    python ml/sci_rob/awareness/experts.py train --holdout DOWN --regime level  -- --folds 1
    python ml/sci_rob/awareness/experts.py train --holdout DOWN --regime ascend -- --folds 1
    python ml/sci_rob/awareness/experts.py score --holdout DOWN          # laptop: needs both checkpoints
    python ml/sci_rob/awareness/experts.py swap-test --holdout DOWN      # how much the conditioned forecaster used its regime input

Feeding the regime probabilities to one forecaster as extra channels did not make it depend on
them: on in-distribution windows a swapped regime label moved its error by only 5-7 % (the
regime is already readable from the raw signal, so the channels were never load-bearing).
Experts make the dependence structural. One ensemble is trained per regime present in
training, on that regime's windows only; at inference the regime probe routes a window to an
expert. A window from an unseen regime is routed to an expert that never saw its dynamics, so
its error rises by construction, and a second quantity appears: the disagreement between the
experts' mean forecasts.

Scores written by ``score`` (all test windows, ID rows then OOD rows, causally filtered):

* ``expert_<regime>_error`` / ``_disagreement``  each expert alone
* ``expert_routed_error`` / ``_disagreement``    the expert the logistic probe chose
* ``expert_min_error``                           the best any expert could do
* ``expert_gap``                                 squared distance between the experts' mean forecasts
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from aw_paths import REGIMES, ROOT, RUNS, SPLITS, run_dir
from splits import build, held_out_modes, restrict
from evaluation.common import smooth_by_trial

MODEL = 'forecast-angle'
HORIZON = 40
SMOOTH = 10


def regime_modes(holdout: str) -> dict[str, list[str]]:
    """regime -> the trained-on modes belonging to it, e.g. DOWN: level=[LG], ascend=[RA, SA]."""
    out = {}
    for mode, regime in REGIMES.items():
        if mode not in held_out_modes(holdout):
            out.setdefault(regime, []).append(mode)
    return out


def restrict_to_modes(data, modes):
    """Keep only ``modes`` in the train/val/test ID views (OOD views untouched), fixing the target caches."""
    modes = list(modes)
    cache = getattr(data, '_cache', None)
    new_cache = dict(cache) if cache is not None else None
    for name in ('train', 'val', 'test'):
        sp = data.splits[name]
        rows = np.flatnonzero(sp.meta['mode'].isin(modes).to_numpy())
        if not len(rows):
            raise ValueError(f'no {modes} windows in the {name} split')
        view = restrict(sp, rows)
        data.splits[name] = view
        if cache is not None and id(sp) in cache:
            new_cache[id(view)] = cache[id(sp)][rows]
        if getattr(data, '_fwd', None) and name in data._fwd:
            data._fwd[name] = data._T.forecast_rows(view.meta, data.horizon)
    data.train, data.val, data.test = (data.splits[s] for s in ('train', 'val', 'test'))
    if new_cache is not None:
        data._cache = new_cache
    data.expert_modes = modes
    return data


def make_builder(holdout: str, regime: str | None):
    modes = regime_modes(holdout)[regime] if regime else None

    def builder(batch_size=1024, horizon=None, **rest):
        data = build('synthetic', holdout, batch_size=batch_size, target='forecast_angle',
                     horizon=HORIZON if horizon is None else horizon, **rest)
        return restrict_to_modes(data, modes) if modes else data
    return builder


def train(holdout: str, regime: str, stage: str = 'all', extra=(), out_root=RUNS) -> int:
    out = run_dir(holdout, f'expert_{regime}', out_root)
    modes = regime_modes(holdout)
    if regime not in modes:
        raise SystemExit(f'{regime!r} is not a regime present in training under {holdout}; have {list(modes)}')
    sys.path.insert(0, str(ROOT / 'ml' / 'sci_rob' / 'vanilla'))
    import registry
    import main as vanilla_main
    from forecast import _tune_backend
    _tune_backend()
    builder = make_builder(holdout, regime)
    registry.REGISTRY[MODEL] = replace(registry.REGISTRY[MODEL], data=lambda: builder)
    argv = ['main.py', stage, '--model', MODEL, '--out', str(out), *extra]
    print(f'=== regime expert | holdout {holdout} | regime {regime} = {modes[regime]} | {" ".join(argv[1:])}', flush=True)
    saved, sys.argv = sys.argv, argv
    try:
        rc = vanilla_main.main()
    finally:
        sys.argv = saved
    (out / 'expert.json').write_text(json.dumps({'holdout': holdout, 'regime': regime, 'modes': modes[regime]}, indent=2))
    return rc


@torch.inference_mode()
def _forecast(model, loader, device):
    """Per window: masked error of the ensemble mean, branch variance, and the mean forecast."""
    E, A, F = [], [], []
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred = model(x)                                # (n, members, targets)
        fbar = pred.mean(1)
        w = m / m.sum(1, keepdim=True).clamp(min=1)    # average over valid target dims only
        E.append((((fbar - y) ** 2) * w).sum(1).cpu().numpy())
        A.append((pred.var(1, unbiased=False) * w).sum(1).cpu().numpy())
        F.append(fbar.cpu().numpy())
    return np.concatenate(E), np.concatenate(A), np.concatenate(F)


def score(holdout: str, out_root=RUNS, probe: str = 'logistic', device=None):
    from evaluation.ensemble import load_checkpoint
    from training.common import pick_device
    device = pick_device(device)
    modes = regime_modes(holdout); regimes = sorted(modes)
    data = make_builder(holdout, None)(batch_size=1024)
    meta = pd.concat([data.test.meta.assign(kind='ID'), data.ood['test'].meta.assign(kind='OOD')], ignore_index=True)
    loaders = [data.loader('test', batch_size=1024), data.loader(data.ood['test'], batch_size=1024)]
    per = {}
    for regime in regimes:
        ck = run_dir(holdout, f'expert_{regime}', out_root) / 'final.pt'
        if not ck.exists():
            raise SystemExit(f'missing expert checkpoint {ck}')
        model, _ = load_checkpoint(ck, data, device)
        parts = [_forecast(model, ld, device) for ld in loaders]
        per[regime] = tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
        print(f'  expert {regime}: median error ID {np.median(per[regime][0][meta.kind == "ID"]):.4f} '
              f'OOD {np.median(per[regime][0][meta.kind == "OOD"]):.4f}', flush=True)
    # routing by the probe's most likely regime (its classes are the same sorted regime names)
    reg = run_dir(holdout, 'regime', out_root)
    P = np.concatenate([np.load(reg / f'{probe}_test_{k}.npz')['probs'] for k in ('id', 'ood')])
    classes = [str(c) for c in np.load(reg / f'{probe}_test_id.npz')['classes']]
    if classes != regimes:
        raise ValueError(f'probe classes {classes} differ from expert regimes {regimes}')
    choice = P.argmax(1)
    E = np.stack([per[r][0] for r in regimes], 1); A = np.stack([per[r][1] for r in regimes], 1)
    F = np.stack([per[r][2] for r in regimes], 1)                  # (n, regimes, targets)
    rows = np.arange(len(meta))
    raw = {f'expert_{r}_error': E[:, i] for i, r in enumerate(regimes)}
    raw.update({f'expert_{r}_disagreement': A[:, i] for i, r in enumerate(regimes)})
    raw['expert_routed_error'] = E[rows, choice]
    raw['expert_routed_disagreement'] = A[rows, choice]
    raw['expert_min_error'] = E.min(1)
    gap = np.zeros(len(meta))
    for i in range(len(regimes)):
        for j in range(i + 1, len(regimes)):
            gap += ((F[:, i] - F[:, j]) ** 2).mean(1)
    raw['expert_gap'] = gap
    smooth = {k: smooth_by_trial(v, meta, SMOOTH, 'median') for k, v in raw.items()}
    out = run_dir(holdout, 'experts', out_root); out.mkdir(parents=True, exist_ok=True)
    labels = (meta['kind'] == 'OOD').to_numpy().astype(int)
    np.savez_compressed(out / 'scores_test.npz', labels=labels, routed=choice, regimes=np.array(regimes),
                        **{k: v for k, v in raw.items()}, **{f'smooth_{k}': v for k, v in smooth.items()})
    meta.to_parquet(out / 'scores_test.parquet', index=False)
    from sklearn.metrics import roc_auc_score
    summary = {k: float(roc_auc_score(labels, v)) for k, v in smooth.items()}
    summary['routed_share'] = {r: float(np.mean(choice == i)) for i, r in enumerate(regimes)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        if isinstance(v, float):
            print(f'  {k:32s} AUROC {v:.4f}')
    return out


def swap_test(holdout: str, out_root=RUNS, probe: str = 'logistic', device=None, results=None):
    """Median realized error of the regime-conditioned forecaster with the true versus swapped regime input."""
    import shutil, tempfile
    from forecast import make_builder as cond_builder
    from evaluation.ensemble import load_checkpoint
    from evaluation.ambiguity import decompose
    from training.common import pick_device
    run = run_dir(holdout, 'forecast_regime', out_root); reg = run_dir(holdout, 'regime', out_root)
    if not (run / 'final.pt').exists():
        raise SystemExit(f'no conditioned forecaster at {run}')
    tmp = Path(tempfile.mkdtemp())
    for split, kind in SPLITS:
        z = np.load(reg / f'{probe}_{split}_{kind}.npz')
        np.savez(tmp / f'{probe}_{split}_{kind}.npz', probs=z['probs'][:, ::-1].copy(), confidence=z['confidence'],
                 entropy=z['entropy'], classes=z['classes'])
    device = pick_device(device)
    table = {}
    for name, rdir in (('true regime input', reg), ('swapped regime input', tmp)):
        data = cond_builder(holdout, 'regime', rdir, probe)(batch_size=1024)
        model, thr = load_checkpoint(run / 'final.pt', data, device)
        d = decompose(model, data, thr, 'test', device)
        modes = d.meta['mode'].to_numpy()
        table[name] = {m: float(np.median(d.E_s[modes == m])) for m in ('LG', 'RA', 'SA', 'RD', 'SD', 'ST', 'TR') if np.any(modes == m)}
    shutil.rmtree(tmp)
    df = pd.DataFrame(table); df['ratio swapped / true'] = df['swapped regime input'] / df['true regime input']
    df.index.name = 'mode'
    if results is not None:
        Path(results).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(results) / 'regime_swap_test.csv')
    print(df.round(4).to_string())
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('train', 'score', 'swap-test'))
    parser.add_argument('--holdout', required=True)
    parser.add_argument('--regime', default=None, help='train: which regime expert')
    parser.add_argument('--stage', default='all', choices=('all', 'train', 'eval', 'figures', 'smoke'))
    parser.add_argument('--probe', default='logistic')
    parser.add_argument('--out-root', default=str(RUNS))
    parser.add_argument('--device', default=None)
    a, extra = parser.parse_known_args(argv)
    if extra and extra[0] == '--':
        extra = extra[1:]
    if a.command == 'train':
        if not a.regime:
            parser.error('train needs --regime')
        return train(a.holdout, a.regime, a.stage, extra, a.out_root)
    if a.command == 'score':
        score(a.holdout, a.out_root, a.probe, a.device); return 0
    from aw_paths import RESULTS
    swap_test(a.holdout, a.out_root, a.probe, a.device, RESULTS / a.holdout); return 0


if __name__ == '__main__':
    raise SystemExit(main())
