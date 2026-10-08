"""Stage C: the 200 ms hip-angle forecast ensemble under a holdout, plain or regime-conditioned.

    python ml/sci_rob/awareness/forecast.py --holdout DOWN --variant plain  -- --folds 1
    python ml/sci_rob/awareness/forecast.py --holdout DOWN --variant regime -- --folds 1
    python ml/sci_rob/awareness/forecast.py --holdout DOWN --variant plain --stage smoke
    python ml/sci_rob/awareness/forecast.py --holdout DOWN --variant plain --horizon 200 -- --folds 1   # 1 s ahead

Nothing about the model is reimplemented: ``ml/sci_rob/vanilla/main.py`` is driven exactly as the
LOTO runner drives it, with the registry's data factory replaced by the holdout view. The
``regime`` variant appends the regime head's class probabilities (from ``regime.py``, the
logistic probe by default) to every window as constant channels, so the ensemble sees
16 sensor channels plus one channel per known regime. The forecast target is unchanged; the
question is what the extra context does to forecast error on windows whose regime the head
cannot name.

Anything after ``--`` is passed to ``main.py`` unchanged. ``--folds N`` trims the twelve
leave-one-subject-out folds of the epoch-selection stage to N, which is the shortcut that
gets a first result back in under an hour instead of six to nine.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from aw_paths import ROOT, RUNS, SPLITS, run_dir
from splits import build

MODEL = 'forecast-angle'
HORIZON = 40


def _view(data, split, kind):
    return data.splits[split] if kind == 'id' else data.ood[split]


def conditioned(data, regime_dir: Path, probe: str = 'logistic'):
    """Append the probe's class probabilities as constant channels to every window yielded.

    Works on the holdout views in place: each view's ``take`` is wrapped so that ``x`` gains
    ``n_classes`` channels; targets and masks are untouched (the forecast target cache was
    built before wrapping and indexes sensor channels by position). ``n_channels`` is raised
    to match, which is what the model factory and the checkpoint loader read.
    """
    regime_dir = Path(regime_dir)
    n_extra = None
    for split, kind in SPLITS:
        sp = _view(data, split, kind)
        P = np.load(regime_dir / f'{probe}_{split}_{kind}.npz')['probs'].astype(np.float32)
        if len(P) != len(sp):
            raise ValueError(f'{split}/{kind}: {len(P):,} regime rows for {len(sp):,} windows; '
                             'the regime stage must come from the same holdout and encoder')
        n_extra = P.shape[1]
        base = sp.take

        def take(idx, _base=base, _P=P):
            out = _base(idx)
            x = out[0] if isinstance(out, tuple) else out
            rows = np.atleast_1d(np.asarray(idx, dtype=np.int64))
            cond = np.repeat(_P[rows][:, :, None], x.shape[-1], axis=2)
            x = np.concatenate([x, cond], axis=1)
            return (x, *out[1:]) if isinstance(out, tuple) else x

        sp.take = take
    extra = n_extra
    data.__class__ = type(data.__class__.__name__ + 'Conditioned', (data.__class__,),
                          {'shape': property(lambda self: (self.train.X.shape[1] + extra, self.train.X.shape[2]))})
    data.conditioning = {'probe': probe, 'channels': extra}
    return data


def make_builder(holdout: str, variant: str, regime_dir: Path | None = None, probe: str = 'logistic'):
    def builder(batch_size=1024, horizon=None, **rest):
        data = build('synthetic', holdout, batch_size=batch_size, target='forecast_angle',
                     horizon=HORIZON if horizon is None else horizon, **rest)
        return conditioned(data, regime_dir, probe) if variant == 'regime' else data
    return builder


def _tune_backend():
    """Same GPU settings as ml/sci_rob/LOTO/run.py: cuDNN autotune on, TF32 off for comparability."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def run(holdout: str, variant: str = 'plain', stage: str = 'all', extra=(), out_root=RUNS, probe: str = 'logistic',
        horizon: int | None = None) -> int:
    """``horizon`` in samples (200 Hz, multiples of 10); the default 40 keeps the run name bare, any
    other value appends ``_h<horizon>`` so a horizon sweep lands in sibling directories."""
    suffix = '' if horizon in (None, HORIZON) else f'_h{horizon}'
    out = run_dir(holdout, f'forecast_{variant}{suffix}', out_root)
    if horizon not in (None, HORIZON):
        extra = ['--horizon', str(horizon), *extra]
    regime_dir = run_dir(holdout, 'regime', out_root)
    if variant == 'regime' and not (regime_dir / f'{probe}_train_id.npz').exists():
        raise SystemExit(f'no regime probabilities at {regime_dir}; run encoder.py and regime.py first')
    sys.path.insert(0, str(ROOT / 'ml' / 'sci_rob' / 'vanilla'))
    import registry
    import main as vanilla_main
    _tune_backend()
    builder = make_builder(holdout, variant, regime_dir, probe)
    registry.REGISTRY[MODEL] = replace(registry.REGISTRY[MODEL], data=lambda: builder)
    argv = ['main.py', stage, '--model', MODEL, '--out', str(out), *extra]
    print(f'=== forecast ensemble | holdout {holdout} | variant {variant} | {" ".join(argv[1:])}', flush=True)
    saved, sys.argv = sys.argv, argv
    try:
        return vanilla_main.main()
    finally:
        sys.argv = saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--holdout', required=True)
    parser.add_argument('--variant', choices=('plain', 'regime'), default='plain')
    parser.add_argument('--stage', default='all', choices=('all', 'train', 'eval', 'figures', 'smoke'))
    parser.add_argument('--probe', default='logistic')
    parser.add_argument('--out-root', default=str(RUNS))
    parser.add_argument('--horizon', type=int, default=None, help='samples ahead (200 Hz); default 40 = 200 ms')
    a, extra = parser.parse_known_args(argv)
    if extra and extra[0] == '--':
        extra = extra[1:]
    return run(a.holdout, a.variant, a.stage, extra, a.out_root, a.probe, a.horizon)


if __name__ == '__main__':
    raise SystemExit(main())
