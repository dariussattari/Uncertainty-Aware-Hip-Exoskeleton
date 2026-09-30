"""Paths and shared constants for the awareness layer (novelty, regime, reliability).

This package assembles the signals an aware controller would read, under a holdout that
removes a whole movement regime from training. It forks nothing: the encoder comes from
``ml/VAE`` (its deterministic β=0 configuration), the forecast ensemble from ``ml/vanilla``,
the holdout views from ``ml/LOTO/splits.py``. Only the regime probes, the conditioning of
the forecaster on regime probabilities, and the score collection are new.

Importing this module only touches ``sys.path``; nothing here needs IPython or a display.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'ml/vanilla/registry.py').is_file())
RUNS = HERE / 'runs'          # runs/<holdout>/<stage>/ ... ignored by git, pulled from the cluster
RESULTS = HERE / 'results'    # tracked summaries, tables and figures

for _folder in ('ml/vanilla', 'ml/LOTO', 'ml/VAE'):
    if str(ROOT / _folder) not in sys.path:
        sys.path.append(str(ROOT / _folder))

# Movement regimes. A probe is trained on the regimes present in training only; a held-out
# regime can never be named, and the question is whether the probe's uncertainty rises on it.
REGIMES = {'LG': 'level', 'RA': 'ascend', 'SA': 'ascend', 'RD': 'descend', 'SD': 'descend',
           'ST': 'still', 'TR': 'transition'}
# (split, kind) pairs every stage iterates over, always in this order and always in the
# builder's row order, so arrays from different stages align row for row.
SPLITS = (('train', 'id'), ('val', 'id'), ('val', 'ood'), ('test', 'id'), ('test', 'ood'))


def run_dir(holdout: str, stage: str, root: Path | str = RUNS) -> Path:
    return Path(root) / holdout / stage


__all__ = ['HERE', 'ROOT', 'RUNS', 'RESULTS', 'REGIMES', 'SPLITS', 'run_dir']
