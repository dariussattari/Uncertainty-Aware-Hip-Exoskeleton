"""Locate the repository and expose the ``ml/vanilla`` and ``ml/LOTO`` modules this package reuses.

Like ``ml/prob_ensemble``, this package forks nothing: the encoder trunk and decoder come from
``ml/vanilla/models/autoencoder.py``, the data objects from ``ml/vanilla/dataset.py`` and
``ml/LOTO/splits.py``, and the smoothing, threshold rule and metrics from
``ml/vanilla/evaluation/common.py``. Only the Gaussian latent heads, the ELBO terms, the
training loop and the result layout are defined here.

Importing this module has no side effects beyond ``sys.path``; in particular it does not
import IPython or matplotlib, so every module here runs inside a SLURM job.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'ml/vanilla/registry.py').is_file())
RUNS = HERE / 'runs'                      # <protocol>_<variant>_seed<seed>/ (ignored by git)

for _folder in ('ml/vanilla', 'ml/LOTO'):
    if str(ROOT / _folder) not in sys.path:
        sys.path.append(str(ROOT / _folder))

__all__ = ['HERE', 'ROOT', 'RUNS']
