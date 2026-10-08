"""Locate the repository and expose the ``ml/sci_rob/vanilla`` and ``ml/sci_rob/LOTO`` modules this package reuses.

Like ``ml/sci_rob/prob_ensemble``, this package forks nothing: the encoder trunk and decoder come from
``ml/common/models/autoencoder.py``, the data objects from ``ml/sci_rob/vanilla/dataset.py`` and
``ml/sci_rob/LOTO/splits.py``, and the smoothing, threshold rule and metrics from
``ml/sci_rob/vanilla/evaluation/common.py``. Only the Gaussian latent heads, the ELBO terms, the
training loop and the result layout are defined here.

Importing this module has no side effects beyond ``sys.path``; in particular it does not
import IPython or matplotlib, so every module here runs inside a SLURM job.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'ml/sci_rob/vanilla/registry.py').is_file())
RUNS = HERE / 'runs'                      # <protocol>_<variant>_seed<seed>/ (ignored by git)

for _folder in ('ml/common', 'ml/sci_rob/vanilla', 'ml/sci_rob/LOTO'):
    if str(ROOT / _folder) not in sys.path:
        sys.path.append(str(ROOT / _folder))

__all__ = ['HERE', 'ROOT', 'RUNS']
