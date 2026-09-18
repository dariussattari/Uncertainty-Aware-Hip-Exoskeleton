"""Locate the repo, the data and the ``ml/vanilla`` modules this package reuses.

``ml/vanilla`` is the faithful reproduction of Tourk et al. — its hyperparameters are the
paper's. This package is the first deliberate departure, so it lives in its own directory. It
does **not** fork the data pipeline or the evaluation protocol: those are imported from
``ml/vanilla`` unchanged, because the whole value of the comparison rests on Experiment 5 and
this model seeing identical windows, identical splits, identical filtering and an identical
threshold rule. The only things defined here are the probabilistic head, the loss and the
uncertainty decomposition.

Data location is resolved in this order:

1. ``$HIPEXO_DATA`` — a directory containing ``AE_GAN/`` and ``ood/``. Set this on a cluster,
   where the arrays usually do not live next to the code.
2. ``<repo>/data/processed`` — the local layout.

The processed arrays total about 4.3 GB (``AE_GAN`` 3.7 GB + ``ood`` 567 MB), all of it
memory-mapped during training, so where they sit matters for throughput. See ``slurm/env.sh``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["REPO", "VANILLA", "data_root", "describe"]


def _find_repo() -> Path:
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / "ml" / "vanilla").is_dir():
            return d
    raise FileNotFoundError(f"could not locate the repo root above {here}")


REPO = _find_repo()
VANILLA = REPO / "ml" / "vanilla"

# ml/vanilla's modules are flat and import each other by bare name, so its directory has to be
# on the path rather than imported as a package.
if str(VANILLA) not in sys.path:
    sys.path.insert(0, str(VANILLA))


def data_root() -> Path:
    """Directory holding ``AE_GAN/`` and ``ood/``."""
    env = os.environ.get("HIPEXO_DATA")
    if env:
        p = Path(env).expanduser().resolve()
        missing = [n for n in ("AE_GAN", "ood") if not (p / n).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"$HIPEXO_DATA={p} is missing {missing}. It should be the directory that "
                f"contains AE_GAN/ and ood/, i.e. the equivalent of <repo>/data/processed.")
        return p
    local = REPO / "data" / "processed"
    if not (local / "AE_GAN").is_dir():
        raise FileNotFoundError(
            f"no data at {local}/AE_GAN and $HIPEXO_DATA is unset. Either run "
            f"data_exploration/02_build_windows.ipynb, or point $HIPEXO_DATA at the copy you "
            f"staged on the cluster.")
    return local


def describe() -> str:
    d = data_root()
    src = "$HIPEXO_DATA" if os.environ.get("HIPEXO_DATA") else "repo-local"
    return f"repo {REPO}\nvanilla {VANILLA}\ndata {d}  ({src})"


if __name__ == "__main__":
    print(describe())
