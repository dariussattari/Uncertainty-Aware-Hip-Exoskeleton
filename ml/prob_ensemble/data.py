"""Data for the probabilistic ensemble — Experiment 5's, unchanged.

Deliberately a thin wrapper. ``SyntheticEnsembleData`` from ``ml/vanilla`` already produces
exactly the windows, targets, masks, participant-disjoint splits and leave-one-subject-out
folds that Experiment 5 trained on, and reusing it is what makes the comparison attributable:
the only difference between Experiment 5 and this model must be the loss and the score.

The single thing this adds is the data root, which on a cluster is usually not next to the
code. See ``paths.data_root``.
"""

from __future__ import annotations

import paths

from dataset import SyntheticEnsembleData

__all__ = ["TARGET", "HORIZON", "build", "N_TRAIN_SUBJECTS"]

# Fixed, not configurable by accident. Experiment 5's target and horizon; changing either
# breaks the comparison this experiment exists to make.
TARGET = "forecast_angle"
HORIZON = 40                 # samples; 200 ms at 200 Hz
N_TRAIN_SUBJECTS = 12


def build(batch_size: int = 1024, scale: bool = True) -> SyntheticEnsembleData:
    """Experiment 5's dataset, reading from ``paths.data_root()``."""
    return SyntheticEnsembleData(root=paths.data_root(), batch_size=batch_size,
                                 scale=scale, target=TARGET, horizon=HORIZON)


def fold_subjects(data: SyntheticEnsembleData) -> list[str]:
    """The leave-one-subject-out fold order, as a stable list.

    A SLURM job array indexes into this, so the order must not depend on anything that varies
    between tasks. ``Split.subjects`` is sorted, so it does not.
    """
    return list(data.splits["train"].subjects)


if __name__ == "__main__":  # python ml/prob_ensemble/data.py
    print(paths.describe(), "\n")
    d = build(batch_size=256)
    print(d)
    subs = fold_subjects(d)
    print(f"\n{len(subs)} LOSO folds: {subs}")
    x, y, m = next(iter(d.loader("train", batch_size=256)))
    print(f"\nbatch: x{tuple(x.shape)} y{tuple(y.shape)} mask{tuple(m.shape)}")
    print(f"  targets     : {d.target_names}")
    print(f"  y mean {y.mean():+.3f} sd {y.std():.3f} | mask {m.mean():.4f}")
    print(f"  n_targets={d.n_targets}  n_channels={d.n_channels}  window={d.window}")
