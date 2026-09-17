"""Label-free targets for the ensemble — one definition, shared by the notebook and training.

The gait-phase ensemble of Experiment 1 needs a target that only exists offline: gait phase is
recovered from force-plate data, so it cannot be computed while the exoskeleton is running. The
targets here are all computable from the sixteen input channels alone, which is what makes them
usable at run time and what the reference work means by a *synthetic* target.

    from targets import summed_pairwise_corr, forecast_rows

Three are provided:

``summed_pairwise_corr``
    The reference work's own choice — the sum of the 120 pairwise Pearson correlations between
    channels over the window. Reproduces the "Ensemble Method (Synthetic Target)" row of
    Table I.
``forecast_rows`` with ``ANGLE_CHANNELS``
    Hip angle for both legs, ``horizon`` samples after the window ends.
``forecast_rows`` with every channel
    All sixteen channels at the same horizon.

**Why this file exists separately.** A target computed one way in the screening notebook and
another way in training is a silent, expensive bug — the notebook would clear a target the
model never sees. Both import from here.

Two properties every candidate is screened on in ``05_synthetic_targets.ipynb``, both of them
lessons paid for by earlier experiments:

1. **Scale invariance.** The GAN's score turned out to be a monotone function of window
   amplitude (rank correlation -0.86) and was beaten by raw amplitude alone. A target that is
   mostly a measure of how loud the window is will reproduce that.
2. **Difficulty.** Ensemble disagreement is driven by how hard the target is. A target a
   two-tap linear filter can solve leaves all seven branches agreeing everywhere, including
   out-of-distribution, and the variance carries nothing. See ``forecast_skill``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["STRIDE", "MIN_SD", "N_PAIRS", "DEFAULT_HORIZON", "ANGLE_CHANNELS",
           "summed_pairwise_corr", "correlation_matrix", "degenerate_fraction",
           "forecast_rows", "forecast_skill", "channel_indices"]

STRIDE = 10                 # windows on disk are cut at this stride (02_build_windows.ipynb)
MIN_SD = 0.01               # standard-deviation floor for the correlation denominator
N_PAIRS = 120               # C(16, 2)
DEFAULT_HORIZON = 40        # samples; 200 ms at 200 Hz. See forecast_skill for why not 1.
ANGLE_CHANNELS = ("enc_angle_l", "enc_angle_r")


def channel_indices(channels: list[str], want) -> np.ndarray:
    """Positions of ``want`` within ``channels``, in the order given."""
    return np.array([channels.index(c) for c in want], dtype=np.int64)


# ------------------------------------------------------------------ target 1: correlations

def correlation_matrix(X: np.ndarray, min_sd: float = MIN_SD) -> np.ndarray:
    """Per-window channel correlation matrices, ``(n, c, c)``.

    The standard-deviation floor is what makes this defined for near-constant windows. Pearson
    correlation divides by the product of the two channels' standard deviations, and for a
    motionless task that product goes to zero — standing has 89% of its channels below 0.01 in
    standardized units, so the unfloored correlation is the correlation of sensor noise, a
    different random number every window.

    Flooring the *denominator* rather than dropping the pair shrinks such pairs smoothly
    toward zero instead of introducing a discontinuity: with a true sd of 0.005 against a floor
    of 0.01, the reported correlation is a quarter of the true one. Zero is also the honest
    value for a constant signal — it has no linear relationship with anything.
    """
    Xc = X - X.mean(axis=2, keepdims=True)
    sd = Xc.std(axis=2)
    cov = np.einsum("nct,ndt->ncd", Xc, Xc) / X.shape[2]
    floored = np.maximum(sd, min_sd)
    return cov / (floored[:, :, None] * floored[:, None, :])


def summed_pairwise_corr(X: np.ndarray, min_sd: float = MIN_SD) -> np.ndarray:
    """The reference work's synthetic target: sum over the upper triangle, ``(n,)``.

    ``X`` is ``(n, channels, time)`` and standardized. With 16 channels this sums 120 pairs.
    Scale-invariant by construction, which is its main virtue — measured rank correlation with
    window amplitude is -0.03, against the GAN score's -0.86.
    """
    R = correlation_matrix(X, min_sd)
    iu = np.triu_indices(X.shape[1], k=1)
    return R[:, iu[0], iu[1]].sum(axis=1).astype(np.float32)


def degenerate_fraction(X: np.ndarray, min_sd: float = MIN_SD) -> np.ndarray:
    """Fraction of channels per window below the sd floor — the conditioning diagnostic."""
    return ((X - X.mean(axis=2, keepdims=True)).std(axis=2) < min_sd).mean(axis=1)


# ------------------------------------------------------------------ targets 2 and 3: forecast

def forecast_rows(meta: pd.DataFrame, horizon: int = DEFAULT_HORIZON,
                  stride: int = STRIDE) -> np.ndarray:
    """For each window, which **row** holds its future target. ``-1`` where none does.

    No re-windowing is needed to build a forecast target. The stored windows are cut at a fixed
    stride from continuous trials, so row ``i + horizon/stride`` is the same trial advanced by
    exactly ``horizon`` samples, and its final sample is the value ``horizon`` steps after row
    ``i`` ends. Verified bit-exact on the training split:
    ``max |window_i[40:200] - window_(i+4)[0:160]| = 0``.

    The target for row ``i`` is therefore ``X[forecast_rows(meta)[i], channel, -1]``.

    Rows near the end of a trial have no partner and are returned as ``-1``; at a horizon of 40
    that is 0.56% of the training split. They must be dropped from the loss, not zero-filled —
    a zero in a standardized target reads as the channel mean, which is a plausible value and
    not a missing one. This is the same trap the gait-phase mask exists to avoid.
    """
    if horizon % stride:
        raise ValueError(
            f"horizon {horizon} must be a multiple of the window stride {stride}; "
            f"the target is read from a later window row, so only multiples are addressable. "
            f"Nearest valid: {stride * round(horizon / stride)}")
    step = horizon // stride
    n = len(meta)
    out = np.arange(step, n + step, dtype=np.int64)

    # a partner is valid only if it is in range and belongs to the same trial
    trial = (meta["subject"].astype(str) + "/" + meta["trial"].astype(str)).to_numpy()
    ok = out < n
    same = np.zeros(n, dtype=bool)
    same[ok] = trial[out[ok]] == trial[ok]

    # and only if it really is `horizon` samples later -- end_idx is authoritative
    if "end_idx" in meta.columns:
        e = meta["end_idx"].to_numpy()
        contiguous = np.zeros(n, dtype=bool)
        contiguous[ok] = (e[out[ok]] - e[ok]) == horizon
        same &= contiguous

    out[~same] = -1
    return out


def forecast_skill(X: np.ndarray, channel: int, horizon: int) -> dict:
    """How much of the forecast a two-tap linear filter already gets — the difficulty floor.

    Run this before choosing a horizon. The prediction skill of persistence and of linear
    extrapolation bounds below what any trained model must beat, and if that floor is already
    near-perfect the ensemble has nothing to disagree about. Measured on hip angle:

        horizon        1 (5 ms)    10 (50 ms)   20 (100 ms)   40 (200 ms)
        linear RMSE      0.011        0.156         0.432         1.107
        skill            0.989        0.844         0.568        -0.108

    At one step ahead a two-tap filter reaches 98.9% skill, so every branch learns the same
    near-exact extrapolator and the variance collapses. Negative skill at 40 means the naive
    baselines have stopped working and structure must actually be learned — the same difficulty
    class as gait phase, which is why 40 is the default.

    ``skill`` is ``1 - RMSE / sd(target)``: 0 means no better than predicting the mean, 1 means
    exact. Uses within-window samples, so it needs no row join.
    """
    t0 = X.shape[2] - 1 - horizon
    if t0 < 1:
        raise ValueError(f"horizon {horizon} needs a window longer than {X.shape[2]}")
    cur, prev, true = X[:, channel, t0], X[:, channel, t0 - 1], X[:, channel, -1]
    per = float(np.sqrt(((true - cur) ** 2).mean()))
    lin = float(np.sqrt(((true - (cur + horizon * (cur - prev))) ** 2).mean()))
    sd = float(true.std())
    return {"horizon": horizon, "ms": 1000 * horizon / 200, "persistence_rmse": per,
            "linear_rmse": lin, "target_sd": sd, "skill": 1 - min(per, lin) / sd}


if __name__ == "__main__":  # python ml/vanilla/targets.py
    from dataset import AutoencoderData

    data = AutoencoderData(batch_size=1024)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(data.test), 6000, replace=False))
    X = data.test.tensors(idx).numpy()
    meta = data.test.meta.iloc[idx].reset_index(drop=True)

    t = summed_pairwise_corr(X)
    print(f"summed_pairwise_corr: range [{t.min():+.2f}, {t.max():+.2f}], sd {t.std():.2f}")
    print(f"  degenerate-channel fraction: mean {degenerate_fraction(X).mean():.3f}, "
          f"max {degenerate_fraction(X).max():.3f}")

    print("\nforecast_skill on hip angle (left):")
    ch = data.channels.index("enc_angle_l")
    for h in (10, 20, 40):
        s = forecast_skill(X, ch, h)
        print(f"  h={h:>3} ({s['ms']:>3.0f} ms)  linear RMSE {s['linear_rmse']:.4f}  "
              f"skill {s['skill']:+.3f}")

    rows = forecast_rows(data.test.meta, DEFAULT_HORIZON)
    print(f"\nforecast_rows(h={DEFAULT_HORIZON}) on the test split: "
          f"{(rows >= 0).mean():.4%} of {len(rows):,} windows have a target")
