"""Training for the label-free ensembles — Experiments 4, 5 and 6.

All three are the Experiment 1 ensemble with a different target and a linear output head, so
this module is deliberately thin: the architecture, the two-stage LOSO protocol, the threshold
rule and the evaluator are all reused unchanged. Only the target differs, and the targets live
in ``targets.py`` so the screening notebook and this file cannot drift apart.

    python ml/vanilla/main.py train --model synthetic        # Exp 4: summed correlation
    python ml/vanilla/main.py train --model forecast-angle   # Exp 5: hip angle, 200 ms ahead
    python ml/vanilla/main.py train --model forecast-all     # Exp 6: 16 channels, 200 ms ahead

Why these targets exist at all: gait phase is recovered from force-plate data offline, so the
reference work's best estimator cannot be run live. Every target here is computable from the
sixteen sensor channels alone.

**Score.** The uncertainty score stays the branch variance, identical in definition to
Experiment 1, so all six models sit on one comparison table under one threshold rule. Realised
prediction error is *not* the score. Measured on these targets it reaches AUROC 0.32 — below
chance, anti-correlated with novelty, matching the autoencoder's reconstruction error at 0.312
— because standing is the easiest task to predict and the most out-of-distribution. Prediction
error ranks windows by signal complexity, not by novelty.

Training records nothing but the branch variance. The realised error and the Krogh-Vedelsby
decomposition are computed **after** training by ``evaluation/ambiguity.py``, which needs only
the checkpoint and the target -- so no training run has to be repeated to obtain them.

**Horizon.** 40 samples (200 ms) by default, and this is not a free choice. A two-tap linear
filter predicts hip angle 5 ms ahead with 0.99 skill and 50 ms ahead with 0.83, so at those
horizons every branch learns the same near-exact extrapolator and the variance collapses to
zero everywhere — including out of distribution. Skill crosses zero near 40 samples, which is
where structure must genuinely be learned. See ``targets.forecast_skill`` and
``05_synthetic_targets.ipynb`` Part 2.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset import SyntheticEnsembleData
from models.ensemble import create_ensemble
from targets import DEFAULT_HORIZON
from training.common import pick_device
from training.ensemble import TrainConfig, fit_threshold, run_loso
from training.ensemble import train_paper_protocol as _ensemble_protocol

__all__ = ["SyntheticConfig", "CorrelationConfig", "ForecastAngleConfig",
           "ForecastAllConfig", "CONFIGS", "make_model", "train_paper_protocol",
           "DEFAULTS", "describe_target"]


@dataclass
class SyntheticConfig(TrainConfig):
    """Experiment 1's configuration plus the two fields that select the target.

    Every hyperparameter is inherited from Table IV's ensemble row and left untouched: the
    point of these experiments is to change the target and nothing else, so any difference in
    result is attributable to the target rather than to the optimiser.
    """

    target: str = "correlation"
    horizon: int = DEFAULT_HORIZON
    out_dir: str = "ml/vanilla/runs/synthetic"


@dataclass
class CorrelationConfig(SyntheticConfig):
    """Experiment 4 — the reference work's own synthetic target."""

    target: str = "correlation"
    out_dir: str = "ml/vanilla/runs/synthetic"


@dataclass
class ForecastAngleConfig(SyntheticConfig):
    """Experiment 5 — hip angle, both legs, ``horizon`` samples ahead."""

    target: str = "forecast_angle"
    out_dir: str = "ml/vanilla/runs/forecast_angle"


@dataclass
class ForecastAllConfig(SyntheticConfig):
    """Experiment 6 — all sixteen channels, ``horizon`` samples ahead."""

    target: str = "forecast_all"
    out_dir: str = "ml/vanilla/runs/forecast_all"


CONFIGS = {"correlation": CorrelationConfig,
           "forecast_angle": ForecastAngleConfig,
           "forecast_all": ForecastAllConfig}

DEFAULTS = {
    "correlation": dict(
        run="ml/vanilla/runs/synthetic",
        experiment="Experiment 4",
        blurb="summed pairwise channel correlation (the reference work's own target)"),
    "forecast_angle": dict(
        run="ml/vanilla/runs/forecast_angle",
        experiment="Experiment 5",
        blurb=f"hip angle, both legs, {DEFAULT_HORIZON} samples "
              f"({1000*DEFAULT_HORIZON/200:.0f} ms) ahead"),
    "forecast_all": dict(
        run="ml/vanilla/runs/forecast_all",
        experiment="Experiment 6",
        blurb=f"all 16 channels, {DEFAULT_HORIZON} samples "
              f"({1000*DEFAULT_HORIZON/200:.0f} ms) ahead"),
}


def describe_target(target: str) -> str:
    d = DEFAULTS[target]
    return f"{d['experiment']}: {d['blurb']}"


def make_model(data, cfg):
    """A linear-headed ensemble. Everything else is Experiment 1's architecture.

    The tanh head of Experiment 1 exists because ``sin(gait phase)`` lives in [-1, 1]. These
    targets are standardized and unbounded, so a tanh would saturate and clip precisely the
    tails that carry information.
    """
    return create_ensemble(data.n_channels, data.n_targets, cfg.n_members,
                           activation="linear")


def train_paper_protocol(data: SyntheticEnsembleData, cfg: SyntheticConfig | None = None):
    """Two-stage LOSO then retrain, exactly as Experiment 1, with the linear-headed model."""
    cfg = cfg or SyntheticConfig()
    if getattr(data, "target", None) != cfg.target:
        raise ValueError(
            f"config asks for target {cfg.target!r} but the data was built with "
            f"{getattr(data, 'target', None)!r}. Build it as "
            f"SyntheticEnsembleData(target={cfg.target!r}, horizon={cfg.horizon}).")

    print(f"{describe_target(cfg.target)}")
    print(f"  {data}")
    print(f"  {cfg.n_members} branches, linear head, {data.n_targets} output(s)")
    if cfg.target != "correlation":
        print(f"  horizon {cfg.horizon} samples ({1000*cfg.horizon/200:.0f} ms); "
              f"targets: {', '.join(data.target_names[:4])}"
              f"{' ...' if len(data.target_names) > 4 else ''}")
    else:
        mu, sd = data.target_stats
        print(f"  target standardized with training statistics: mean {mu:+.3f}, sd {sd:.3f}")
    print(f"  score: branch variance (NOT prediction error -- see the module docstring)\n")

    return _ensemble_protocol(data, cfg, make_model)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train a label-free ensemble (Experiments 4-6).")
    p.add_argument("--target", default="correlation", choices=tuple(DEFAULTS))
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                   help="samples; must be a multiple of the window stride (10)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--folds", type=int, default=None, help="limit LOSO folds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="default: per-target")
    p.add_argument("--smoke", action="store_true", help="2 folds x 2 epochs, wiring only")
    a = p.parse_args()

    cfg = SyntheticConfig(
        target=a.target, horizon=a.horizon, lr=a.lr, batch_size=a.batch_size,
        max_epochs=a.max_epochs, patience=a.patience, loso_folds=a.folds, seed=a.seed,
        device=a.device, out_dir=a.out or DEFAULTS[a.target]["run"])
    if a.smoke:
        cfg.max_epochs, cfg.patience, cfg.loso_folds = 2, 1, 2
        cfg.out_dir = f"ml/vanilla/runs/{a.target}_smoke"

    data = SyntheticEnsembleData(batch_size=cfg.batch_size, target=cfg.target,
                                 horizon=cfg.horizon)
    print(f"device: {pick_device(cfg.device)}\n")
    train_paper_protocol(data, cfg)
