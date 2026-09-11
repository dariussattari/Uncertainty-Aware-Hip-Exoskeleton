"""Model registry — one name maps to everything needed to train, score and plot a model.

This is the seam that keeps ``main.py`` model-agnostic. Adding an architecture means adding a
:class:`ModelSpec` here and the three modules it points at; the CLI, the checkpoint layout and
the metric set are then inherited for free.

    python main.py train --model autoencoder
    python main.py eval  --model ensemble
    python main.py all   --model autoencoder

Entries are declared lazily — each field is a callable returning the real object — so importing
the registry does not import torch, sklearn and every model at once. That matters for
``main.py --help``, and it means a half-finished model cannot break the CLI for the others.

The four architectures of the paper's Table I: the gait-phase ensemble and the autoencoder are
implemented; the synthetic-target ensemble and the GAN are declared but not yet built, and
``available()`` reports which is which rather than failing at import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["ModelSpec", "REGISTRY", "get", "names", "available", "describe"]


@dataclass(frozen=True)
class ModelSpec:
    """Everything ``main.py`` needs to drive one architecture.

    Every field except ``name``/``description``/``default_run`` is a zero-argument callable so
    that nothing heavy is imported until the model is actually selected.
    """

    name: str
    description: str
    default_run: str
    paper_row: str                      # which row of Table I this reproduces
    data: Callable[[], Any] | None = None          # -> data class
    config: Callable[[], Any] | None = None        # -> config dataclass
    train: Callable[[], Any] | None = None         # -> train_paper_protocol(data, cfg)
    evaluate: Callable[[], Any] | None = None      # -> evaluate(run, data, split, ...)
    figures: Callable[[], Any] | None = None       # -> run(out, split, batch_size, device, reuse)
    audit: Callable[[], Any] | None = None         # -> fidelity audit for this model
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def implemented(self) -> bool:
        return self.train is not None

    def load(self) -> dict[str, Any]:
        """Resolve every callable. Raises if the model is not implemented yet."""
        if not self.implemented:
            raise NotImplementedError(
                f"{self.name!r} is declared but not implemented yet. "
                f"Implemented: {', '.join(available())}")
        return {k: (v() if callable(v) else v)
                for k, v in (("data", self.data), ("config", self.config),
                             ("train", self.train), ("evaluate", self.evaluate),
                             ("figures", self.figures), ("audit", self.audit))
                if v is not None}


# --- lazy accessors ---------------------------------------------------------------
# Each is a function so that `import registry` stays cheap and a broken model module
# cannot take the whole CLI down with it.

def _ensemble_data():
    from dataset import EnsembleGaitPhase
    return EnsembleGaitPhase


def _ensemble_config():
    from training.ensemble import TrainConfig
    return TrainConfig


def _ensemble_train():
    from training.ensemble import train_paper_protocol
    return train_paper_protocol


def _ensemble_eval():
    from evaluation.ensemble import evaluate_model
    return evaluate_model


def _ensemble_figures():
    from evaluation.plots_ensemble import run
    return run


def _ensemble_audit():
    from paper_spec import audit
    return audit


def _ae_data():
    from dataset import AutoencoderData
    return AutoencoderData


def _ae_config():
    from training.autoencoder import AeConfig
    return AeConfig


def _ae_train():
    from training.autoencoder import train_paper_protocol
    return train_paper_protocol


def _ae_eval():
    from evaluation.autoencoder import evaluate_autoencoder
    return evaluate_autoencoder


def _ae_figures():
    from evaluation.plots_autoencoder import run
    return run


def _ae_audit():
    from paper_spec import audit_ae
    return audit_ae


REGISTRY: dict[str, ModelSpec] = {
    "ensemble": ModelSpec(
        name="ensemble",
        description="Ensemble of 7 gait-phase TCNs; Psi = variance across branches",
        default_run="ml/vanilla/runs/paper",
        paper_row="Ensemble Method (Gait Phase) — their best performer, F1 90.3",
        data=_ensemble_data, config=_ensemble_config, train=_ensemble_train,
        evaluate=_ensemble_eval, figures=_ensemble_figures, audit=_ensemble_audit,
        notes=("needs gait-phase labels, so reads data/processed/",
               "two-stage: LOSO for the epoch count, then retrain on all subjects",
               "the long one — roughly 8 h for the full protocol"),
    ),
    "autoencoder": ModelSpec(
        name="autoencoder",
        description="Conv autoencoder; Psi = LOF on the latent space",
        default_run="ml/vanilla/runs/autoencoder",
        paper_row="Autoencoder — F1 55.7",
        data=_ae_data, config=_ae_config, train=_ae_train,
        evaluate=_ae_eval, figures=_ae_figures, audit=_ae_audit,
        notes=("label-free, so reads data/processed/AE_GAN/ (19k more windows)",
               "single stage with an 80/20 participant split, patience 5",
               "score is LOF on the latent, NOT reconstruction error",
               "the paper's lr of 0.064 collapses this model; default is 0.003"),
    ),
    "synthetic": ModelSpec(
        name="synthetic",
        description="Ensemble of 7 TCNs predicting summed pairwise channel correlations",
        default_run="ml/vanilla/runs/synthetic",
        paper_row="Ensemble Method (Synthetic Target) — F1 62.5",
        notes=("not implemented yet",
               "label-free: the target is computed from the input window itself",
               "reuses the ensemble architecture with one linear output instead of two tanh"),
    ),
    "gan": ModelSpec(
        name="gan",
        description="TCN GAN; Psi = 1 - D(x) from the discriminator",
        default_run="ml/vanilla/runs/gan",
        paper_row="GAN — F1 71.5",
        notes=("not implemented yet",
               "label-free, reads data/processed/AE_GAN/",
               "500 fixed epochs at batch 256, 5 generator updates per discriminator update",
               "the expensive one, and the only model likely to be compute-bound"),
    ),
}


def names() -> list[str]:
    """Every registered name, implemented or not."""
    return list(REGISTRY)


def available() -> list[str]:
    """Names that can actually be run."""
    return [n for n, s in REGISTRY.items() if s.implemented]


def get(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}. Registered: {', '.join(names())}")
    return REGISTRY[name]


def describe() -> str:
    """A table of what is registered — used by ``main.py models``."""
    w = max(len(n) for n in names())
    lines = [f"  {'model'.ljust(w)}  {'status':<16s} reproduces",
             f"  {'-'*w}  {'-'*16} {'-'*46}"]
    for n, s in REGISTRY.items():
        status = "implemented" if s.implemented else "not implemented"
        lines.append(f"  {n.ljust(w)}  {status:<16s} {s.paper_row}")
    for n, s in REGISTRY.items():
        lines.append(f"\n  {n}: {s.description}")
        lines.append(f"    run dir: {s.default_run}")
        for note in s.notes:
            lines.append(f"    - {note}")
    return "\n".join(lines)
