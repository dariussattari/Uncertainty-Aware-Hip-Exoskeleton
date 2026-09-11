"""Helpers shared by every training loop: device selection and run-directory paths.

Kept separate from the model-specific trainers so that adding a model does not mean
importing from another model's module.
"""

from __future__ import annotations

from pathlib import Path

import torch

from dataset import repo_root

__all__ = ["pick_device", "resolve_out"]


def resolve_out(path: str | Path) -> Path:
    """Anchor a run directory to the repo root.

    Relative paths would otherwise land wherever the process happened to start, so running
    from ml/vanilla/ would create ml/vanilla/ml/vanilla/runs/.
    """
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def pick_device(prefer: str | None = None) -> torch.device:
    """CUDA, else Apple MPS, else CPU.

    The MPS branch is not optional on this machine: it is roughly 7x faster than the CPU
    path for these models, and a plain ``cuda if available else cpu`` check silently lands
    on CPU on Apple silicon.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
