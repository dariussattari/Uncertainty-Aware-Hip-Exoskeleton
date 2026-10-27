"""The joint-moment estimator as the device sees it, in PyTorch only (no TensorRT): load a run, its standardization
folded in, and the per-leg input conventions. Shared by moment_engine.py (Jetson) and the replay tools (any machine)."""
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn.utils import parametrize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'ml' / 'common'))
from models.moment_tcn import MomentTCN  # noqa: E402

MIRRORED = ('accel_z', 'gyro_x', 'gyro_y', 'cop_z')            # channel suffixes negated on the left leg (moment_data.py)
HIP_DELAY_SAMPLES = 10                                       # the hip output estimates the moment 50 ms behind the window


class Standardized(nn.Module):
    """The estimator with its training standardization in front: (B, C, T) raw channels -> (B, J) N m/kg."""

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer('mean', mean.float().view(1, -1, 1))                # (1, C, 1)
        self.register_buffer('std', std.float().view(1, -1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def get_unit(channel: str) -> str:
    """Unit of a Nature input channel, from its name."""
    units = {'(causal derivative)': 'deg/s', 'angle': 'deg', 'accel': 'm/s^2', 'gyro': 'deg/s', 'force': 'as published', 'cop': 'as published'}
    return next((u for key, u in units.items() if key in channel), 'unknown')


def get_left_sign(channels: list[str]) -> npt.NDArray[np.float32]:
    """Per channel, the factor that mirrors the left leg into the right leg's frame: -1 on MIRRORED channels. (C,)."""
    return np.array([-1.0 if c.endswith(MIRRORED) else 1.0 for c in channels], dtype = np.float32)


def load_run(run: Path) -> tuple[Standardized, dict]:
    """The run's best.pt as an eval-mode Standardized model with weight norm folded into plain weights, and the
    checkpoint's metadata."""
    ckpt = torch.load(run / 'best.pt', map_location = 'cpu', weights_only = True)
    model = MomentTCN(len(ckpt['channels']), n_outputs = len(ckpt['joints']))
    model.load_state_dict(ckpt['model'])
    for module in model.modules():
        if parametrize.is_parametrized(module, 'weight'):
            parametrize.remove_parametrizations(module, 'weight', leave_parametrized = True)
    meta = {k: ckpt[k] for k in ('channels', 'joints', 'epoch', 'config')}
    return Standardized(model, ckpt['mean'], ckpt['std']).eval(), meta
