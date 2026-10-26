"""Joint moment estimator TCN of Molinaro, Scherpereel et al., *Task-agnostic exoskeleton control via biological joint
moment estimation*, Nature 635 (2024), the OOD-safe fallback controller of this project.

The network is the TCN of Bai et al. (2018), as the paper states, at the values Extended Data Table 1 selected:

    residual blocks       5, two causal convolutions each
    filters per layer     80
    kernel size           5
    activation            ReLU
    block normalization   weight normalization
    dropout               spatial (whole channels), p = 0.15
    output                two heads at the last time step: total hip and knee moment (N m/kg)

Unstated in the paper and resolved here: the dilations double per block, (1, 2, 4, 8, 16), as in Bai et al. That gives
a receptive field of 1 + 2 (k - 1) sum(d) = 249 samples (1.245 s at 200 Hz), consistent with the table's note that
combinations needing an input sequence longer than 250 were omitted, so the input window is the receptive field.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

__all__ = ['MomentTCN', 'get_receptive_field']

# Extended Data Table 1, "Selected Value"
N_FILTERS = 80
N_BLOCKS = 5
KERNEL_SIZE = 5
DROPOUT = 0.15
N_OUTPUTS = 2                                    # total hip and knee flexion/extension moment


def get_receptive_field(n_blocks: int = N_BLOCKS, kernel_size: int = KERNEL_SIZE) -> int:
    """Samples one output sees: two convolutions per block at dilation 2^i."""
    return 1 + 2 * (kernel_size - 1) * (2 ** n_blocks - 1)


class TemporalBlock(nn.Module):
    """Bai et al.'s residual block: two weight-normalized causal dilated convolutions, ReLU, spatial dropout, and a
    1x1 convolution on the skip path when the channel count changes."""

    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation                                # left padding only: causal
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel_size, dilation = dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel_size, dilation = dilation))
        self.drop = nn.Dropout1d(dropout)
        self.skip = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        for conv in (self.conv1, self.conv2):                                  # Bai et al.'s init
            nn.init.normal_(conv.parametrizations.weight.original1, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(torch.relu(self.conv1(nn.functional.pad(x, (self.pad, 0)))))   # (B, C, T) -> (B, F, T)
        h = self.drop(torch.relu(self.conv2(nn.functional.pad(h, (self.pad, 0)))))
        return torch.relu(h + self.skip(x))


class MomentTCN(nn.Module):
    """(B, C, T) sensor window -> (B, 2) total hip and knee moment at its last sample."""

    def __init__(self, n_inputs: int, n_filters: int = N_FILTERS, n_blocks: int = N_BLOCKS, kernel_size: int = KERNEL_SIZE,
                 dropout: float = DROPOUT, n_outputs: int = N_OUTPUTS) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[TemporalBlock(n_inputs if i == 0 else n_filters, n_filters, kernel_size, 2 ** i, dropout)
                                      for i in range(n_blocks)])
        self.head = nn.Linear(n_filters, n_outputs)
        self.receptive_field = get_receptive_field(n_blocks, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x)[:, :, -1])                             # (B, F, T) -> (B, F) -> (B, 2)
