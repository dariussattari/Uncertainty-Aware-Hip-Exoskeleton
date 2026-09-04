"""Ensemble of gait-phase TCNs — the ankle paper's best-performing uncertainty estimator.

Reproduces the "Ensemble Method (Gait Phase)" row of Table I in Tourk et al.,
*Uncertainty-Aware Ankle Exoskeleton Control* (arXiv:2508.21221), applied to hip data.

The idea: train several TCNs on in-distribution data only. On familiar movement they agree;
on unfamiliar movement they diverge, because nothing in training constrained them there. The
*disagreement* is the uncertainty signal — never the prediction error, which needs a label and
so cannot be computed at run time.

    Psi(x) = ( Var(l_1..l_7) + Var(r_1..r_7) ) / 2                       [paper, eq. 2]

Specifications from the paper's Table IV ("Ensemble Models" / "Ensemble: Gait Phase Model"):

    branches            7 (more did not improve performance)
    layers per branch   3
    filters per layer   30
    kernel size         20
    activation / norm   ReLU, BatchNorm1d after each convolution
    output              two heads (left / right leg), tanh -> [-1, 1]
    loss                MSE (masked here; see dataset.EnsembleGaitPhase.masked_mse)
    anomaly scoring     variance across branch predictions, 99.5th percentile threshold

Each branch predicts the *sine* of gait phase, which is why the head is a tanh: the paper
predicts sin(phase) rather than phase to avoid the discontinuity at heel strike where phase
wraps from 100% to 0%. The targets built by ``02_build_windows.ipynb`` already carry that
transform.

Two choices the paper leaves unstated, resolved here and flagged so they are easy to revisit:

* **Dilations.** Table IV gives 3 layers of kernel 20 but no dilation factors. ``(1, 2, 8)``
  gives a receptive field of 210 samples, which covers the whole 200-sample (1 s) input
  window. Conventional doubling ``(1, 2, 4)`` would reach only 134 samples, leaving the first
  third of every window unreachable — the model would be paying to load context it cannot see.
* **Ensemble diversity** comes from random initialisation alone, which is what the paper's
  "multi-branch convolutional predictor" description implies.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CausalConv1d", "TCNBlock", "TCNEnsembleMember", "GaitPhaseEnsemble", "create_ensemble"]

# Table IV, "Ensemble Models"
N_MEMBERS = 7
N_LAYERS = 3
N_FILTERS = 30
KERNEL_SIZE = 20
DILATIONS = (1, 2, 8)
THRESHOLD_PERCENTILE = 99.5


class CausalConv1d(nn.Conv1d):
    """1D convolution that cannot see the future.

    Necessary for time-series control: at run time the exoskeleton has no access to samples
    beyond the present, so training must not either. ``nn.Conv1d`` pads symmetrically, so we
    pad by the full receptive field and discard the right-hand overhang, leaving each output
    position dependent only on the current and preceding inputs.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, **kwargs):
        self.causal_padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=self.causal_padding, dilation=dilation, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        # Slice the TIME axis (dim 2), not channels -- x is (batch, channels, time).
        return out[:, :, :-self.causal_padding] if self.causal_padding > 0 else out


class TCNBlock(nn.Module):
    """Causal convolution -> BatchNorm1d -> ReLU, the unit specified in Table IV."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = KERNEL_SIZE, dilation: int = 1):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class TCNEnsembleMember(nn.Module):
    """One branch of the ensemble: a TCN with two heads (left and right leg).

    Prediction is taken at the **final** time step of the window — the causal setup, where the
    model sees one second of history and estimates the present instant.
    """

    def __init__(self, input_dim: int, output_dim: int = 2,
                 num_channels: tuple[int, ...] = (N_FILTERS,) * N_LAYERS,
                 kernel_size: int = KERNEL_SIZE,
                 dilations: tuple[int, ...] = DILATIONS):
        super().__init__()
        if len(dilations) != len(num_channels):
            raise ValueError(
                f"need one dilation per layer: {len(num_channels)} layers, {len(dilations)} dilations"
            )
        self.kernel_size, self.dilations = kernel_size, dilations

        layers, in_ch = [], input_dim
        for out_ch, d in zip(num_channels, dilations):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation=d))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    @property
    def receptive_field(self) -> int:
        """How many past samples the prediction actually depends on."""
        return 1 + sum((self.kernel_size - 1) * d for d in self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, in_channels, seq_len)
        features = self.network(x)
        # tanh matches the sin(gait phase) target range of [-1, 1]
        return torch.tanh(self.fc(features[:, :, -1]))


class GaitPhaseEnsemble(nn.Module):
    """The full ensemble: independently initialised branches, plus the uncertainty score.

    A single ``nn.Module`` rather than a list, so one ``.to(device)`` and one optimizer cover
    every branch. Branches share no weights — only the input — so they train jointly on the
    same batches but disagree wherever the data does not pin them down.

    ``forward`` returns ``(batch, n_members, 2)``: per-branch predictions for both legs.
    """

    def __init__(self, input_dim: int, output_dim: int = 2, num_models: int = N_MEMBERS, **kwargs):
        super().__init__()
        self.members = nn.ModuleList(
            TCNEnsembleMember(input_dim, output_dim, **kwargs) for _ in range(num_models)
        )

    def __len__(self) -> int:
        return len(self.members)

    @property
    def receptive_field(self) -> int:
        return self.members[0].receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members], dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Ensemble mean, ``(batch, 2)`` — what a controller would consume."""
        return self(x).mean(dim=1)

    @staticmethod
    def uncertainty_from(preds: torch.Tensor) -> torch.Tensor:
        """Psi from stacked predictions ``(batch, n_members, 2)``.

        The average of the per-leg variances across branches, i.e. eq. 2 of the paper.
        Population variance (``unbiased=False``): with a fixed 7 branches we are describing
        the spread of the ensemble we have, not estimating a wider population's.
        """
        return preds.var(dim=1, unbiased=False).mean(dim=1)

    def uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Psi for a batch of windows, ``(batch,)``. Needs no labels — usable at run time."""
        return self.uncertainty_from(self(x))

    @staticmethod
    def fit_threshold(scores: torch.Tensor, percentile: float = THRESHOLD_PERCENTILE) -> float:
        """Decision threshold: the given percentile of in-distribution training scores.

        Anything scoring above it is called out-of-distribution.
        """
        return torch.quantile(scores.flatten().float(), percentile / 100.0).item()


def create_ensemble(input_dim: int, output_dim: int = 2, num_models: int = N_MEMBERS, **kwargs):
    """Build the ensemble. Diversity comes from each branch's random initialisation."""
    return GaitPhaseEnsemble(input_dim, output_dim, num_models, **kwargs)


if __name__ == "__main__":  # python ml/vanilla/model.py
    net = create_ensemble(input_dim=16)
    x = torch.randn(8, 16, 200)
    preds = net(x)
    print(f"{type(net).__name__}: {len(net)} branches, "
          f"{sum(p.numel() for p in net.parameters()):,} params")
    print(f"  receptive field : {net.receptive_field} samples "
          f"({net.receptive_field / 200:.2f} s of the 1.00 s window)")
    print(f"  forward         : {tuple(x.shape)} -> {tuple(preds.shape)} (batch, branches, legs)")
    print(f"  predict         : {tuple(net.predict(x).shape)}")
    print(f"  uncertainty     : {tuple(net.uncertainty(x).shape)}")
    print(f"  output range    : [{preds.min():+.3f}, {preds.max():+.3f}] (tanh-bounded)")
