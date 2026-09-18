r"""Probabilistic ensemble — each branch predicts a Gaussian, not a point.

Experiment 5 trains seven TCN branches to predict the hip angle 200 ms ahead under squared
error, and takes the variance across branches as the uncertainty. That variance is a Monte
Carlo estimate of **epistemic** uncertainty: the branches disagree where nothing in training
pinned them down. What it cannot represent is **aleatoric** uncertainty — the future hip angle
is genuinely stochastic even on familiar ground, because heel-strike timing and step-to-step
variability are not predictable from the preceding second whatever the model knows.

Conflating the two matters for this problem. A window can be unpredictable because the movement
is unfamiliar (which should reduce assistance) or because gait is inherently variable at that
instant (which should not). Experiment 5's score cannot tell those apart.

This model gives each branch a second output head emitting log-variance, trains under Gaussian
negative log-likelihood, and reads the two apart exactly as Lakshminarayanan et al. (2017)
prescribe for an ensemble of Gaussian networks. Treating the ensemble as a uniform mixture of
``M`` Gaussians, the mixture variance decomposes without approximation:

    Var[y] = (1/M) sum_i sigma_i^2  +  (1/M) sum_i (mu_i - mubar)^2
             \_________________/       \_____________________/
                  aleatoric                  epistemic

**The epistemic term is Experiment 5's score, exactly.** Same target, same trunk, same
protocol, so it should reproduce Experiment 5's AUROC of 0.899 closely. That is a correctness
check with a known answer, in the same spirit as the Krogh-Vedelsby identity residual: if the
epistemic component lands far from 0.899, the implementation is wrong before the idea is.

The trunk is imported from ``ml/vanilla/models/ensemble.py`` rather than reimplemented, so the
convolutions, dilations, normalisation and initialisation are bit-identical to Experiment 5's.
Only the head differs: ``Linear(30, 2*D)`` in place of ``Linear(30, D)``.

Named ``prob_models`` rather than ``models`` deliberately: ``ml/vanilla`` has a
``models/`` package, and a same-named module here shadows it and breaks the import.
"""

from __future__ import annotations

import paths  # noqa: F401  -- puts ml/vanilla on sys.path

import torch
import torch.nn as nn

from models.ensemble import (DILATIONS, KERNEL_SIZE, N_FILTERS, N_LAYERS, N_MEMBERS,
                             THRESHOLD_PERCENTILE, TCNBlock)

__all__ = ["ProbTCNMember", "ProbabilisticEnsemble", "create_prob_ensemble",
           "LOGVAR_MIN", "LOGVAR_MAX", "N_MEMBERS", "THRESHOLD_PERCENTILE"]

# Clamping log-variance is not cosmetic. Unclamped, a branch can drive log-variance toward
# -inf on windows it happens to fit early, which sends the NLL to -inf and the gradients to
# NaN. The range below spans sigma from about 0.03 to 4.5 in standardised units, which brackets
# everything physically plausible for a standardised hip angle.
LOGVAR_MIN, LOGVAR_MAX = -7.0, 3.0


class ProbTCNMember(nn.Module):
    """One branch: the Experiment 5 trunk, with a mean head and a log-variance head.

    A single ``Linear(filters, 2*output_dim)`` produces both, split on the channel axis. One
    layer rather than two keeps the parameter count and the initialisation as close to
    Experiment 5's as the extra outputs allow.
    """

    def __init__(self, input_dim: int, output_dim: int = 2,
                 num_channels: tuple[int, ...] = (N_FILTERS,) * N_LAYERS,
                 kernel_size: int = KERNEL_SIZE,
                 dilations: tuple[int, ...] = DILATIONS):
        super().__init__()
        if len(dilations) != len(num_channels):
            raise ValueError(
                f"need one dilation per layer: {len(num_channels)} layers, "
                f"{len(dilations)} dilations")
        self.kernel_size, self.dilations = kernel_size, dilations
        self.output_dim = output_dim

        layers, in_ch = [], input_dim
        for out_ch, dil in zip(num_channels, dilations):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation=dil))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 2 * output_dim)

    @property
    def receptive_field(self) -> int:
        return 1 + sum((self.kernel_size - 1) * d for d in self.dilations)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mu, logvar)``, each ``(batch, output_dim)``.

        The prediction is read from the final time step, which is the causal setup: one second
        of history, an estimate of the instant ``horizon`` samples later.
        """
        out = self.fc(self.network(x)[:, :, -1])
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)


class ProbabilisticEnsemble(nn.Module):
    """Independently initialised probabilistic branches, plus the uncertainty decomposition.

    ``forward`` returns ``(mu, logvar)`` stacked as ``(batch, n_branches, output_dim)``.
    """

    def __init__(self, input_dim: int, output_dim: int = 2,
                 num_models: int = N_MEMBERS, **kwargs):
        super().__init__()
        self.members = nn.ModuleList(
            ProbTCNMember(input_dim, output_dim, **kwargs) for _ in range(num_models))
        self.output_dim = output_dim

    def __len__(self) -> int:
        return len(self.members)

    @property
    def receptive_field(self) -> int:
        return self.members[0].receptive_field

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mus, logvars = zip(*(m(x) for m in self.members))
        return torch.stack(mus, dim=1), torch.stack(logvars, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Mixture mean, ``(batch, output_dim)`` — what a controller would consume."""
        mu, _ = self(x)
        return mu.mean(dim=1)

    # -- uncertainty ----------------------------------------------------------

    @staticmethod
    def decompose(mu: torch.Tensor, logvar: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split the mixture variance into aleatoric and epistemic parts.

        ``mu`` and ``logvar`` are ``(batch, M, D)``. Every returned tensor is ``(batch,)``,
        averaged over the output dimensions so the scale matches Experiment 5's score.

        ``epistemic`` uses the population variance (``unbiased=False``), identically to
        ``models.ensemble.GaitPhaseEnsemble.uncertainty_from``, so the two are numerically
        comparable rather than merely analogous.
        """
        aleatoric = logvar.exp().mean(dim=1).mean(dim=1)
        epistemic = mu.var(dim=1, unbiased=False).mean(dim=1)
        return {"aleatoric": aleatoric,
                "epistemic": epistemic,
                "total": aleatoric + epistemic}

    @torch.no_grad()
    def uncertainty_all(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every candidate score for a batch of windows. Needs no target."""
        mu, logvar = self(x)
        return self.decompose(mu, logvar)

    @torch.no_grad()
    def uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """The primary score: the epistemic component.

        Chosen as the default deliberately. It is the term that answers "is this unfamiliar"
        rather than "is this inherently variable", and it is the term that is directly
        comparable to Experiment 5. ``uncertainty_all`` exposes the others, and
        ``evaluate.py`` scores all three side by side rather than assuming this one wins.
        """
        return self.uncertainty_all(x)["epistemic"]

    @staticmethod
    def fit_threshold(scores, percentile: float = THRESHOLD_PERCENTILE) -> float:
        """The paper's rule, unchanged: a percentile of in-distribution training scores."""
        import numpy as np
        return float(np.percentile(np.asarray(scores), percentile))


def create_prob_ensemble(input_dim: int, output_dim: int = 2,
                         num_models: int = N_MEMBERS, **kwargs) -> ProbabilisticEnsemble:
    return ProbabilisticEnsemble(input_dim, output_dim, num_models, **kwargs)


if __name__ == "__main__":  # python ml/prob_ensemble/models.py
    from models.ensemble import create_ensemble

    net = create_prob_ensemble(16, 2)
    ref = create_ensemble(16, 2, activation="linear")
    x = torch.randn(8, 16, 200)
    mu, logvar = net(x)
    u = net.uncertainty_all(x)

    print(f"ProbabilisticEnsemble: {len(net)} branches, "
          f"{sum(p.numel() for p in net.parameters()):,} params")
    print(f"  Experiment 5 for reference        "
          f"{sum(p.numel() for p in ref.parameters()):,} params "
          f"(+{sum(p.numel() for p in net.parameters()) - sum(p.numel() for p in ref.parameters()):,} "
          f"for the log-variance head)")
    print(f"  receptive field : {net.receptive_field} samples "
          f"(Experiment 5: {ref.receptive_field}) -- must match")
    print(f"  forward         : {tuple(x.shape)} -> mu {tuple(mu.shape)}, "
          f"logvar {tuple(logvar.shape)}")
    print(f"  predict         : {tuple(net.predict(x).shape)}")
    print(f"  logvar range    : [{logvar.min():+.2f}, {logvar.max():+.2f}] "
          f"(clamped to [{LOGVAR_MIN}, {LOGVAR_MAX}])")
    for k, v in u.items():
        print(f"  {k:10s}      : {tuple(v.shape)}  mean {v.mean():.3e}")
    assert torch.allclose(u["total"], u["aleatoric"] + u["epistemic"]), "decomposition broken"
    print("  total == aleatoric + epistemic  OK")
