r"""Masked Gaussian negative log-likelihood, with the two stabilisers this loss needs.

Experiment 5 trains under masked squared error (``training/ensemble.py:masked_ensemble_mse``).
Replacing that with Gaussian NLL is what turns a point predictor into a probabilistic one, but
NLL is not a drop-in: it has a well-documented failure mode that produces a model with
plausible-looking variance and a worse mean than MSE would have given.

**The failure.** The per-sample NLL is

    0.5 * [ logvar + (y - mu)^2 / exp(logvar) ]

and the gradient reaching ``mu`` is scaled by ``1/exp(logvar)``. Early in training the network
can reduce the loss more cheaply by *raising* log-variance on hard samples than by improving
their mean, and doing so then suppresses their gradient contribution. Hard samples are
progressively ignored; the mean converges to something worse than the MSE solution while the
variance dutifully reports that it was hard. Seitzer et al. (2022), *On the Pitfalls of
Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks*, characterises this
and gives the fix used here.

Two mitigations, both on by default:

``beta`` — beta-NLL
    Multiply each sample's loss by ``detach(sigma^2)^beta``. At ``beta = 0`` this is plain NLL;
    at ``beta = 1`` the ``1/sigma^2`` factor cancels exactly and the mean receives MSE-like
    gradients while the variance head still trains. The detach is essential: the weight must
    not itself be differentiated. ``beta = 0.5`` is the paper's recommendation and the default.

``warmup_epochs``
    Train on masked MSE alone for the first few epochs, ignoring the log-variance head, then
    switch to beta-NLL. Costs nothing and makes the early loss values directly comparable to
    Experiment 5's, which is useful for confirming the trunk is behaving identically.

Both losses average over branches and divide by ``mask.sum() * n_branches``, matching
``masked_ensemble_mse`` exactly, so the reported number stays on the scale of one branch's loss
and remains comparable across folds and across experiments.
"""

from __future__ import annotations

import math

import paths  # noqa: F401  -- puts ml/vanilla on sys.path

import torch

__all__ = ["masked_ensemble_mse_mu", "masked_ensemble_beta_nll", "gaussian_nll_terms",
           "LOG_2PI"]

LOG_2PI = math.log(2.0 * math.pi)


def _denominator(mask: torch.Tensor, n_branches: int) -> torch.Tensor:
    return torch.clamp(mask.sum() * n_branches, min=1.0)


def masked_ensemble_mse_mu(mu: torch.Tensor, y: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE on the mean head only — the warm-up loss.

    Numerically identical to ``training/ensemble.py:masked_ensemble_mse``, so warm-up epochs
    can be compared directly against Experiment 5's loss curve.
    """
    se = (mu - y.unsqueeze(1)) ** 2 * mask.unsqueeze(1)
    return se.sum() / _denominator(mask, mu.shape[1])


def masked_ensemble_beta_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor,
                             mask: torch.Tensor, beta: float = 0.5,
                             full: bool = False) -> torch.Tensor:
    """Masked beta-NLL across every branch.

    ``mu``, ``logvar`` are ``(batch, n_branches, D)``; ``y`` and ``mask`` are ``(batch, D)``
    and broadcast over the branch axis.

    ``full=False`` drops the constant ``0.5 * log(2*pi)`` term, which changes no gradient. Set
    it ``True`` when the number needs to be a real log-likelihood — for instance when comparing
    held-out likelihood across models rather than watching a training curve.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    var = logvar.exp()
    nll = 0.5 * (logvar + (mu - y.unsqueeze(1)) ** 2 / var)
    if full:
        nll = nll + 0.5 * LOG_2PI
    if beta > 0.0:
        # detach: the reweighting must not contribute gradient of its own
        nll = nll * var.detach() ** beta
    return (nll * mask.unsqueeze(1)).sum() / _denominator(mask, mu.shape[1])


@torch.no_grad()
def gaussian_nll_terms(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor,
                       mask: torch.Tensor) -> dict[str, float]:
    """Diagnostics for one batch: the true NLL plus the quantities that reveal the failure.

    ``nll`` is the honest masked Gaussian NLL at ``beta = 0`` with the constant included, so it
    is a comparable held-out number. The rest exist to catch the pathology early:

    ``mse``
        Masked squared error of the mean. If this drifts upward while ``nll`` improves, the
        variance head is buying loss reductions at the mean's expense.
    ``mean_sigma``
        Mean predicted standard deviation. Should settle near the residual scale, not grow.
    ``z_var``
        Variance of the standardised residual ``(y - mu) / sigma``. A calibrated model gives
        **1.0**. Below 1 the model is over-cautious; above 1 it is over-confident. This is the
        single most informative number to watch, and it has a known target.
    ``frac_clamped``
        Fraction of log-variance outputs sitting on either clamp bound. Anything large means
        the clamp is doing load-bearing work and the range should be reconsidered.
    """
    from prob_models import LOGVAR_MAX, LOGVAR_MIN

    m = mask.unsqueeze(1)
    # the numerators below broadcast over the branch axis, so the denominator must count
    # branches too -- otherwise every quantity comes out inflated by n_branches
    w = _denominator(mask, mu.shape[1])
    var = logvar.exp()
    resid = mu - y.unsqueeze(1)
    nll = 0.5 * (LOG_2PI + logvar + resid ** 2 / var)
    z = resid / var.sqrt()
    z_mean = (z * m).sum() / w
    return {
        "nll": float((nll * m).sum() / w),
        "mse": float((resid ** 2 * m).sum() / w),
        "mean_sigma": float((var.sqrt() * m).sum() / w),
        "z_var": float(((z - z_mean) ** 2 * m).sum() / w),
        "frac_clamped": float(
            (((logvar <= LOGVAR_MIN + 1e-4) | (logvar >= LOGVAR_MAX - 1e-4)).float()
             * m).sum() / w),
    }


if __name__ == "__main__":  # python ml/prob_ensemble/losses.py
    torch.manual_seed(0)
    B, M, D = 512, 7, 2
    y = torch.randn(B, D)
    mask = (torch.rand(B, D) > 0.1).float()

    # a perfectly calibrated model: mu = y + noise of sd s, logvar = log(s^2)
    for s in (0.25, 1.0):
        mu = y.unsqueeze(1) + s * torch.randn(B, M, D)
        logvar = torch.full((B, M, D), math.log(s ** 2))
        t = gaussian_nll_terms(mu, logvar, y, mask)
        print(f"calibrated, sigma={s}:  z_var {t['z_var']:.3f} (target 1.000)  "
              f"mean_sigma {t['mean_sigma']:.3f}  nll {t['nll']:+.3f}  mse {t['mse']:.4f}")

    # over-confident: true residual sd 1.0, claimed 0.25 -> z_var should be ~16
    mu = y.unsqueeze(1) + torch.randn(B, M, D)
    logvar = torch.full((B, M, D), math.log(0.25 ** 2))
    print(f"over-confident      :  z_var {gaussian_nll_terms(mu, logvar, y, mask)['z_var']:.2f} "
          f"(expect ~16)")

    # At beta=1 with sigma=1 the beta-weight is a no-op and NLL reduces to 0.5*resid^2, so
    # the mean's gradient must be exactly half the MSE gradient. Compare the tensors directly
    # rather than a ratio -- dividing by a signed tensor with near-zero entries is meaningless.
    mu = torch.randn(B, M, D, requires_grad=True)
    lv = torch.zeros(B, M, D)
    g_nll = torch.autograd.grad(masked_ensemble_beta_nll(mu, lv, y, mask, beta=1.0), mu)[0]
    g_mse = torch.autograd.grad(masked_ensemble_mse_mu(mu, y, mask), mu)[0]
    ok = torch.allclose(g_nll, 0.5 * g_mse, atol=1e-9)
    print(f"beta=1, sigma=1: grad(NLL) == 0.5 * grad(MSE)  ->  {ok}  "
          f"(max abs diff {(g_nll - 0.5 * g_mse).abs().max():.2e})")

    # beta=0 must NOT equal a rescaled MSE gradient once sigma varies -- confirms the
    # 1/sigma^2 reweighting is actually present at beta=0
    lv2 = torch.randn(B, M, D).clamp(-2, 2)
    g0 = torch.autograd.grad(masked_ensemble_beta_nll(mu, lv2, y, mask, beta=0.0), mu)[0]
    g1 = torch.autograd.grad(masked_ensemble_beta_nll(mu, lv2, y, mask, beta=1.0), mu)[0]
    print(f"varying sigma: beta=0 and beta=1 gradients differ  ->  "
          f"{not torch.allclose(g0, g1, atol=1e-6)}  "
          f"(median |g0/g1| {(g0.abs() / g1.abs().clamp(min=1e-12)).median():.3f}, "
          f"expect ~1/sigma^2 spread)")
