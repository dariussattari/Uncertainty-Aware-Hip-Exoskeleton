"""Convolutional VAE built on the existing autoencoder's trunk and decoder.

Standard VAE and beta-VAE share this architecture; only the KL weight in the training
objective differs. The decoder likelihood is N(reconstruction, I) in scaled input units, so
the reconstruction term is a fixed-variance Gaussian negative log likelihood, not an MSE.
``learned_variance=True`` replaces I by one learned variance per input channel (a calibrated
decoder, Rybkin et al. 2021), held above ``DECODER_LOGVAR_FLOOR`` by a soft floor.
Nothing trains on import.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .autoencoder import CHANNELS, KERNELS, LATENT_FILTERS, POOL, ConvAutoencoder

__all__ = ['ConvVAE', 'loss_terms', 'get_decoder_logvar', 'DECODER_LOGVAR_FLOOR']

DECODER_LOGVAR_FLOOR = -6.0       # log variance in scaled input units: sigma >= 0.05 channel SDs


class ConvVAE(nn.Module):
    """16x200 window -> shared conv trunk (24x8) -> mu and log-variance heads (4x8 each).

    ``pool`` and ``latent_filters`` pass through to the AE whose trunk and decoder are reused, so a 23x600 window with
    pool (5, 15) gives the same 4x8 latent (the Nature 3 s + spectral input). The defaults keep the original model.
    """

    def __init__(self, channels=16, window=200, learned_variance=False, pool=POOL, latent_filters=LATENT_FILTERS):
        super().__init__()
        reference = ConvAutoencoder(channels, window, pool=tuple(pool), latent_filters=latent_filters)
        self.trunk = reference.encoder[:8]   # 16x200 -> 24x8, the AE's first two blocks
        c2, k3 = CHANNELS[1], KERNELS[2]
        self.mu_head = nn.Conv1d(c2, latent_filters, k3, padding=k3 // 2)
        self.logvar_head = nn.Conv1d(c2, latent_filters, k3, padding=k3 // 2)
        self.decoder = reference.decoder     # the AE's decoder, 4x8 -> 16x200
        # one raw log variance per input channel; None keeps the unit-variance likelihood
        self.decoder_logvar = nn.Parameter(torch.zeros(channels)) if learned_variance else None

    def encode(self, x):
        h = self.trunk(x)
        return self.mu_head(h), self.logvar_head(h).clamp(-8, 8)

    def forward(self, x, sample=None):
        mu, logvar = self.encode(x)
        sample = self.training if sample is None else sample
        z = mu + torch.exp(.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decoder(z), mu, logvar


def get_decoder_logvar(raw):
    """Effective per-channel decoder log variance: a soft floor at DECODER_LOGVAR_FLOOR, so the variance cannot
    collapse and the parameter keeps a gradient near the floor (a hard clamp would stop it dead there). (C,) -> (C,)."""
    return DECODER_LOGVAR_FLOOR + F.softplus(raw - DECODER_LOGVAR_FLOOR)


def loss_terms(x, reconstruction, mu, logvar, decoder_logvar=None):
    """Per-window sums of the two ELBO terms (never a mean NLL mixed with a summed KL).

    R(x) = 1/2 sum_{c,t} [(x - x_hat)^2 + log 2 pi]        Gaussian NLL, unit variance
    R(x) = 1/2 sum_{c,t} [(x - x_hat)^2 / s_c + log s_c + log 2 pi]    with a learned variance s_c per channel,
                                                            ``decoder_logvar`` the raw parameter (ConvVAE.decoder_logvar)
    K(x) = 1/2 sum_j [mu_j^2 + exp(l_j) - 1 - l_j]           KL(q(z|x) || N(0, I))
    """
    if decoder_logvar is None:
        nll = .5 * ((x - reconstruction).square() + math.log(2 * math.pi)).flatten(1).sum(1)
    else:
        lv = get_decoder_logvar(decoder_logvar)[None, :, None]               # (C,) -> (1, C, 1)
        nll = .5 * ((x - reconstruction).square() * torch.exp(-lv) + lv + math.log(2 * math.pi)).flatten(1).sum(1)
    kl = .5 * (mu.square() + logvar.exp() - 1 - logvar).flatten(1).sum(1)
    return nll, kl
