"""Convolutional VAE built on the existing autoencoder's trunk and decoder.

Standard VAE and beta-VAE share this architecture; only the KL weight in the training
objective differs. The decoder likelihood is N(reconstruction, I) in scaled input units, so
the reconstruction term is a fixed-variance Gaussian negative log likelihood, not an MSE.
Nothing trains on import.
"""
import math

import torch
from torch import nn

from .autoencoder import ConvAutoencoder

__all__ = ['ConvVAE', 'loss_terms']


class ConvVAE(nn.Module):
    """16x200 window -> shared conv trunk (24x8) -> mu and log-variance heads (4x8 each)."""

    def __init__(self, channels=16, window=200):
        super().__init__()
        reference = ConvAutoencoder(channels, window)
        self.trunk = reference.encoder[:8]   # 16x200 -> 24x8, the AE's first two blocks
        self.mu_head = nn.Conv1d(24, 4, 19, padding=9)
        self.logvar_head = nn.Conv1d(24, 4, 19, padding=9)
        self.decoder = reference.decoder     # the AE's decoder, 4x8 -> 16x200

    def encode(self, x):
        h = self.trunk(x)
        return self.mu_head(h), self.logvar_head(h).clamp(-8, 8)

    def forward(self, x, sample=None):
        mu, logvar = self.encode(x)
        sample = self.training if sample is None else sample
        z = mu + torch.exp(.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decoder(z), mu, logvar


def loss_terms(x, reconstruction, mu, logvar):
    """Per-window sums of the two ELBO terms (never a mean NLL mixed with a summed KL).

    R(x) = 1/2 sum_{c,t} [(x - x_hat)^2 + log 2 pi]        Gaussian NLL, unit variance
    K(x) = 1/2 sum_j [mu_j^2 + exp(l_j) - 1 - l_j]           KL(q(z|x) || N(0, I))
    """
    nll = .5 * ((x - reconstruction).square() + math.log(2 * math.pi)).flatten(1).sum(1)
    kl = .5 * (mu.square() + logvar.exp() - 1 - logvar).flatten(1).sum(1)
    return nll, kl
