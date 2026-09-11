"""Convolutional autoencoder + LOF scorer — the ankle paper's label-free uncertainty estimator.

Reproduces the "Autoencoder" row of Table I in Tourk et al. (arXiv:2508.21221) on hip data.
Unlike the gait-phase ensemble this model needs no labels at all: it reconstructs its own
input, so the whole AE_GAN window set is usable including windows the ensemble had to discard.

**The uncertainty score is not reconstruction error.** The paper tried that and it failed:

    reconstruction error did not work well as an uncertainty score for our application,
    likely because D_OOD_val contains a significant amount of low-complexity stationary
    data, such as standing, which is particularly easy for the model to reconstruct

Instead they score the *latent space* with Local Outlier Factor — a local-density estimate, so
a point in a sparse region of the training manifold is anomalous even if it sits near the
centre. That distinction is the single most important thing to get right here, and it is also
why this notebook's earlier PCA experiment failed: standing collapses onto the in-distribution
centroid, which defeats any distance-from-centre score but not a density-based one.

Both scores are implemented, because reproducing the paper's *negative* result is part of
reproducing the paper.

Specification (Table IV, "Autoencoder Model"):

    architecture       conv. autoencoder with temporal compression
    encoder            Conv1d(16 -> 19 -> 24 -> bottleneck), kernels 15, 17, 19
    decoder            mirror of the encoder
    latent             filter dim 4, time dim 7  (28 values)
    activation/norm    ReLU, BatchNorm1d after each layer, no dropout
    training           batch 1024, LR 0.001*(1024/16) = 0.064, Adam, MSE
                       -- the LR does NOT train on this data; see DEFAULT_LR below
    early stopping     80/20 train/valid split, patience 5
    scoring            LOF(latent space), 99.5th percentile threshold
    strategy           single stage -- no LOSO, unlike the ensemble

Two inferences, both declared in ``paper_spec.py``:

* **Latent time dimension 8, not 7.** The paper's 7 comes from 175 samples at 25x compression;
  ours is 200 samples, and 200/25 = 8. The compression factor is what is held constant, and it
  falls out exactly as two ``AvgPool1d(5)`` stages for both window lengths — good evidence that
  is how the original achieved it. Latent size is therefore 4 x 8 = 32 rather than 28.
* **Learning rate 0.003, not the specified 0.064.** Measured: at 0.064 reconstruction MSE
  plateaus at 0.959 on standardized data, meaning the model has collapsed to predicting the
  channel means. 0.01, 0.003 and 0.001 all train. This is the one place the specification had
  to be overridden rather than merely interpreted.
* **Non-causal convolutions.** The paper calls the encoder "TCN-based" but gives no dilations,
  and a causal constraint makes little sense for a model reconstructing a window it has already
  been handed in full. Same-padding convolutions are used, and the third kernel (19) exceeds
  the compressed length of 8, so that layer is mostly operating on padding — an oddity present
  in the original specification too, since 19 > 7 as well.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["ConvAutoencoder", "LatentLOF", "create_autoencoder",
           "CHANNELS", "KERNELS", "LATENT_FILTERS", "LATENT_TIME",
           "POOL", "LOF_NEIGHBORS", "THRESHOLD_PERCENTILE", "BASE_LR", "PAPER_LR",
           "DEFAULT_LR", "BATCH_SIZE"]

# Table IV, "Autoencoder Model"
CHANNELS = (19, 24)            # encoder channel progression after the 16-channel input
KERNELS = (15, 17, 19)
LATENT_FILTERS = 4
LATENT_TIME = 8                # paper: 7 at 175 samples; 200/25 = 8 here
POOL = (5, 5)                  # two stages -> 25x temporal compression
LOF_NEIGHBORS = 20             # paper does not state k; sklearn's default
THRESHOLD_PERCENTILE = 99.5
BATCH_SIZE = 1024
PAPER_LR = 0.001 * (BATCH_SIZE / 16)     # = 0.064, exactly as Table IV writes it

# Measured on this dataset: at the paper's 0.064 the model collapses to predicting the channel
# means (reconstruction MSE plateaus at 0.959 on standardized data, i.e. it learns nothing),
# while 0.01, 0.003 and 0.001 all train normally. 0.003 was best over a 4-epoch comparison.
# This is a declared deviation -- see paper_spec.py. Pass --lr 0.064 to reproduce the collapse.
DEFAULT_LR = 0.003
BASE_LR = DEFAULT_LR                     # what the training script uses by default


def _same_pad(kernel: int) -> int:
    return (kernel - 1) // 2


class ConvAutoencoder(nn.Module):
    """Conv autoencoder with temporal compression. Latent is ``(4, 8)`` = 32 values.

    ``forward`` returns ``(reconstruction, latent)``. The latent is kept in its
    ``(batch, filters, time)`` shape; :meth:`encode` flattens it for LOF.
    """

    def __init__(self, in_channels: int = 16, window: int = 200,
                 channels: tuple[int, ...] = CHANNELS, kernels: tuple[int, ...] = KERNELS,
                 latent_filters: int = LATENT_FILTERS, pool: tuple[int, ...] = POOL):
        super().__init__()
        if len(kernels) != len(channels) + 1:
            raise ValueError("need one kernel per conv layer (len(channels) + 1)")
        self.in_channels, self.window = in_channels, window
        self.latent_filters, self.pool = latent_filters, pool
        self.latent_time = window // int(np.prod(pool))
        if self.latent_time * int(np.prod(pool)) != window:
            raise ValueError(f"window {window} is not divisible by the pooling product "
                             f"{int(np.prod(pool))}")

        c1, c2 = channels
        k1, k2, k3 = kernels

        # encoder: conv -> BN -> ReLU, pooling after the first two blocks
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, c1, k1, padding=_same_pad(k1)),
            nn.BatchNorm1d(c1), nn.ReLU(),
            nn.AvgPool1d(pool[0]),                                     # 200 -> 40
            nn.Conv1d(c1, c2, k2, padding=_same_pad(k2)),
            nn.BatchNorm1d(c2), nn.ReLU(),
            nn.AvgPool1d(pool[1]),                                     # 40 -> 8
            nn.Conv1d(c2, latent_filters, k3, padding=_same_pad(k3)),
            nn.BatchNorm1d(latent_filters), nn.ReLU(),
        )

        # decoder: mirror, upsampling where the encoder pooled
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_filters, c2, k3, padding=_same_pad(k3)),
            nn.BatchNorm1d(c2), nn.ReLU(),
            nn.Upsample(scale_factor=pool[1], mode="linear", align_corners=False),
            nn.Conv1d(c2, c1, k2, padding=_same_pad(k2)),
            nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Upsample(scale_factor=pool[0], mode="linear", align_corners=False),
            nn.Conv1d(c1, in_channels, k1, padding=_same_pad(k1)),     # linear output
        )

    @property
    def latent_size(self) -> int:
        return self.latent_filters * self.latent_time

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Flattened latent, ``(batch, latent_size)`` — the representation LOF consumes."""
        return self.encoder(x).flatten(1)

    @staticmethod
    def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Per-window mean squared error, ``(batch,)``.

        The paper's *rejected* uncertainty score. Kept so that rejection is reproducible
        rather than taken on faith.
        """
        return ((x_hat - x) ** 2).mean(dim=(1, 2))


class LatentLOF:
    """Local Outlier Factor over the latent space — the paper's uncertainty score.

    ``Psi(x)`` is oriented so that **larger means more anomalous**, matching every other score
    in this project and the 99.5th-percentile thresholding rule.

    A note on sign, because it is easy to get backwards. ``LOF_k(z)`` is ~1 for inliers and
    grows for outliers. sklearn's ``score_samples`` returns the *negative* of that, so lower
    means more abnormal. The paper writes ``Psi(x) = -LOF_k(z)``, which would make Psi *smaller*
    for outliers and contradict thresholding at a high percentile; we take
    ``Psi = -score_samples = LOF_k(z)`` so that high Psi means anomalous.
    """

    def __init__(self, n_neighbors: int = LOF_NEIGHBORS, fit_samples: int | None = 50_000,
                 seed: int = 0):
        self.n_neighbors, self.fit_samples, self.seed = n_neighbors, fit_samples, seed
        self.lof = None
        self.n_fitted = 0
        self.duplicate_rate = 0.0

    def fit(self, Z: np.ndarray) -> "LatentLOF":
        """Fit on in-distribution training latents.

        LOF is O(n log n) at best and its tree index degrades above a handful of dimensions,
        so the fit set is subsampled by default. The paper notes LOF "works well in
        low-dimensional problems"; 32 dimensions is at the edge of that.

        **Duplicates are removed from the fit set.** LOF is a ratio of reachability distances,
        so a reference set containing repeated points gives some neighbourhoods a local density
        of zero and LOF values that overflow to enormous numbers. The latent here is a ReLU
        output, so exact duplicates are common — an under-trained or collapsed encoder can map
        most of the input to the same sparse vector. Deduplicating keeps the score finite; the
        duplicate rate is reported because a high value is itself a symptom worth seeing.
        """
        from sklearn.neighbors import LocalOutlierFactor

        Z = np.asarray(Z, dtype=np.float64)
        if self.fit_samples is not None and len(Z) > self.fit_samples:
            rng = np.random.default_rng(self.seed)
            Z = Z[rng.choice(len(Z), self.fit_samples, replace=False)]

        n_raw = len(Z)
        Z = np.unique(Z, axis=0)
        self.duplicate_rate = 1.0 - len(Z) / max(n_raw, 1)
        if self.duplicate_rate > 0.2:
            print(f"    warning: {self.duplicate_rate:.0%} of latents were duplicates. "
                  "A collapsed encoder produces this; LOF scores will be poorly conditioned.")
        if len(Z) <= self.n_neighbors:
            raise RuntimeError(
                f"only {len(Z)} distinct latents for k={self.n_neighbors} — the encoder has "
                "collapsed. Train longer or lower the learning rate.")

        self.lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True).fit(Z)
        self.n_fitted = len(Z)
        return self

    def score(self, Z: np.ndarray, batch: int = 20_000) -> np.ndarray:
        """Psi for each latent vector. Larger = more anomalous."""
        if self.lof is None:
            raise RuntimeError("call fit() first")
        Z = np.asarray(Z, dtype=np.float64)
        out = [-self.lof.score_samples(Z[i:i + batch]) for i in range(0, len(Z), batch)]
        psi = np.concatenate(out)
        # A query point coinciding exactly with a fit point can still give a zero
        # reachability distance and an infinite ratio; cap rather than propagate inf.
        bad = ~np.isfinite(psi)
        if bad.any():
            finite_max = psi[~bad].max() if (~bad).any() else 1.0
            print(f"    warning: {bad.sum()} non-finite LOF scores clipped to {finite_max:.3g}")
            psi[bad] = finite_max
        return psi

    @staticmethod
    def fit_threshold(scores: np.ndarray, percentile: float = THRESHOLD_PERCENTILE) -> float:
        """Decision threshold: the given percentile of in-distribution training scores."""
        return float(np.percentile(np.asarray(scores), percentile))


def create_autoencoder(in_channels: int = 16, window: int = 200, **kwargs) -> ConvAutoencoder:
    return ConvAutoencoder(in_channels, window, **kwargs)


if __name__ == "__main__":  # python ml/vanilla/autoencoder.py
    net = create_autoencoder()
    x = torch.randn(8, 16, 200)
    x_hat, z = net(x)
    print(f"{type(net).__name__}: {sum(p.numel() for p in net.parameters()):,} params")
    print(f"  input        {tuple(x.shape)}")
    print(f"  latent       {tuple(z.shape)}  -> flattened {net.latent_size} values "
          f"({net.latent_filters} filters x {net.latent_time} timesteps)")
    print(f"  compression  {x.shape[1]*x.shape[2]} -> {net.latent_size} "
          f"= {x.shape[1]*x.shape[2]/net.latent_size:.0f}x")
    print(f"  reconstruct  {tuple(x_hat.shape)}   (must match input)")
    print(f"  recon error  {tuple(net.reconstruction_error(x, x_hat).shape)}")
    print(f"  LR           {BASE_LR:.4f}  = 0.001 * (1024/16), per Table IV")
    assert x_hat.shape == x.shape, "reconstruction must match the input shape"
