"""TCN GAN — the ankle paper's third label-free uncertainty estimator.

Reproduces the "GAN" row of Table I in Tourk et al. (arXiv:2508.21221). A generator learns to
fabricate in-distribution hip windows from noise; a discriminator learns to tell real windows
from fabricated ones. The uncertainty score is the discriminator's own verdict:

    Psi(x) = 1 - D(x)

so a window the discriminator is confident is real scores near zero, and one it doubts scores
high. Of the four architectures this has the simplest scorer — no ensemble variance, no LOF, no
second pass.

**The discriminator being imperfect is load-bearing, not a defect.** Its training objective is
separating real windows from *generator output*, and stair ascent is real hip data — a fully
trained discriminator would output ~1 for every real window, in-distribution or not, and
``Psi`` would collapse to ~0 everywhere. The score only grades novelty while D remains
uncertain enough that its output tracks *typicality* rather than mere realness. That is why the
paper runs **five generator updates per discriminator update** — backwards from usual GAN
practice, where the discriminator is kept strong — and why its reported F1 of 71.5 sits between
the ensemble's 90.3 and the autoencoder's 55.7 rather than near-perfect.

The practical consequence: watch ``D(real)`` and ``D(fake)``. If they diverge toward 1 and 0
the discriminator has won and the detector is dead, however healthy the losses look.

Specification (Table IV, "GAN Model"):

    generator          ConvTranspose1d(10 -> 10 -> 12 -> 16), kernel 3, LeakyReLU(0.2)
    discriminator      Conv1d(16 -> 30 -> 30) w/ spectral norm, kernel 5, ReLU, dropout 0.2
    output / latent    sigmoid (discriminator); latent 10 channels x 35 timesteps
    training           step 20, batch 256, LR G=2e-4 D=5e-5, exponential decay gamma=0.99
    optimiser / loss   Adam beta=(0.5, 0.999), binary cross-entropy
    schedule           500 epochs fixed, 80/20 split, 5 G updates per 1 D update
    checkpointing      every 5 epochs
    scoring            discriminator output D(x), 99.5th percentile threshold

Four inferences, all declared in ``paper_spec.py``:

* **Latent 10 x 40, not 10 x 35.** Their 35 is a 175-sample window at a 5x upsample; ours is
  200 samples, and 200/5 = 40. The upsample factor is what is held constant, and it is exactly
  5 for both — the same clean scaling the autoencoder's pooling showed.
* **Upsampling distributed as (5, 1, 1).** Three transpose layers must multiply to 5x and the
  paper does not say how. Putting the stride on the first layer is the simplest reading; the
  remaining two refine at full resolution.
* **Discriminator head.** The spec ends at ``Conv1d(16->30->30)`` and a sigmoid, with no stated
  reduction from 30 channels x 200 timesteps to one number. Global average pooling then a
  linear layer is used, which keeps the discriminator translation-invariant over the window.
* **Noise distribution** is unstated; standard normal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

__all__ = ["Generator", "Discriminator", "HipGan", "create_gan",
           "LATENT_CHANNELS", "LATENT_TIME", "G_CHANNELS", "D_CHANNELS",
           "G_KERNEL", "D_KERNEL", "D_DROPOUT", "LEAKY_SLOPE", "UPSAMPLE",
           "BATCH_SIZE", "LR_G", "LR_D", "LR_DECAY", "ADAM_BETAS",
           "EPOCHS", "G_STEPS_PER_D", "CHECKPOINT_EVERY", "WINDOW_STRIDE_FACTOR",
           "THRESHOLD_PERCENTILE"]

# Table IV, "GAN Model"
LATENT_CHANNELS = 10
LATENT_TIME = 40                 # paper: 35 at 175 samples; 200/5 = 40 here
UPSAMPLE = (5, 1, 1)             # must multiply to 5; distribution is inferred
G_CHANNELS = (10, 12, 16)        # ConvTranspose1d(10 -> 10 -> 12 -> 16)
D_CHANNELS = (30, 30)            # Conv1d(16 -> 30 -> 30)
G_KERNEL, D_KERNEL = 3, 5
LEAKY_SLOPE, D_DROPOUT = 0.2, 0.2
BATCH_SIZE = 256
LR_G, LR_D = 2e-4, 5e-5
LR_DECAY = 0.99                  # ExponentialLR gamma
ADAM_BETAS = (0.5, 0.999)
EPOCHS = 500                     # fixed; GANs have no reliable early-stopping signal
G_STEPS_PER_D = 5                # see the module docstring -- this is load-bearing
CHECKPOINT_EVERY = 5
WINDOW_STRIDE_FACTOR = 2         # "Step: 20" vs the others' 10 -> every 2nd window
THRESHOLD_PERCENTILE = 99.5


def _same_pad(kernel: int) -> int:
    return (kernel - 1) // 2


class Generator(nn.Module):
    """Noise -> a fabricated ``(16, window)`` hip window.

    Transpose convolutions upsample the latent's time axis by ``prod(UPSAMPLE)``. The first
    layer carries the stride; the remaining two refine at full resolution. Output is linear,
    matching the standardised input space the discriminator sees.
    """

    def __init__(self, out_channels: int = 16, window: int = 200,
                 latent_channels: int = LATENT_CHANNELS, latent_time: int = LATENT_TIME,
                 channels: tuple[int, ...] = G_CHANNELS, kernel: int = G_KERNEL,
                 upsample: tuple[int, ...] = UPSAMPLE):
        super().__init__()
        if len(channels) != len(upsample):
            raise ValueError("need one upsample factor per transpose layer")
        total = 1
        for u in upsample:
            total *= u
        if latent_time * total != window:
            raise ValueError(
                f"latent_time {latent_time} x upsample {total} = {latent_time*total}, "
                f"but the window is {window}")

        self.latent_channels, self.latent_time = latent_channels, latent_time
        self.window = window

        layers, c = [], latent_channels
        for i, (out_c, stride) in enumerate(zip(channels, upsample)):
            last = i == len(channels) - 1
            if stride == 1:
                conv = nn.ConvTranspose1d(c, out_c, kernel, stride=1,
                                          padding=_same_pad(kernel))
            else:
                # length = (L-1)*stride - 2*pad + kernel + output_pad; solve for exact upsample
                out_pad = window // total * stride - ((window // total - 1) * stride + kernel)
                conv = nn.ConvTranspose1d(c, out_c, kernel, stride=stride, padding=0,
                                          output_padding=max(out_pad, 0))
            layers.append(conv)
            # BatchNorm on the hidden layers only; the output stays linear
            if not last:
                layers += [nn.BatchNorm1d(out_c), nn.LeakyReLU(LEAKY_SLOPE)]
            c = out_c
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def noise(self, n: int, device=None) -> torch.Tensor:
        """Standard-normal latent, ``(n, latent_channels, latent_time)``."""
        return torch.randn(n, self.latent_channels, self.latent_time, device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """A ``(16, window)`` hip window -> one logit: is this real?

    Spectral normalisation on the convolutions bounds the Lipschitz constant, which is the
    standard stabiliser for adversarial training and is what the specification calls for.
    Global average pooling before the linear head keeps the verdict invariant to where in the
    window an anomaly falls.

    ``forward`` returns a **logit**; use :meth:`probability` for ``D(x)``. Training uses
    ``BCEWithLogitsLoss``, which is numerically stabler than a sigmoid followed by BCE.
    """

    def __init__(self, in_channels: int = 16, channels: tuple[int, ...] = D_CHANNELS,
                 kernel: int = D_KERNEL, dropout: float = D_DROPOUT):
        super().__init__()
        layers, c = [], in_channels
        for out_c in channels:
            layers += [spectral_norm(nn.Conv1d(c, out_c, kernel, padding=_same_pad(kernel))),
                       nn.ReLU(), nn.Dropout(dropout)]
            c = out_c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.features(x)).flatten(1)
        return self.head(h).squeeze(-1)                # logits, (batch,)

    @torch.no_grad()
    def probability(self, x: torch.Tensor) -> torch.Tensor:
        """``D(x)`` in [0, 1] — the discriminator's confidence the window is real."""
        return torch.sigmoid(self(x))


class HipGan(nn.Module):
    """Generator and discriminator together, plus the uncertainty score.

    Holding both in one module means one ``.to(device)`` and one checkpoint, even though the
    two are optimised separately with different learning rates.
    """

    def __init__(self, channels: int = 16, window: int = 200, **kwargs):
        super().__init__()
        g_kw = {k: v for k, v in kwargs.items()
                if k in ("latent_channels", "latent_time", "channels", "kernel", "upsample")}
        self.generator = Generator(channels, window, **g_kw)
        self.discriminator = Discriminator(channels)
        self.window = window

    @property
    def latent_shape(self) -> tuple[int, int]:
        return self.generator.latent_channels, self.generator.latent_time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminator logits for real input — the path used at scoring time."""
        return self.discriminator(x)

    @torch.no_grad()
    def uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """``Psi(x) = 1 - D(x)``, ``(batch,)``. Larger means more anomalous."""
        return 1.0 - torch.sigmoid(self.discriminator(x))

    @staticmethod
    def fit_threshold(scores, percentile: float = THRESHOLD_PERCENTILE) -> float:
        """Decision threshold: the given percentile of in-distribution training scores."""
        import numpy as np
        return float(np.percentile(np.asarray(scores), percentile))


def create_gan(channels: int = 16, window: int = 200, **kwargs) -> HipGan:
    return HipGan(channels, window, **kwargs)


if __name__ == "__main__":  # python ml/vanilla/models/gan.py
    net = create_gan()
    g, d = net.generator, net.discriminator
    x = torch.randn(8, 16, 200)
    z = g.noise(8)
    fake = g(z)
    print(f"HipGan: generator {sum(p.numel() for p in g.parameters()):,} params, "
          f"discriminator {sum(p.numel() for p in d.parameters()):,}")
    print(f"  latent          {tuple(z.shape)}  ({net.latent_shape[0]} x {net.latent_shape[1]})")
    print(f"  generated       {tuple(fake.shape)}   (must match the real window shape)")
    print(f"  D(real) logits  {tuple(d(x).shape)}")
    print(f"  D(real) prob    {d.probability(x).mean():.4f}   D(fake) prob "
          f"{d.probability(fake).mean():.4f}")
    print(f"  Psi             {tuple(net.uncertainty(x).shape)}, "
          f"mean {net.uncertainty(x).mean():.4f}")
    print(f"  spectral norm on D convs: "
          f"{any('parametrizations' in n for n, _ in d.named_buffers() or [('', None)]) or any('parametrizations' in n for n, _ in d.named_parameters())}")
    assert fake.shape == x.shape, "generated windows must match the real input shape"
