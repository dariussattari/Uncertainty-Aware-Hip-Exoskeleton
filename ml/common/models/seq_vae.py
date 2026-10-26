"""Sequence VAEs with the conv VAE's latent: an LSTM-VAE and a Transformer VAE.

Both map a (B, C, W) window to Gaussian mean and log-variance heads of shape (B, 4, 8), the same 4 filters x 8 time
steps as ``models.vae.ConvVAE``, so every score, threshold and plot that reads a latent works on them unchanged; and
both have a non-autoregressive decoder (the latent is upsampled in time and decoded in one pass), so the decoder has no
path to the window but the latent and the KL stays well above zero at beta 1 and 32 (Bowman et al. 2016); beta 128 still
collapses it on raw input (LSTM-VAE on 1 s and 3 s windows, Transformer VAE on 1 s; notebooks 03 and 04).
``forward`` returns (reconstruction, mu, logvar) and ``decoder_logvar`` is the optional per-channel decoder log
variance, as in ConvVAE, so ``models.vae.loss_terms`` gives the ELBO terms of all three.

  LSTMVAE         encoder: 2-layer LSTM over time (hidden 64), its outputs averaged over 8 equal segments of the window,
                  then 1x1 heads to mean and log variance. Decoder: the latent interpolated back to every sample, a
                  2-layer LSTM (hidden 64), a linear map to the channels. The encoder is causal within the window.
  TransformerVAE  encoder: the window cut into 40 patches (5 samples, 25 ms, at 1 s; 15 samples, 75 ms, at 3 s, as the conv
                  VAE's pooling scales with the window; 5 tokens per latent step), each embedded
                  linearly with a learned position, 3 pre-norm self-attention layers (width 64, 4 heads), the tokens
                  averaged in 8 groups of 5, 1x1 heads. Decoder: the latent mapped to width 64, interpolated to the tokens,
                  plus a second learned position, 3 self-attention layers, a linear map from each token back to its patch.
  TCNVAE          encoder: residual blocks of two dilated causal convolutions (kernel 3, width 64, BatchNorm, ReLU), the
                  dilation doubling per block until every output sees the whole window (1 to 32 at 1 s, receptive field
                  253 samples; 1 to 128 at 3 s, 1,021), its outputs averaged over 8 equal segments, 1x1 heads. Decoder: the
                  latent mapped to width 64 and interpolated to every sample, the same blocks with the dilations reversed,
                  a linear 1x1 map to the channels (Bai, Kolter and Koltun 2018).
Nothing trains on import.
"""
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['LSTMVAE', 'TransformerVAE', 'TCNVAE', 'get_tcn_levels', 'LATENT_FILTERS', 'LATENT_STEPS']

LATENT_FILTERS = 4
LATENT_STEPS = 8
LOGVAR_CLAMP = 8.0                       # as ConvVAE: posterior log variance in [-8, 8]


def sample_latent(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
    """The reparameterized sample, or the mean."""
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu


class LSTMVAE(nn.Module):
    """(B, C, W) -> LSTM encoder -> mean and log variance (B, 4, 8) -> LSTM decoder -> (B, C, W)."""

    def __init__(self, channels: int = 16, window: int = 200, learned_variance: bool = False, hidden: int = 64,
                 layers: int = 2, latent_filters: int = LATENT_FILTERS, latent_steps: int = LATENT_STEPS):
        super().__init__()
        if window % latent_steps:
            raise ValueError(f'window {window} is not divisible by {latent_steps} latent steps')
        self.window, self.latent_steps = window, latent_steps
        self.encoder = nn.LSTM(channels, hidden, layers, batch_first = True)
        self.mu_head = nn.Conv1d(hidden, latent_filters, 1)
        self.logvar_head = nn.Conv1d(hidden, latent_filters, 1)
        self.decoder = nn.LSTM(latent_filters, hidden, layers, batch_first = True)
        self.output = nn.Linear(hidden, channels)
        self.decoder_logvar = nn.Parameter(torch.zeros(channels)) if learned_variance else None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.encoder(x.transpose(1, 2))                                   # (B, C, W) -> (B, W, H)
        h = F.avg_pool1d(h.transpose(1, 2), self.window // self.latent_steps)    # (B, H, W) -> (B, H, 8)
        return self.mu_head(h), self.logvar_head(h).clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        u = F.interpolate(z, size = self.window, mode = 'linear', align_corners = False)   # (B, 4, 8) -> (B, 4, W)
        h, _ = self.decoder(u.transpose(1, 2))                                  # (B, W, 4) -> (B, W, H)
        return self.output(h).transpose(1, 2)                                   # (B, W, C) -> (B, C, W)

    def forward(self, x: torch.Tensor, sample: bool | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = sample_latent(mu, logvar, self.training if sample is None else sample)
        return self.decode(z), mu, logvar


class TransformerVAE(nn.Module):
    """(B, C, W) -> patch tokens -> self-attention encoder -> (B, 4, 8) -> self-attention decoder -> (B, C, W)."""

    def __init__(self, channels: int = 16, window: int = 200, learned_variance: bool = False, tokens: int = 40,
                 width: int = 64, heads: int = 4, layers: int = 3, latent_filters: int = LATENT_FILTERS,
                 latent_steps: int = LATENT_STEPS):
        super().__init__()
        if window % tokens or tokens % latent_steps:
            raise ValueError(f'window {window} must divide into {tokens} patches, and they into {latent_steps} latent steps')
        patch = window // tokens
        self.channels, self.patch, self.tokens, self.latent_steps = channels, patch, tokens, latent_steps
        self.embed = nn.Conv1d(channels, width, patch, stride = patch)          # one linear map per patch
        self.pos_enc = nn.Parameter(0.02 * torch.randn(1, self.tokens, width))
        self.pos_dec = nn.Parameter(0.02 * torch.randn(1, self.tokens, width))

        def stack() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout = 0.0, batch_first = True, norm_first = True)
            return nn.TransformerEncoder(layer, layers, enable_nested_tensor = False)
        self.encoder, self.decoder = stack(), stack()
        self.mu_head = nn.Conv1d(width, latent_filters, 1)
        self.logvar_head = nn.Conv1d(width, latent_filters, 1)
        self.latent_in = nn.Conv1d(latent_filters, width, 1)
        self.output = nn.Linear(width, channels * patch)
        self.decoder_logvar = nn.Parameter(torch.zeros(channels)) if learned_variance else None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.embed(x).transpose(1, 2) + self.pos_enc                        # (B, C, W) -> (B, T, D)
        h = self.encoder(t).transpose(1, 2)                                     # (B, T, D) -> (B, D, T)
        h = F.avg_pool1d(h, self.tokens // self.latent_steps)                    # (B, D, T) -> (B, D, 8)
        return self.mu_head(h), self.logvar_head(h).clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        u = F.interpolate(self.latent_in(z), size = self.tokens, mode = 'linear', align_corners = False)   # (B, D, T)
        h = self.decoder(u.transpose(1, 2) + self.pos_dec)                      # (B, T, D)
        out = self.output(h).reshape(len(z), self.tokens, self.channels, self.patch)   # (B, T, C, P)
        return out.permute(0, 2, 1, 3).reshape(len(z), self.channels, self.tokens * self.patch)   # (B, C, W)

    def forward(self, x: torch.Tensor, sample: bool | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = sample_latent(mu, logvar, self.training if sample is None else sample)
        return self.decode(z), mu, logvar


def get_tcn_levels(window: int, kernel: int = 3) -> int:
    """Fewest residual blocks (dilations 1, 2, 4, ...) whose receptive field, 1 + 2 (kernel - 1) (2^levels - 1) samples
    with two convolutions per block, covers ``window``."""
    levels = 1
    for levels in range(1, 16):
        if 1 + 2 * (kernel - 1) * (2 ** levels - 1) >= window:
            break
    return levels


class CausalBlock(nn.Module):
    """Two dilated causal convolutions (left padding only), BatchNorm and ReLU, with a residual 1x1 map when widths differ."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation = dilation)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation = dilation)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(F.pad(x, (self.pad, 0)))))             # (B, C_in, W) -> (B, C_out, W)
        h = self.bn2(self.conv2(F.pad(h, (self.pad, 0))))
        return F.relu(h + self.skip(x))


class TCNVAE(nn.Module):
    """(B, C, W) -> dilated causal TCN encoder -> mean and log variance (B, 4, 8) -> TCN decoder -> (B, C, W)."""

    def __init__(self, channels: int = 16, window: int = 200, learned_variance: bool = False, hidden: int = 64,
                 kernel: int = 3, latent_filters: int = LATENT_FILTERS, latent_steps: int = LATENT_STEPS):
        super().__init__()
        if window % latent_steps:
            raise ValueError(f'window {window} is not divisible by {latent_steps} latent steps')
        self.window, self.latent_steps = window, latent_steps
        dilations = [2 ** i for i in range(get_tcn_levels(window, kernel))]
        self.encoder = nn.Sequential(*[CausalBlock(channels if i == 0 else hidden, hidden, kernel, d) for i, d in enumerate(dilations)])
        self.mu_head = nn.Conv1d(hidden, latent_filters, 1)
        self.logvar_head = nn.Conv1d(hidden, latent_filters, 1)
        self.latent_in = nn.Conv1d(latent_filters, hidden, 1)
        # linear interpolation from the latent steps to every sample as a fixed (steps, W) matrix: the same values as
        # F.interpolate(mode = 'linear'), but its backward pass is a matrix product (25% of a training step otherwise)
        eye = torch.eye(latent_steps)[None]                                     # (1, steps, steps)
        self.register_buffer('upsample', F.interpolate(eye, size = window, mode = 'linear', align_corners = False)[0],
                             persistent = False)                               # (steps, W)
        self.decoder = nn.Sequential(*[CausalBlock(hidden, hidden, kernel, d) for d in reversed(dilations)])
        self.output = nn.Conv1d(hidden, channels, 1)                           # linear: standardized signals take any sign
        self.decoder_logvar = nn.Parameter(torch.zeros(channels)) if learned_variance else None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.avg_pool1d(self.encoder(x), self.window // self.latent_steps)    # (B, C, W) -> (B, H, W) -> (B, H, 8)
        return self.mu_head(h), self.logvar_head(h).clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.latent_in(z)                                                   # (B, 4, 8) -> (B, H, 8)
        u = torch.einsum('bhl,lw->bhw', h, self.upsample.to(h.dtype))           # (B, H, 8) -> (B, H, W)
        return self.output(self.decoder(u))                                     # (B, H, W) -> (B, C, W)

    def forward(self, x: torch.Tensor, sample: bool | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = sample_latent(mu, logvar, self.training if sample is None else sample)
        return self.decode(z), mu, logvar
