"""Vanilla and sparse convolutional autoencoders on the Nature data, for the awareness layer.

    python ml/nature/awareness/train_ae.py --variant vanilla --seed 0
    python ml/nature/awareness/train_ae.py --variant sparse --seed 0
    python ml/nature/awareness/train_ae.py --variant sparse --seed 0 --inputs raw_3s_spectral
    python ml/nature/awareness/train_ae.py --variant gaussian --seed 0
    python ml/nature/awareness/train_ae.py --variant vanilla --smoke      # wiring check on a few thousand windows

The model is ``ml/common/models/autoencoder.py``'s ConvAutoencoder, the ankle paper's AE (16 channels x 200 samples
-> 4 x 8 = 32 latent values), at this project's settings: Adam, learning rate 0.003, batch 1,024, early stopping with
patience 5 on held-out participants. As in the ankle paper it trains on in-distribution windows only: the cyclic
ID-safe tasks (normal, incline and weighted walking, stairs), in task, from the training participants. Arrays come
from ``ae_data.py``.

Inputs (``--inputs``; tables from ``ae_data.py``)
  raw_1s           the 16 raw channels over 1 s (200 samples), as in the ankle paper.
  raw_3s_spectral  the 16 raw channels over 3 s (600 samples) plus the seven spectral features at the window end, each
                   held constant across the window as an extra channel (23 channels): the setting the boosted-tree
                   probe and the PCA / UMAP ablation both found best (nature notebooks 02 and 03). The second pooling
                   stage is 15 instead of 5, so the latent stays 4 x 8 = 32 values and only the inputs differ.

Variants
  vanilla  reconstruction MSE only.
  sparse   MSE + lambda * mean over windows and latent units of z / rms_batch(z): an L1 penalty on the (non-negative)
           latent, each unit divided by its RMS over the batch. The division matters. The decoder's first layer is
           followed by BatchNorm, so reconstruction is blind to the latent's scale, and a plain L1 term is met by
           shrinking every unit through the encoder's last BatchNorm without making any of them sparse. Divided by
           its own scale, a unit can lower the penalty only by being zero for more windows (or by being dropped).
  gaussian the Science Robotics VAE study's deterministic control (``ae_control_aehp``, its best model: test AUROC 0.822
           against 0.782 for the plain AE): the encoder ends at its last convolution, without BatchNorm and ReLU, so
           the latent is linear, and the objective is the unit-variance Gaussian NLL summed over each window,
           1/2 sum_{c,t} [(x - x_hat)^2 + log 2 pi]. That is the MSE times C W / 2 plus a constant, so under Adam
           (no gradient clipping here) the update is the MSE update; what this variant changes is the latent head.

Outputs, under ``<out-root>/<run name>/`` (``get_run_name``: variant, non-default inputs, lambda and latent, seed): config.json,
history.json, summary.json, best.pt (weights and the channel standardization), last.pt (weights, optimizer, RNG states,
epoch) and latents.npz (latent and per-window reconstruction MSE for every in-task window of the test participants,
plus a sample of training windows; ``rows`` index the input setting's window table, ``rec`` and ``end`` name each
window in any table).
"""
import argparse
from dataclasses import asdict, dataclass, replace
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'common'))                 # ml/common: dataset-agnostic models
from models.autoencoder import BATCH_SIZE, DEFAULT_LR, ConvAutoencoder  # noqa: E402

LOG = logging.getLogger('train_ae')
# lambda for the sparse variant. On a 30,000-window training subset (40 epochs) lambda 0.03 / 0.1 / 0.3 / 1.0 gave
# scale-free L1 0.66 / 0.36 / 0.19 / 0.11 (vanilla 0.88) at validation MSE +1% / +15% / +28% / +61%; 0.3 is the
# strongest that keeps reconstruction within about 30% of vanilla.
SPARSITY_WEIGHT = 0.3
LOG_2PI = math.log(2 * math.pi)
PENALTY_EPS = 1e-8             # floor on rms^2: far below the decoder BatchNorm's own floor (var + 1e-5)


@dataclass
class AeConfig:
    """Everything that defines a run; written to config.json."""

    variant: str = 'vanilla'
    sparsity_weight: float = 0.0
    linear_latent: bool = False                  # encoder ends at its last conv (no BatchNorm, no ReLU)
    objective: str = 'mse'                       # 'mse', or 'gaussian_nll' (unit variance, summed per window)
    lr: float = DEFAULT_LR                       # 0.003; the paper's 0.064 collapses this model (models/autoencoder.py)
    batch_size: int = BATCH_SIZE                 # 1,024
    max_epochs: int = 200
    patience: int = 5
    seed: int = 0
    train_class: str = 'ID-safe'                 # controller class the AE learns
    inputs: str = 'raw_1s'                       # input setting (ae_data.INPUTS); set with INPUTS below
    latent: str = '4x8'                          # latent filters x time steps (LATENTS); both shapes hold 32 values
    latent_filters: int = 4
    pool: tuple[int, int] = (5, 5)               # the encoder's two pooling stages; window / (pool product) = time steps
    bf16_on_cuda: bool = True                    # autocast; fp32 cuDNN picks FFT convolutions here, 24x slower on A100
    max_train_windows: int | None = None         # smoke runs only
    stats_samples: int = 4_000_000               # samples (in training windows) that fit the channel standardization
    bn_batches: int = 50                         # training batches that set BatchNorm statistics before each validation
    sparsity_warmup_epochs: int = 5              # lambda ramps linearly to its value; early stopping starts after
    export_train_windows: int = 20_000           # training windows exported beside the test participants


VARIANTS = {'vanilla': {'sparsity_weight': 0.0}, 'sparse': {'sparsity_weight': SPARSITY_WEIGHT},
            'gaussian': {'sparsity_weight': 0.0, 'linear_latent': True, 'objective': 'gaussian_nll'}}
INPUTS = {'raw_1s': {'pool': (5, 5)}, 'raw_3s_spectral': {'pool': (5, 15)}}        # both give a 4 x 8 latent
# The same 32 latent values as 8 filters x 4 time steps: on 3 s inputs a step then covers 750 ms, and the 4 x 8 latent's
# neighbouring steps of one filter were strongly correlated (|r| 0.49-0.65), leaving about 4 effective directions.
LATENTS = {'4x8': {'latent_filters': 4, 'pools': {'raw_1s': (5, 5), 'raw_3s_spectral': (5, 15)}},
           '8x4': {'latent_filters': 8, 'pools': {'raw_1s': (5, 10), 'raw_3s_spectral': (5, 30)}}}
SMOKE = {'max_epochs': 2, 'patience': 1, 'max_train_windows': 4_096, 'stats_samples': 400_000, 'export_train_windows': 512,
         'bn_batches': 2, 'sparsity_warmup_epochs': 1}


@dataclass
class AeData:
    """Standardized signals and per-window features on the device, and the window rows of each set."""

    signals: torch.Tensor                        # (n_samples, C) float32, standardized, NaN outside valid windows
    features: torch.Tensor | None                # (n_windows, F) float32, standardized per-window features, or None
    starts: torch.Tensor                         # (n_windows,) int64 first sample of each window
    windows: pd.DataFrame                        # one row per window (the input setting's table)
    train: torch.Tensor                          # (n_train,) int64 rows of ``windows``
    val: torch.Tensor                            # (n_val,) int64 rows of ``windows``
    mean: npt.NDArray[np.float32]                # (C + F,) training-window means, channels then features
    std: npt.NDArray[np.float32]                 # (C + F,) training-window standard deviations
    window: int
    data_version: str

    @property
    def n_inputs(self) -> int:
        return len(self.mean)


# ----------------------------------------------------------------------------- data

def gather_windows(signals: torch.Tensor, starts: torch.Tensor, window: int) -> torch.Tensor:
    """Windows ``signals[start : start + window]`` as the model's (batch, channels, time)."""
    offsets = torch.arange(window, device = starts.device)                     # (W,)
    x = signals[starts[:, None] + offsets]                                     # (B, W, C)
    return x.transpose(1, 2).contiguous()                                      # (B, W, C) -> (B, C, W)


def get_batch(data: AeData, rows: torch.Tensor) -> torch.Tensor:
    """Model input for some window rows: the raw channels, then each per-window feature held constant over time."""
    x = gather_windows(data.signals, data.starts[rows], data.window)          # (B, C, W)
    if data.features is not None:
        held = data.features[rows][:, :, None].expand(-1, -1, data.window)    # (B, F) -> (B, F, W)
        x = torch.cat([x, held], dim = 1)                                      # (B, C + F, W)
    return x


def get_set_rows(windows: pd.DataFrame, split: str, train_class: str | None) -> npt.NDArray[np.int64]:
    """Rows of in-task windows of one participant split, optionally only one controller class."""
    keep = (windows['split'] == split) & windows['in_task']
    if train_class is not None:
        keep &= windows['controller_class'] == train_class
    return np.flatnonzero(keep.to_numpy())


def load_data(data_dir: Path, cfg: AeConfig, device: torch.device, rng: np.random.Generator) -> AeData:
    """Read and check the arrays, fit the standardization on training windows, move the signals to the device."""
    meta = json.loads((data_dir / 'meta.json').read_text())
    setting = meta['inputs'][cfg.inputs]
    windows = pd.read_parquet(data_dir / setting['table'])
    raw = np.load(data_dir / 'signals.npy')                                   # one sequential read (1.2 GB); no NFS seeks
    window = int(setting['window'])
    if raw.dtype != np.float32 or raw.shape != (meta['n_samples'], len(meta['channels'])):
        raise ValueError(f'{data_dir}/signals.npy is {raw.dtype} {raw.shape}; meta.json expects float32 '
                         f'({meta["n_samples"]}, {len(meta["channels"])}). Rebuild with ae_data.py.')
    train_rows = get_set_rows(windows, 'train', cfg.train_class)
    val_rows = get_set_rows(windows, 'val', cfg.train_class)
    if cfg.max_train_windows is not None:
        train_rows = rng.choice(train_rows, min(cfg.max_train_windows, len(train_rows)), replace = False)
        val_rows = rng.choice(val_rows, min(cfg.max_train_windows, len(val_rows)), replace = False)
    starts = windows['start'].to_numpy()
    sample = starts[rng.choice(train_rows, min(cfg.stats_samples // window, len(train_rows)), replace = False)]
    x = raw[sample[:, None] + np.arange(window)]                               # (N, W, C)
    mean, std = x.mean(axis = (0, 1)), x.std(axis = (0, 1))
    features = None
    if setting['features']:
        values = windows[setting['features']].to_numpy(dtype = np.float32)    # (n_windows, F)
        mean = np.r_[mean, values[train_rows].mean(axis = 0)]
        std = np.r_[std, values[train_rows].std(axis = 0)]
        features = torch.from_numpy((values - mean[raw.shape[1]:]) / std[raw.shape[1]:]).to(device)
    if not (np.isfinite(mean).all() and (std > 0).all()):
        raise ValueError(f'standardization is not usable (mean {mean}, std {std}): a training window has a missing or '
                         'constant channel or feature.')
    signals = torch.from_numpy(raw).to(device)                                # (n_samples, C), about 1.2 GB
    signals.sub_(torch.from_numpy(mean[:raw.shape[1]]).to(device)).div_(torch.from_numpy(std[:raw.shape[1]]).to(device))
    as_rows = lambda rows: torch.from_numpy(np.asarray(rows, dtype = np.int64)).to(device)   # noqa: E731
    return AeData(signals, features, torch.from_numpy(starts.astype(np.int64)).to(device), windows, as_rows(train_rows),
                  as_rows(val_rows), mean.astype(np.float32), std.astype(np.float32), window,
                  meta.get('data_version', 'unknown'))


def get_autocast(device: torch.device, cfg: AeConfig) -> torch.autocast:
    """bf16 autocast on CUDA when configured. Measured on an A100 (job 51347312): 551 ms per training step in fp32,
    98% of it in cuDNN's FFT convolution, against 23 ms under bf16 autocast at the same loss. fp32 elsewhere."""
    return torch.autocast(device.type, dtype = torch.bfloat16, enabled = cfg.bf16_on_cuda and device.type == 'cuda')


# ----------------------------------------------------------------------------- loss

def get_reconstruction_objective(mse: torch.Tensor, values_per_window: int, cfg: AeConfig) -> torch.Tensor:
    """The reconstruction term of the objective from the MSE over a batch of equal-sized windows.

    'gaussian_nll' is the unit-variance Gaussian negative log likelihood summed over each window and averaged over
    windows, 1/2 sum_{c,t} [(x - x_hat)^2 + log 2 pi] = values_per_window / 2 * (MSE + log 2 pi), the ``nll`` of
    ``models.vae.loss_terms``.
    """
    result = mse
    if cfg.objective == 'gaussian_nll':
        result = 0.5 * values_per_window * (mse + LOG_2PI)
    elif cfg.objective != 'mse':
        raise ValueError(f'unknown objective {cfg.objective!r}; expected mse or gaussian_nll')
    return result


def get_penalty_from_moments(mean_abs: torch.Tensor, mean_sq: torch.Tensor) -> torch.Tensor:
    """The sparsity penalty from each latent unit's mean |z| and mean z^2 over a set of windows, (D,), (D,) -> ()."""
    return (mean_abs / mean_sq.add(PENALTY_EPS).sqrt()).mean()


def get_sparsity_penalty(z: torch.Tensor) -> torch.Tensor:
    """Mean over windows and latent units of z / rms over the batch: L1 on the latent, scale-free per unit.

    ``z`` is (B, ...) and non-negative (the encoder ends in ReLU). A unit that is constant across the batch scores 1,
    one active in a fraction p of windows (at equal size) scores sqrt(p), a dead unit 0. Multiplying a unit by any
    positive constant leaves the score unchanged. Batches must be shuffled: in a batch drawn from one recording, a
    unit selective for that task looks dense. ``evaluate`` therefore uses moments over the whole set instead.
    """
    flat = z.flatten(1)                                                        # (B, ...) -> (B, D)
    return get_penalty_from_moments(flat.abs().mean(dim = 0), flat.pow(2).mean(dim = 0))


@torch.inference_mode()
def evaluate(model: ConvAutoencoder, data: AeData, rows: torch.Tensor, cfg: AeConfig) -> dict[str, float]:
    """Validation metrics: reconstruction MSE, the sparsity penalty, the objective, active and dead latent units.

    The penalty uses each unit's moments over the whole set, so it does not depend on window order or batching.
    """
    model.eval()
    sums = torch.zeros(2, device = rows.device)                                # mse * n, active * n
    abs_sum, sq_sum, alive = 0.0, 0.0, None
    for batch in rows.split(cfg.batch_size):
        x = get_batch(data, batch)
        with get_autocast(rows.device, cfg):
            x_hat, z = model(x)
        x_hat, z = x_hat.float(), z.float()
        flat = z.flatten(1)                                                    # (B, D)
        n = len(batch)
        sums += torch.stack([F.mse_loss(x_hat, x) * n, (flat > 0).float().mean() * n])
        abs_sum, sq_sum = abs_sum + flat.abs().sum(dim = 0), sq_sum + flat.pow(2).sum(dim = 0)   # (D,)
        fired = (flat > 0).any(dim = 0)                                        # (D,)
        alive = fired if alive is None else alive | fired
    mse, active = (sums / len(rows)).tolist()
    penalty = get_penalty_from_moments(abs_sum / len(rows), sq_sum / len(rows)).item()
    recon = get_reconstruction_objective(torch.tensor(mse), data.n_inputs * data.window, cfg).item()
    return {'mse': mse, 'penalty': penalty, 'objective': recon + cfg.sparsity_weight * penalty,
            'active_fraction': active, 'dead_units': int((~alive).sum())}


# ----------------------------------------------------------------------------- training

@torch.no_grad()
def recalibrate_batchnorm(model: ConvAutoencoder, data: AeData, cfg: AeConfig, generator: torch.Generator) -> None:
    """Replace BatchNorm running statistics by exact averages over ``cfg.bn_batches`` random training batches.

    The running averages (momentum 0.1) drift batch to batch when latent units fire rarely, which made the sparse
    model's validation MSE jump by up to 0.09 between epochs and early stopping pick a lucky dip (job 51360065; with
    recalibrated statistics the same checkpoints scored 0.313 instead of 0.359). Training itself is unaffected: it
    normalizes with each batch's own statistics.
    """
    norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None                                                      # cumulative average
    model.train()
    order = data.train[torch.randperm(len(data.train), generator = generator)[:cfg.bn_batches * cfg.batch_size].to(data.train.device)]
    for batch in order.split(cfg.batch_size):
        with get_autocast(data.train.device, cfg):
            model(get_batch(data, batch))
    for m, momentum in zip(norms, momenta):
        m.momentum = momentum


def get_sparsity_weight(cfg: AeConfig, epoch: int) -> float:
    """lambda for an epoch: ramped linearly over the warm-up so units learn something before the penalty can silence
    them (one seed lost 24 of 32 units in its first two epochs without it)."""
    return cfg.sparsity_weight * min(1.0, (epoch + 1) / max(cfg.sparsity_warmup_epochs, 1))


def save_atomic(obj: dict, path: Path) -> None:
    """torch.save through a temporary file, so a crash never leaves a truncated checkpoint."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(model: ConvAutoencoder, data: AeData, cfg: AeConfig, out: Path, generator: torch.Generator) -> list[dict]:
    """Adam with early stopping on the validation objective; leaves the best weights in ``model``.

    The objective always uses the full lambda, and epochs inside the sparsity warm-up are neither kept as best nor
    counted towards patience, because the model is still optimizing a weaker penalty.
    """
    opt = torch.optim.Adam(model.parameters(), lr = cfg.lr)
    device = data.train.device
    warmup = cfg.sparsity_warmup_epochs if cfg.sparsity_weight > 0 else 0
    history, best, stale = [], float('inf'), 0
    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()
        weight = get_sparsity_weight(cfg, epoch)
        model.train()
        order = data.train[torch.randperm(len(data.train), generator = generator).to(device)]
        n_full = len(order) // cfg.batch_size * cfg.batch_size                 # drop the ragged last batch (BatchNorm)
        sums = torch.zeros(2, device = device)
        for batch in order[:n_full].split(cfg.batch_size):
            x = get_batch(data, batch)
            with get_autocast(device, cfg):
                x_hat, z = model(x)
            mse, penalty = F.mse_loss(x_hat.float(), x), get_sparsity_penalty(z.float())   # losses in fp32
            loss = get_reconstruction_objective(mse, x[0].numel(), cfg) + weight * penalty
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()
            sums += torch.stack([mse.detach(), penalty.detach()])
        train_mse, train_penalty = (sums / max(n_full // cfg.batch_size, 1)).tolist()
        if not np.isfinite([train_mse, train_penalty]).all():
            raise FloatingPointError(f'epoch {epoch}: training loss is not finite (MSE {train_mse}, penalty '
                                     f'{train_penalty}); lower --lr or check the arrays for NaN inside windows.')
        recalibrate_batchnorm(model, data, cfg, generator)
        val = evaluate(model, data, data.val, cfg)
        record = {'epoch': epoch, 'sparsity_weight': weight, 'train_mse': train_mse, 'train_penalty': train_penalty,
                  **{f'val_{k}': v for k, v in val.items()}, 'seconds': round(time.perf_counter() - t0, 2)}
        history.append(record)
        if epoch < warmup - 1:
            stale = -1                                                         # still warming up: no best, no patience
        elif val['objective'] < best:
            best, stale = val['objective'], 0
            save_atomic({'model': model.state_dict(), 'mean': torch.from_numpy(data.mean), 'std': torch.from_numpy(data.std),
                         'epoch': epoch, 'config': asdict(cfg)}, out / 'best.pt')
        else:
            stale += 1
        save_atomic({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'history': history,
                     'torch_rng': torch.get_rng_state(), 'generator': generator.get_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, out / 'last.pt')
        LOG.info('epoch %3d  lambda %.3f  train %.5f (penalty %.3f)  val %.5f (penalty %.3f, active %.2f, dead %d)%s  %.1fs',
                 epoch, weight, train_mse, train_penalty, val['mse'], val['penalty'], val['active_fraction'],
                 val['dead_units'], '  *' if stale == 0 else '', record['seconds'])
        if stale >= cfg.patience:
            LOG.info('early stop: no improvement for %d epochs', cfg.patience)
            break
    model.load_state_dict(torch.load(out / 'best.pt', map_location = device, weights_only = True)['model'])
    return history


@torch.inference_mode()
def export_latents(model: ConvAutoencoder, data: AeData, cfg: AeConfig, rng: np.random.Generator,
                   out: Path) -> int:
    """latents.npz: every in-task test-participant window (all tasks) and a sample of training windows."""
    model.eval()
    test_rows = get_set_rows(data.windows, 'test', None)
    train_rows = get_set_rows(data.windows, 'train', cfg.train_class)
    train_rows = np.sort(rng.choice(train_rows, min(cfg.export_train_windows, len(train_rows)), replace = False))
    rows = np.concatenate([test_rows, train_rows])
    device_rows = torch.from_numpy(rows.astype(np.int64)).to(data.train.device)
    z_parts, err_parts = [], []
    for batch in device_rows.split(cfg.batch_size):
        x = get_batch(data, batch)
        with get_autocast(device_rows.device, cfg):
            x_hat, z = model(x)
        x_hat = x_hat.float()
        z_parts.append(z.flatten(1).float().cpu().numpy())                     # (B, 32)
        err_parts.append(((x_hat - x) ** 2).mean(dim = (1, 2)).float().cpu().numpy())
    z_all, err = np.concatenate(z_parts), np.concatenate(err_parts)
    if not (np.isfinite(z_all).all() and np.isfinite(err).all()):
        raise FloatingPointError('non-finite latents or reconstruction errors in the export')
    np.savez(out / 'latents.npz', rows = rows, rec = data.windows['rec'].to_numpy()[rows], end = data.windows['end'].to_numpy()[rows],
             z = z_all.astype(np.float32), recon_mse = err.astype(np.float32),
             is_train = np.r_[np.zeros(len(test_rows), bool), np.ones(len(train_rows), bool)])
    return len(rows)


def get_git_sha() -> str:
    """The code version: GIT_SHA written by slurm/push.sh on the cluster, else the local checkout's HEAD."""
    stamp = HERE / 'GIT_SHA'
    result = stamp.read_text().strip() if stamp.exists() else 'unknown'
    if not stamp.exists():
        try:
            result = subprocess.run(['git', 'describe', '--always', '--dirty', '--abbrev=7'], cwd = HERE, capture_output = True,
                                    text = True, timeout = 10, check = True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            result = 'unknown'
    return result


def get_run_name(cfg: AeConfig) -> str:
    """variant, then any non-default input setting, lambda and latent shape, then the seed: e.g. sparse_raw_3s_spectral_l0.1_s0."""
    inputs = '' if cfg.inputs == 'raw_1s' else f'_{cfg.inputs}'
    weight = f'_l{cfg.sparsity_weight:g}' if cfg.variant == 'sparse' and cfg.sparsity_weight != SPARSITY_WEIGHT else ''
    latent = '' if cfg.latent == '4x8' else f'_{cfg.latent}'
    return f'{cfg.variant}{inputs}{weight}{latent}_s{cfg.seed}'


def resolve_device(name: str | None) -> torch.device:
    """cuda, then mps, then cpu, unless one is named."""
    auto = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.device(name or auto)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', choices = sorted(VARIANTS), required = True)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--inputs', choices = sorted(INPUTS), default = 'raw_1s')
    parser.add_argument('--sparsity-weight', type = float, default = None, help = f'lambda of the sparse variant (default {SPARSITY_WEIGHT})')
    parser.add_argument('--latent', choices = sorted(LATENTS), default = '4x8', help = 'latent filters x time steps (32 values either way)')
    parser.add_argument('--data-dir', type = Path,
                        default = Path(os.environ.get('HIPEXO_NATURE_AE_DATA', HERE.parents[2] / 'data' / 'processed_nature' / 'ae')))
    parser.add_argument('--out-root', type = Path, default = HERE / 'runs' / 'ae')
    parser.add_argument('--device', default = None)
    parser.add_argument('--smoke', action = 'store_true')
    a = parser.parse_args(argv)
    cfg = replace(AeConfig(variant = a.variant, seed = a.seed, inputs = a.inputs, **VARIANTS[a.variant], **INPUTS[a.inputs]),
                  **(SMOKE if a.smoke else {}))
    if a.sparsity_weight is not None and a.variant == 'sparse':          # the vanilla variant keeps lambda 0
        cfg = replace(cfg, sparsity_weight = a.sparsity_weight)
    cfg = replace(cfg, latent = a.latent, latent_filters = LATENTS[a.latent]['latent_filters'], pool = LATENTS[a.latent]['pools'][a.inputs])
    out = a.out_root / f'{get_run_name(cfg)}{"_smoke" if a.smoke else ""}'
    out.mkdir(parents = True, exist_ok = True)
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(message)s', datefmt = '%H:%M:%S',
                        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(out / 'train.log', mode = 'w')])
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    device = resolve_device(a.device)
    t0 = time.time()
    data = load_data(a.data_dir, cfg, device, rng)
    model = ConvAutoencoder(data.n_inputs, data.window, latent_filters = cfg.latent_filters, pool = cfg.pool,
                            linear_latent = cfg.linear_latent).to(device)
    provenance = {'git_sha': get_git_sha(), 'data_version': data.data_version, 'device': str(device),
                  'torch': torch.__version__, 'n_train': len(data.train), 'n_val': len(data.val)}
    (out / 'config.json').write_text(json.dumps({**asdict(cfg), **provenance}, indent = 2))
    LOG.info('%s seed %d | %s (%d inputs x %d samples, pool %s, latent %s) | lambda %g | %s | train %d, val %d windows | data %s, code %s',
             cfg.variant, cfg.seed, cfg.inputs, data.n_inputs, data.window, cfg.pool, cfg.latent, cfg.sparsity_weight, device,
             len(data.train), len(data.val), data.data_version, provenance['git_sha'])
    history = train(model, data, cfg, out, generator)
    (out / 'history.json').write_text(json.dumps(history, indent = 2))
    n_export = export_latents(model, data, cfg, rng, out)
    best = min(history, key = lambda h: h['val_objective'])
    summary = {**provenance, 'variant': cfg.variant, 'inputs': cfg.inputs, 'seed': cfg.seed, 'sparsity_weight': cfg.sparsity_weight,
               'best_epoch': best['epoch'], 'epochs_run': len(history),
               **{k: v for k, v in best.items() if k.startswith('val_')}, 'exported_windows': n_export,
               'minutes': round((time.time() - t0) / 60, 2)}
    (out / 'summary.json').write_text(json.dumps(summary, indent = 2))
    LOG.info('done in %.1f min: best epoch %d, val MSE %.5f, active %.2f, dead units %d -> %s', summary['minutes'],
             best['epoch'], best['val_mse'], best['val_active_fraction'], best['val_dead_units'], out)


if __name__ == '__main__':
    main()
