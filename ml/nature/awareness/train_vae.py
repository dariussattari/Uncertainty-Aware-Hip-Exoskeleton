"""Variational autoencoders on the Nature data: the standard VAE and the beta grid of the awareness plan (Live_Update 3.3).

    python ml/nature/awareness/train_vae.py --list                                  # the grid and its array indices
    python ml/nature/awareness/train_vae.py --index 0                               # one cell: the SLURM array entry point
    python ml/nature/awareness/train_vae.py --beta 1 --decoder-variance fixed --seed 0 --smoke   # wiring check
    python ml/nature/awareness/train_vae.py --inputs raw_3s_spectral --index 0    # the 3 s + spectral grid (GRIDS)
    python ml/nature/awareness/train_vae.py --arch lstm --inputs raw_3s --index 0  # an LSTM-VAE cell (CORE_CELLS)
    python ml/nature/awareness/train_vae.py --arch lstm --inputs raw_3s --beta 1 --seed 0 --train-class OOD-safe   # training-class ablation

Grid: beta in ``BETAS`` x decoder variance in ``DECODER_VARIANCES`` x seeds 0-2 = 30 cells. Indices 0-14 use the fixed
unit-variance likelihood, 15-29 the learned one; within each, beta in order, three seeds each. Indices 0-2 are the
standard VAE (beta 1, fixed variance, raw 1 s input). On the 3 s + spectral input (``--inputs raw_3s_spectral``; 23 x 600
windows, pooling (5, 15), the same 4 x 8 latent) the grid is the cells the 1 s scoring made worth testing: beta 1, 32,
64 and 128 with the fixed variance and beta 1 with the learned one, three seeds each (indices 0-14). Every other
(architecture, input) uses ``CORE_CELLS``: beta 1, 32 and 128 with the fixed variance and beta 1 with the learned one
(indices 0-11). Inputs (``train_ae.INPUTS``): raw_1s, raw_1s_spectral, raw_3s, raw_3s_spectral.

Architectures (``--arch``): 'conv' (below), and the sequence VAEs of ``ml/common/models/seq_vae.py``, 'lstm' (LSTMVAE),
'transformer' (TransformerVAE) and 'tcn' (TCNVAE), which keep the same 4 x 8 latent so every downstream score applies unchanged. The
LSTM-VAE trains in fp32 (cuDNN's LSTM; autocast off), the others under bf16 autocast on CUDA.

Model: ``ml/common/models/vae.py``'s ConvVAE, the Nature AE's trunk and decoder with Gaussian mean and log-variance heads
(4 x 8 = 32 latent dimensions). 'learned' adds one decoder log variance per input channel (a calibrated decoder, Rybkin
et al. 2021). With the variance fixed at 1, the Science Robotics study found the KL term under 2% of the objective and the
posterior width flat across tasks (``ml/sci_rob/VAE/README.md``); the learned variance is the change that could alter that.

Training follows ``train_ae.py``, so the AE, the Gaussian-NLL control and the VAE differ only in the latent: the same
arrays, participants and standardization, Adam at lr 0.003 and batch 1,024 (the Science Robotics VAE used 0.001 and 256;
its beta 0 control trained better at these), BatchNorm statistics recalibrated before each validation, bf16 autocast on
CUDA, early stopping with patience 5 on the validation negative ELBO (NLL + beta KL, each summed over the window). The
validation objective decodes the posterior mean, so early stopping is not driven by sampling noise. Inputs: raw 1 s, or 3 s + spectral on the smaller grid above.
``--train-class`` (``train_ae.TRAIN_CLASSES``) trains on every safe movement or on the OOD-safe tasks alone instead of the
ID-safe tasks (the training-class ablation; ``train_class_tasks.py``).

Outputs, under ``<out-root>/<arch prefix><_inputs>_b<beta>_<variance>_s<seed>/`` (prefix vae, lstm_vae, xformer_vae; the
input is omitted for raw_1s): config.json, history.json, summary.json, best.pt (weights
and the channel standardization), last.pt (weights, optimizer, RNG states, epoch), and two exports in the format the
notebooks read (``rows``, ``rec``, ``end``, ``z``, ``recon_mse``, ``is_train``, plus ``logvar``, ``nll``, ``kl``):
latents.npz (every in-task window of the test participants and a sample of training windows) and latents_val.npz
(every in-task window of the validation participants). ``z`` is the posterior mean; ``recon_mse``, ``nll`` and ``kl``
come from decoding it. Nothing is scored here (LOF and the other detection scores are notebook 05's).
"""
import argparse
from dataclasses import asdict, dataclass, replace
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import numpy.typing as npt
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'common'))                 # ml/common: dataset-agnostic models
sys.path.insert(0, str(HERE))
from models.autoencoder import BATCH_SIZE, DEFAULT_LR  # noqa: E402
from models.seq_vae import LSTMVAE, TCNVAE, TransformerVAE  # noqa: E402
from models.vae import ConvVAE, get_decoder_logvar, loss_terms  # noqa: E402
from train_ae import (INPUTS, TRAIN_CLASSES, AeData, get_autocast, get_batch, get_git_sha, get_set_rows, load_data,  # noqa: E402
                      recalibrate_batchnorm, resolve_device, save_atomic)

LOG = logging.getLogger('train_vae')
BETAS = (1.0, 8.0, 32.0, 64.0, 128.0)
DECODER_VARIANCES = ('fixed', 'learned')
SEEDS = (0, 1, 2)
ACTIVE_SIGMA2 = 0.9                # a latent dimension is active when its median posterior variance is below this
SMOKE = {'max_epochs': 2, 'patience': 1, 'max_train_windows': 4_096, 'stats_samples': 400_000, 'export_train_windows': 512,
         'bn_batches': 2}


@dataclass
class VaeConfig:
    """Everything that defines a run; written to config.json. Field names shared with train_ae.AeConfig are read by
    train_ae's data and BatchNorm helpers. There is no ``variant`` field: notebook 05 then labels each run by its
    folder stem, which carries beta and the decoder variance."""

    model: str = 'vae'
    arch: str = 'conv'                           # 'conv' (ConvVAE), 'lstm' (LSTMVAE) or 'transformer' (TransformerVAE)
    beta: float = 1.0
    decoder_variance: str = 'fixed'              # 'fixed' (unit variance) or 'learned' (one per input channel)
    seed: int = 0
    inputs: str = 'raw_1s'                       # the window table (ae_data.py; train_ae.INPUTS)
    pool: tuple[int, int] = (5, 5)               # the encoder's pooling for that window (train_ae.INPUTS)
    latent: str = '4x8'                          # latent filters x time steps, for the notebooks' unit labels
    train_class: str = 'ID-safe'                 # controller classes trained on ('+'-joined; train_ae.TRAIN_CLASSES)
    lr: float = DEFAULT_LR                       # 0.003
    batch_size: int = BATCH_SIZE                 # 1,024
    max_epochs: int = 200
    patience: int = 5
    bf16_on_cuda: bool = True
    compile_on_cuda: bool = False                # torch.compile the training forward pass (TCN-VAE: see main)
    max_train_windows: int | None = None         # smoke runs only
    stats_samples: int = 4_000_000
    bn_batches: int = 50
    export_train_windows: int = 20_000


ARCHS = {'conv': 'vae', 'lstm': 'lstm_vae', 'transformer': 'xformer_vae', 'tcn': 'tcn_vae'}   # architecture -> run-name prefix
# (beta, decoder variance) cells of the conv VAE per input; the seeds run fastest within each cell
GRIDS = {'raw_1s': [(b, v) for v in DECODER_VARIANCES for b in BETAS],
         'raw_3s_spectral': [(1.0, 'fixed'), (32.0, 'fixed'), (64.0, 'fixed'), (128.0, 'fixed'), (1.0, 'learned')]}
# every other (architecture, input): the cells that mattered in the conv grid
CORE_CELLS = [(1.0, 'fixed'), (32.0, 'fixed'), (128.0, 'fixed'), (1.0, 'learned')]


def get_cells(arch: str = 'conv', inputs: str = 'raw_1s') -> list[tuple[float, str]]:
    """The (beta, decoder variance) cells of an (architecture, input)."""
    return GRIDS[inputs] if arch == 'conv' and inputs in GRIDS else CORE_CELLS


def get_model(cfg: 'VaeConfig', n_inputs: int, window: int) -> torch.nn.Module:
    """The VAE of ``cfg.arch`` for (n_inputs, window) windows; all give a 4 x 8 latent."""
    learned = cfg.decoder_variance == 'learned'
    if cfg.arch == 'conv':
        model = ConvVAE(n_inputs, window, learned_variance = learned, pool = cfg.pool)
    elif cfg.arch == 'lstm':
        model = LSTMVAE(n_inputs, window, learned_variance = learned)
    elif cfg.arch == 'transformer':
        model = TransformerVAE(n_inputs, window, learned_variance = learned)
    elif cfg.arch == 'tcn':
        model = TCNVAE(n_inputs, window, learned_variance = learned)
    else:
        raise ValueError(f'unknown architecture {cfg.arch!r}; expected one of {sorted(ARCHS)}')
    return model


def get_grid_cell(index: int, inputs: str = 'raw_1s', arch: str = 'conv') -> tuple[float, str, int]:
    """(beta, decoder variance, seed) of an array index: the cell in ``get_cells(arch, inputs)``, then the seed."""
    cells = get_cells(arch, inputs)
    if not 0 <= index < len(cells) * len(SEEDS):
        raise IndexError(f'array index {index} is outside the {inputs} grid 0-{len(cells) * len(SEEDS) - 1}')
    cell, seed = divmod(index, len(SEEDS))
    beta, variance = cells[cell]
    return beta, variance, SEEDS[seed]


def get_run_name(cfg: VaeConfig) -> str:
    """e.g. vae_b8_learned_s2, vae_raw_3s_spectral_b32_fixed_s0, lstm_vae_raw_3s_b1_fixed_s0."""
    inputs = '' if cfg.inputs == 'raw_1s' else f'_{cfg.inputs}'
    return f'{ARCHS[cfg.arch]}{inputs}_b{cfg.beta:g}_{cfg.decoder_variance}_s{cfg.seed}'



def get_losses(model: ConvVAE, x: torch.Tensor, cfg: VaeConfig, device: torch.device, sample: bool
               ) -> tuple[torch.Tensor, ...]:
    """Per-window NLL, KL and MSE (fp32), posterior mean and log variance, for one batch. (B, C, W) -> (B,) x 3, (B, D) x 2."""
    with get_autocast(device, cfg):
        x_hat, mu, logvar = model(x, sample = sample)
    x_hat, mu, logvar = x_hat.float(), mu.float(), logvar.float()
    nll, kl = loss_terms(x, x_hat, mu, logvar, model.decoder_logvar)
    mse = (x_hat - x).square().mean(dim = (1, 2))
    return nll, kl, mse, mu.flatten(1), logvar.flatten(1)


@torch.inference_mode()
def evaluate(model: ConvVAE, data: AeData, rows: torch.Tensor, cfg: VaeConfig) -> dict[str, float]:
    """Validation means per window of NLL, KL, MSE and the objective NLL + beta KL, decoding the posterior mean."""
    model.eval()
    sums = torch.zeros(3, device = rows.device)
    for batch in rows.split(cfg.batch_size):
        nll, kl, mse, _, _ = get_losses(model, get_batch(data, batch), cfg, rows.device, sample = False)
        sums += torch.stack([nll.sum(), kl.sum(), mse.sum()])
    nll, kl, mse = (sums / len(rows)).tolist()
    return {'nll': nll, 'kl': kl, 'mse': mse, 'objective': nll + cfg.beta * kl}


def train(model: ConvVAE, data: AeData, cfg: VaeConfig, out: Path, generator: torch.Generator) -> list[dict]:
    """Adam on the negative ELBO with early stopping on its validation value; leaves the best weights in ``model``."""
    opt = torch.optim.Adam(model.parameters(), lr = cfg.lr)
    device = data.train.device
    forward = torch.compile(model) if cfg.compile_on_cuda and device.type == 'cuda' else model   # shares the parameters
    history, best, stale = [], float('inf'), 0
    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()
        model.train()
        order = data.train[torch.randperm(len(data.train), generator = generator).to(device)]
        n_full = len(order) // cfg.batch_size * cfg.batch_size                 # drop the ragged last batch (BatchNorm)
        sums = torch.zeros(3, device = device)
        for batch in order[:n_full].split(cfg.batch_size):
            nll, kl, mse, _, _ = get_losses(forward, get_batch(data, batch), cfg, device, sample = True)
            loss = (nll + cfg.beta * kl).mean()
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()
            sums += torch.stack([nll.mean().detach(), kl.mean().detach(), mse.mean().detach()])
        train_nll, train_kl, train_mse = (sums / max(n_full // cfg.batch_size, 1)).tolist()
        if not np.isfinite([train_nll, train_kl, train_mse]).all():
            raise FloatingPointError(f'epoch {epoch}: training loss is not finite (NLL {train_nll}, KL {train_kl}); '
                                     'lower the learning rate or check the arrays for NaN inside windows.')
        recalibrate_batchnorm(model, data, cfg, generator)
        val = evaluate(model, data, data.val, cfg)
        record = {'epoch': epoch, 'train_nll': train_nll, 'train_kl': train_kl, 'train_mse': train_mse,
                  **{f'val_{k}': v for k, v in val.items()}, 'seconds': round(time.perf_counter() - t0, 2)}
        history.append(record)
        if val['objective'] < best:
            best, stale = val['objective'], 0
            save_atomic({'model': model.state_dict(), 'mean': torch.from_numpy(data.mean), 'std': torch.from_numpy(data.std),
                         'epoch': epoch, 'config': asdict(cfg)}, out / 'best.pt')
        else:
            stale += 1
        save_atomic({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'history': history,
                     'torch_rng': torch.get_rng_state(), 'generator': generator.get_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, out / 'last.pt')
        LOG.info('epoch %3d  train NLL %.1f KL %.2f MSE %.4f  val NLL %.1f KL %.2f MSE %.4f  objective %.1f%s  %.1fs', epoch,
                 train_nll, train_kl, train_mse, val['nll'], val['kl'], val['mse'], val['objective'],
                 '  *' if stale == 0 else '', record['seconds'])
        if stale >= cfg.patience:
            LOG.info('early stop: no improvement for %d epochs', cfg.patience)
            break
    model.load_state_dict(torch.load(out / 'best.pt', map_location = device, weights_only = True)['model'])
    return history


@torch.inference_mode()
def encode_rows(model: ConvVAE, data: AeData, rows: npt.NDArray[np.int64], cfg: VaeConfig) -> dict[str, npt.NDArray]:
    """Posterior means and log variances (N, 32) and per-window NLL, KL and MSE (N,) of some window rows."""
    model.eval()
    device_rows = torch.from_numpy(rows.astype(np.int64)).to(data.train.device)
    parts = {k: [] for k in ('z', 'logvar', 'nll', 'kl', 'recon_mse')}
    for batch in device_rows.split(cfg.batch_size):
        nll, kl, mse, mu, logvar = get_losses(model, get_batch(data, batch), cfg, device_rows.device, sample = False)
        for key, value in zip(('nll', 'kl', 'recon_mse', 'z', 'logvar'), (nll, kl, mse, mu, logvar)):
            parts[key].append(value.cpu().numpy())
    out = {k: np.concatenate(v).astype(np.float32) for k, v in parts.items()}
    if not all(np.isfinite(v).all() for v in out.values()):
        raise FloatingPointError('non-finite latents or losses in the export')
    return out


def export_latents(model: ConvVAE, data: AeData, cfg: VaeConfig, rng: np.random.Generator, out: Path) -> dict[str, int]:
    """latents.npz (test participants and a training sample) and latents_val.npz (validation participants)."""
    test_rows = get_set_rows(data.windows, 'test', None)
    train_rows = get_set_rows(data.windows, 'train', cfg.train_class)
    train_rows = np.sort(rng.choice(train_rows, min(cfg.export_train_windows, len(train_rows)), replace = False))
    val_rows = get_set_rows(data.windows, 'val', None)
    counts = {}
    for name, rows, is_train in (('latents.npz', np.concatenate([test_rows, train_rows]),
                                  np.r_[np.zeros(len(test_rows), bool), np.ones(len(train_rows), bool)]),
                                 ('latents_val.npz', val_rows, np.zeros(len(val_rows), bool))):
        enc = encode_rows(model, data, rows, cfg)
        np.savez(out / name, rows = rows, rec = data.windows['rec'].to_numpy()[rows], end = data.windows['end'].to_numpy()[rows],
                 is_train = is_train, **enc)
        counts[name] = len(rows)
    return counts


def get_active_dims(out: Path, data: AeData, familiar_class: str) -> tuple[int, list[float]]:
    """Latent dimensions whose median posterior variance over the validation participants' familiar windows (the classes
    trained on, '+'-joined) is below ``ACTIVE_SIGMA2`` (the Science Robotics study's definition), and every dimension's median."""
    export = np.load(out / 'latents_val.npz')
    familiar = np.isin(data.windows['controller_class'].to_numpy()[export['rows']], familiar_class.split('+'))
    median = np.median(np.exp(export['logvar'][familiar]), axis = 0)                 # (N, D) -> (D,)
    return int((median < ACTIVE_SIGMA2).sum()), [round(float(v), 5) for v in median]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--index', type = int, default = None, help = 'grid cell (the SLURM array index); see --list')
    parser.add_argument('--beta', type = float, default = 1.0)
    parser.add_argument('--decoder-variance', choices = DECODER_VARIANCES, default = 'fixed')
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--inputs', choices = sorted(INPUTS), default = 'raw_1s')
    parser.add_argument('--arch', choices = sorted(ARCHS), default = 'conv')
    parser.add_argument('--train-class', choices = TRAIN_CLASSES, default = 'ID-safe', help = 'controller classes to train on')
    parser.add_argument('--list', action = 'store_true', help = 'print the grid and exit')
    parser.add_argument('--data-dir', type = Path,
                        default = Path(os.environ.get('HIPEXO_NATURE_AE_DATA', HERE.parents[2] / 'data' / 'processed_nature' / 'ae')))
    parser.add_argument('--out-root', type = Path, default = HERE / 'runs' / 'vae')
    parser.add_argument('--device', default = None)
    parser.add_argument('--smoke', action = 'store_true')
    parser.add_argument('--force', action = 'store_true', help = 'retrain a cell whose summary.json exists')
    a = parser.parse_args(argv)
    if a.list:
        for i in range(len(get_cells(a.arch, a.inputs)) * len(SEEDS)):
            beta, variance, seed = get_grid_cell(i, a.inputs, a.arch)
            print(f'{i:2d}  {get_run_name(VaeConfig(beta = beta, decoder_variance = variance, seed = seed, inputs = a.inputs, arch = a.arch))}')
        return
    beta, variance, seed = get_grid_cell(a.index, a.inputs, a.arch) if a.index is not None else (a.beta, a.decoder_variance, a.seed)
    cfg = replace(VaeConfig(beta = beta, decoder_variance = variance, seed = seed, inputs = a.inputs, arch = a.arch,
                            pool = INPUTS[a.inputs]['pool'], bf16_on_cuda = a.arch != 'lstm', train_class = a.train_class,
                            # TCN-VAE: memory-bound BatchNorm and elementwise kernels dominate its step; compiling fuses
                            # them, 299 -> 164 ms per step on an A100 (23 x 600 windows, batch 1,024; slurm/profile_tcn.py)
                            compile_on_cuda = a.arch == 'tcn'),
                  **(SMOKE if a.smoke else {}))
    out = a.out_root / f'{get_run_name(cfg)}{"_smoke" if a.smoke else ""}'
    if (out / 'summary.json').exists() and not a.force:
        print(f'already complete: {out} (--force to retrain)')
        return
    out.mkdir(parents = True, exist_ok = True)
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(message)s', datefmt = '%H:%M:%S',
                        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(out / 'train.log', mode = 'w')])
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    device = resolve_device(a.device)
    t0 = time.time()
    data = load_data(a.data_dir, cfg, device, rng)
    model = get_model(cfg, data.n_inputs, data.window).to(device)
    provenance = {'git_sha': get_git_sha(), 'data_version': data.data_version, 'device': str(device),
                  'torch': torch.__version__, 'n_train': len(data.train), 'n_val': len(data.val)}
    (out / 'config.json').write_text(json.dumps({**asdict(cfg), **provenance}, indent = 2))
    LOG.info('%s VAE on %s, beta %g, %s decoder variance, seed %d, trained on %s | %s | train %d, val %d windows | data %s, code %s',
             cfg.arch, cfg.inputs, cfg.beta, cfg.decoder_variance, cfg.seed, cfg.train_class, device, len(data.train), len(data.val),
             data.data_version, provenance['git_sha'])
    history = train(model, data, cfg, out, generator)
    (out / 'history.json').write_text(json.dumps(history, indent = 2))
    counts = export_latents(model, data, cfg, rng, out)
    active, median_sigma2 = get_active_dims(out, data, cfg.train_class)
    best = min(history, key = lambda h: h['val_objective'])
    sigma = (None if model.decoder_logvar is None
             else [round(float(v), 5) for v in torch.exp(0.5 * get_decoder_logvar(model.decoder_logvar.detach().float().cpu()))])
    summary = {**provenance, 'arch': cfg.arch, 'inputs': cfg.inputs, 'train_class': cfg.train_class, 'n_parameters': sum(p.numel() for p in model.parameters()), 'beta': cfg.beta, 'decoder_variance': cfg.decoder_variance, 'seed': cfg.seed,
               'best_epoch': best['epoch'], 'epochs_run': len(history), **{k: v for k, v in best.items() if k.startswith('val_')},
               'active_dims': active, 'median_posterior_variance': median_sigma2, 'decoder_sigma': sigma,
               'exported_windows': counts, 'minutes': round((time.time() - t0) / 60, 2)}
    (out / 'summary.json').write_text(json.dumps(summary, indent = 2))
    LOG.info('done in %.1f min: best epoch %d, val NLL %.1f, KL %.2f, MSE %.4f, %d active dims -> %s', summary['minutes'],
             best['epoch'], best['val_nll'], best['val_kl'], best['val_mse'], active, out)


if __name__ == '__main__':
    main()
