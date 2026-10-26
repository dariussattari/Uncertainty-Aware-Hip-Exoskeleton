"""Encode one participant split's in-task windows with the trained Nature autoencoder runs, on the laptop.

    python ml/nature/awareness/encode_split.py --split val                        # every run under runs/ae
    python ml/nature/awareness/encode_split.py --split val --stem vanilla         # one model stem, every seed
    python ml/nature/awareness/encode_split.py --check --stem vanilla             # re-encode exported test windows
    python ml/nature/awareness/encode_split.py --split train --runs notebooks/nature/model_exploration/runs/vae \
        --stem lstm_vae_raw_3s_b1_fixed                                           # a VAE run (any --arch): every train window

Writes ``latents_<split>.npz`` beside each run's ``latents.npz`` in the same format (``rows``, ``rec``, ``end``, ``z``,
``recon_mse``, ``is_train`` all False), so the notebooks' ``_novelty_scores.py`` reads both the same way. The cluster
export (``train_ae.export_latents``) holds the test participants and a sample of training windows only; the
validation participants are needed to set the alarm threshold and choose LOF's dimensionality on held-out people
without touching the test participants (notebook 05, section 8). VAE runs (``train_vae.py``, any architecture) are
encoded with ``train_vae``'s own model and data path and also write ``logvar``, ``nll`` and ``kl``; ``--split train``
gives every in-task training-participant window of every class, which the regime classifier is fitted on (notebook 13).

``--check`` re-encodes a sample of each run's exported test windows and compares them with the export: the cluster
encoded under bf16 autocast, this script in fp32, so agreement is checked as a relative RMS difference, not equality.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'common'))                 # ml/common: dataset-agnostic models
sys.path.insert(0, str(HERE))
from models.autoencoder import ConvAutoencoder  # noqa: E402
from train_ae import get_set_rows, load_data, resolve_device  # noqa: E402
import train_vae as tv  # noqa: E402

LOG = logging.getLogger('encode_split')
BATCH = 2_048
CHECK_WINDOWS = 4_000
CHECK_TOLERANCE = 0.05            # relative RMS difference between fp32 here and the cluster's bf16 export
DEFAULT_RUNS = HERE.parents[2] / 'notebooks' / 'nature' / 'model_exploration' / 'runs' / 'ae'
DEFAULT_DATA = HERE.parents[2] / 'data' / 'processed_nature' / 'ae'


def discover_runs(root: Path, stem: str | None) -> list[Path]:
    """Finished runs under ``root`` (``best.pt``, ``config.json``, ``latents.npz``; no smoke runs), optionally one stem."""
    runs = []
    for best in sorted(root.rglob('best.pt')):
        run = best.parent
        complete = (run / 'config.json').exists() and (run / 'latents.npz').exists() and not run.name.endswith('_smoke')
        if complete and (stem is None or run.name.rpartition('_s')[0] == stem):
            runs.append(run)
    return runs


def load_model(run: Path, config: dict, window: int, device: torch.device
               ) -> tuple[ConvAutoencoder, npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """The run's model in eval mode, and the channel (then feature) standardization it trained with."""
    state = torch.load(run / 'best.pt', map_location = 'cpu', weights_only = True)
    mean, std = state['mean'].numpy().astype(np.float32), state['std'].numpy().astype(np.float32)   # (C + F,)
    latent_filters = int(state['model']['encoder.8.weight'].shape[0])      # the last encoder conv; older configs lack the field
    model = ConvAutoencoder(len(mean), window, latent_filters = latent_filters, pool = tuple(config['pool']),
                            linear_latent = config.get('linear_latent', False))
    model.load_state_dict(state['model'])
    model.to(device).eval()
    return model, mean, std


def load_inputs(data_dir: Path, inputs: str, signals: npt.NDArray[np.float32]) -> tuple[pd.DataFrame, int, list[str], str]:
    """The window table, window length and feature names of one input setting, plus the arrays' data version."""
    meta = json.loads((data_dir / 'meta.json').read_text())
    setting = meta['inputs'][inputs]
    if signals.shape != (meta['n_samples'], len(meta['channels'])):
        raise ValueError(f'signals.npy is {signals.shape}; meta.json expects ({meta["n_samples"]}, {len(meta["channels"])}).')
    return pd.read_parquet(data_dir / setting['table']), int(setting['window']), list(setting['features']), meta['data_version']


@torch.inference_mode()
def encode_rows(model: ConvAutoencoder, signals: npt.NDArray[np.float32], windows: pd.DataFrame, window: int,
                features: list[str], mean: npt.NDArray[np.float32], std: npt.NDArray[np.float32], rows: npt.NDArray[np.int64],
                device: torch.device) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Latent (N, 32) and per-window reconstruction MSE (N,) of some window rows, standardized as in training."""
    n_channels = signals.shape[1]
    starts = windows['start'].to_numpy()[rows]                                           # (N,)
    values = windows[features].to_numpy(dtype = np.float32)[rows] if features else None   # (N, F) or None
    offsets = np.arange(window)
    z_parts, err_parts = [], []
    for i in range(0, len(rows), BATCH):
        x = signals[starts[i:i + BATCH, None] + offsets]                                  # (B, W, C)
        x = (x - mean[:n_channels]) / std[:n_channels]
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))).to(device)      # (B, W, C) -> (B, C, W)
        if values is not None:
            held = torch.from_numpy((values[i:i + BATCH] - mean[n_channels:]) / std[n_channels:]).to(device)
            x = torch.cat([x, held[:, :, None].expand(-1, -1, window)], dim = 1)          # (B, C + F, W)
        x_hat, z = model(x)
        z_parts.append(z.flatten(1).float().cpu().numpy())                                # (B, 32)
        err_parts.append(((x_hat - x) ** 2).mean(dim = (1, 2)).float().cpu().numpy())    # (B,)
    z, err = np.concatenate(z_parts), np.concatenate(err_parts)
    if not (np.isfinite(z).all() and np.isfinite(err).all()):
        raise FloatingPointError(f'non-finite latents or reconstruction errors encoding {len(rows)} windows')
    return z.astype(np.float32), err.astype(np.float32)


def load_vae(run: Path, data_dir: Path, device: torch.device) -> tuple[torch.nn.Module, 'tv.AeData', 'tv.VaeConfig']:
    """A VAE run's model in eval mode with its arrays on the device, standardized exactly as in training (checked against
    the checkpoint: ``load_data`` redraws the same statistics sample from the run's seed)."""
    config = json.loads((run / 'config.json').read_text())
    fields = {k: config[k] for k in tv.VaeConfig.__dataclass_fields__ if k in config}
    cfg = tv.VaeConfig(**{**fields, 'pool': tuple(fields['pool']), 'compile_on_cuda': False})
    data = load_data(data_dir, cfg, device, np.random.default_rng(cfg.seed))
    if config.get('data_version') not in (None, data.data_version):
        raise ValueError(f'{run.name} trained on data version {config.get("data_version")}, the arrays are {data.data_version}.')
    state = torch.load(run / 'best.pt', map_location = 'cpu', weights_only = True)
    for name in ('mean', 'std'):
        np.testing.assert_allclose(getattr(data, name), state[name].numpy(), rtol = 1e-5,
                                   err_msg = f'{run.name}: the standardization differs from the checkpoint\'s')
    model = tv.get_model(cfg, data.n_inputs, data.window)
    model.load_state_dict(state['model'])
    return model.to(device).eval(), data, cfg


def write_vae_split(run: Path, split: str, data_dir: Path, device: torch.device) -> int:
    """Encode every in-task window of one participant split with a VAE run and write ``latents_<split>.npz`` in the
    format of ``train_vae.export_latents``; returns the count."""
    model, data, cfg = load_vae(run, data_dir, device)
    rows = get_set_rows(data.windows, split, None)
    enc = tv.encode_rows(model, data, rows, cfg)
    np.savez(run / f'latents_{split}.npz', rows = rows, rec = data.windows['rec'].to_numpy()[rows],
             end = data.windows['end'].to_numpy()[rows], is_train = np.zeros(len(rows), bool), split = split, **enc)
    return len(rows)


def check_vae_export(run: Path, data_dir: Path, device: torch.device, rng: np.random.Generator) -> dict[str, float]:
    """Re-encode a sample of the run's exported validation windows; relative RMS difference against the export."""
    model, data, cfg = load_vae(run, data_dir, device)
    export = np.load(run / 'latents_val.npz')
    pick = np.sort(rng.choice(len(export['rows']), min(CHECK_WINDOWS, len(export['rows'])), replace = False))
    enc = tv.encode_rows(model, data, export['rows'][pick], cfg)
    rms = lambda a: float(np.sqrt(np.mean(np.square(a, dtype = np.float64))))             # noqa: E731
    result = {f'{k}_rel_rms': rms(enc[k] - export[k][pick]) / rms(export[k][pick]) for k in ('z', 'logvar', 'recon_mse')}
    if max(result.values()) > CHECK_TOLERANCE:
        raise AssertionError(f'{run.name}: re-encoded validation windows differ from the export by {result} (tolerance {CHECK_TOLERANCE})')
    return result


def write_split(run: Path, split: str, signals: npt.NDArray[np.float32], data_dir: Path, device: torch.device) -> int:
    """Encode every in-task window of one participant split and write ``latents_<split>.npz``; returns the count."""
    config = json.loads((run / 'config.json').read_text())
    windows, window, features, data_version = load_inputs(data_dir, config['inputs'], signals)
    model, mean, std = load_model(run, config, window, device)
    if config.get('data_version') not in (None, data_version):
        raise ValueError(f'{run.name} trained on data version {config.get("data_version")}, the arrays are {data_version}.')
    rows = np.flatnonzero(((windows['split'] == split) & windows['in_task']).to_numpy())
    z, err = encode_rows(model, signals, windows, window, features, mean, std, rows, device)
    np.savez(run / f'latents_{split}.npz', rows = rows, rec = windows['rec'].to_numpy()[rows], end = windows['end'].to_numpy()[rows],
             z = z, recon_mse = err, is_train = np.zeros(len(rows), bool), split = split)
    return len(rows)


def check_export(run: Path, signals: npt.NDArray[np.float32], data_dir: Path, device: torch.device,
                 rng: np.random.Generator) -> dict[str, float]:
    """Re-encode a sample of the run's exported test windows; relative RMS difference against the export."""
    config = json.loads((run / 'config.json').read_text())
    windows, window, features, _ = load_inputs(data_dir, config['inputs'], signals)
    model, mean, std = load_model(run, config, window, device)
    export = np.load(run / 'latents.npz')
    test = np.flatnonzero(~export['is_train'])
    pick = np.sort(rng.choice(test, min(CHECK_WINDOWS, len(test)), replace = False))
    z, err = encode_rows(model, signals, windows, window, features, mean, std, export['rows'][pick], device)
    rms = lambda a: float(np.sqrt(np.mean(np.square(a, dtype = np.float64))))             # noqa: E731
    result = {'z_rel_rms': rms(z - export['z'][pick]) / rms(export['z'][pick]),
              'mse_rel_rms': rms(err - export['recon_mse'][pick]) / rms(export['recon_mse'][pick])}
    if max(result.values()) > CHECK_TOLERANCE:
        raise AssertionError(f'{run.name}: re-encoded test windows differ from the export by {result} (tolerance {CHECK_TOLERANCE})')
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', default = 'val', choices = ('train', 'val', 'test'))
    parser.add_argument('--stem', default = None, help = 'run folder name without _s<seed>, e.g. vanilla_raw_3s_spectral')
    parser.add_argument('--runs', type = Path, default = DEFAULT_RUNS)
    parser.add_argument('--data-dir', type = Path, default = DEFAULT_DATA)
    parser.add_argument('--device', default = None)
    parser.add_argument('--force', action = 'store_true', help = 'rewrite an existing latents_<split>.npz')
    parser.add_argument('--check', action = 'store_true', help = 'compare re-encoded test windows with each export instead')
    a = parser.parse_args(argv)
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(message)s', datefmt = '%H:%M:%S')
    device = resolve_device(a.device)
    runs = discover_runs(a.runs, a.stem)
    if not runs:
        raise FileNotFoundError(f'no finished runs under {a.runs}' + (f' with stem {a.stem}' if a.stem else ''))
    t0 = time.time()
    signals = np.load(a.data_dir / 'signals.npy')                            # (n_samples, 16) float32, 1.2 GB, read once
    LOG.info('%d runs | %s | signals %s read in %.0f s', len(runs), device, signals.shape, time.time() - t0)
    rng = np.random.default_rng(0)
    for run in runs:
        t1 = time.time()
        is_vae = json.loads((run / 'config.json').read_text()).get('model') == 'vae'
        if a.check and is_vae:
            result = check_vae_export(run, a.data_dir, device, rng)
            LOG.info('%s: %d validation windows re-encoded, relative RMS difference %s (%.0f s)', run.name, CHECK_WINDOWS,
                     {k: round(v, 4) for k, v in result.items()}, time.time() - t1)
        elif a.check:
            result = check_export(run, signals, a.data_dir, device, rng)
            LOG.info('%s: %d test windows re-encoded, relative RMS difference z %.4f, MSE %.4f (%.0f s)', run.name, CHECK_WINDOWS,
                     result['z_rel_rms'], result['mse_rel_rms'], time.time() - t1)
        elif (run / f'latents_{a.split}.npz').exists() and not a.force:
            LOG.info('%s: latents_%s.npz exists, skipped (--force to rewrite)', run.name, a.split)
        else:
            n = write_vae_split(run, a.split, a.data_dir, device) if is_vae else write_split(run, a.split, signals, a.data_dir, device)
            LOG.info('%s: %d %s windows -> latents_%s.npz (%.0f s)', run.name, n, a.split, a.split, time.time() - t1)
    LOG.info('done in %.1f min', (time.time() - t0) / 60)


if __name__ == '__main__':
    main()
