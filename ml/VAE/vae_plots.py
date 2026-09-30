"""Static (matplotlib) figures for one VAE run: PCA of posterior means, score histograms,
and a recorded-versus-decoded trace. The interactive UMAP views live with the notebooks
(``notebooks/model_exploration/_umap_exploration.py``) because they are presentation only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vae_model import ConvVAE
from vae_train import VAEConfig, data_for

__all__ = ['plot_representation']


def plot_representation(out):
    """Fit the display projection on training latents, then project held-out test data."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    out = Path(out)
    if not (out / 'latents_test.npz').exists():
        print('No trained VAE representation yet. Enable training above, then rerun.'); return
    reference = np.load(out / 'train_reference_mu.npy')
    z = np.load(out / 'latents_test.npz'); meta = pd.read_parquet(out / 'latents_test.parquet')
    pca = PCA(n_components=2).fit(reference)
    points = pca.transform(z['Z']); rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mode, g in meta.groupby('mode', sort=False):
        idx = g.index.to_numpy(); idx = rng.choice(idx, min(700, len(idx)), replace=False)
        axes[0].scatter(*points[idx].T, s=8, alpha=.5, label=mode)
    axes[0].set(xlabel='Training-fit PC1', ylabel='Training-fit PC2', title='Posterior means by task')
    axes[0].legend(markerscale=2, ncol=2)
    for kind in (0, 1):
        values = z['smooth_latent_lof'][z['labels'] == kind]
        axes[1].hist(values, bins=60, density=True, alpha=.5, label='OOD' if kind else 'ID')
    threshold = json.loads((out / 'report.json').read_text())['thresholds']['latent_lof']
    axes[1].axvline(threshold, color='black', ls='--', label='Train threshold')
    axes[1].set(xlabel='Latent LOF score', ylabel='Density', title='Test score distributions')
    axes[1].legend(); fig.tight_layout(); plt.show()
    # Representative traces are actual recorded inputs, not synthetic placeholders.
    saved = json.loads((out / 'report.json').read_text())
    cfg = VAEConfig(**saved['config'])
    try:
        dat = data_for(cfg)
    except FileNotFoundError:
        print('Input arrays unavailable; reconstruction traces skipped.'); return
    model = ConvVAE(dat.n_channels, dat.window).eval()
    model.load_state_dict(torch.load(out / 'final.pt', map_location='cpu', weights_only=True)['model'])
    channel = dat.channels.index('enc_angle_l')
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for ax, sp, kind in zip(axes, (dat.test, dat.ood['test']), ('ID', 'OOD')):
        x = torch.from_numpy(sp.take([0]))
        with torch.inference_mode(): rec, _, _ = model(x, sample=False)
        times = (np.arange(dat.window) - dat.window + 1) / 200
        ax.plot(times, x[0, channel], label='Recorded input')
        ax.plot(times, rec[0, channel], ls='--', label='Decode posterior mean')
        row = sp.meta.iloc[0]
        ax.set(title=f'{kind}: {row["subject"]}, {row["mode"]} (first window)',
               xlabel='Time relative to window end (s)', ylabel='Standardized left encoder angle')
        ax.legend()
    fig.tight_layout(); plt.show()
