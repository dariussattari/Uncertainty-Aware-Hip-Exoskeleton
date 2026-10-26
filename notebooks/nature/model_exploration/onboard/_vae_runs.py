"""Load the pulled Nature VAE runs and describe their latent spaces (notebook 02_vae_latent_spaces).

A run folder (``ml/nature/awareness/train_vae.py``; pulled to ``runs/vae`` by ``slurm/pull.sh``) holds config.json,
summary.json, history.json and latents_val.npz: posterior means ``z`` and log variances ``logvar`` (N, 32) of every
in-task window of the four validation participants, with per-window ``nll``, ``kl`` and ``recon_mse``. Everything here
uses the validation participants, so the test participants stay untouched until scoring. Nothing is scored here: LOF,
reconstruction error and the other detection scores are notebook 05's, which reads the same folders.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

ACTIVE_SIGMA2 = 0.9                                      # as train_vae.ACTIVE_SIGMA2
VARIANCE_C = {'fixed': '#2266cc', 'learned': '#e0572b'}  # first two slots of the notebooks' validated palette
CELL_COLS = ['inputs', 'decoder_variance', 'beta']
INPUT_SHORT = {'raw_1s': '1 s', 'raw_1s_spectral': '1 s + spectral', 'raw_3s': '3 s raw', 'raw_3s_spectral': '3 s + spectral'}
INPUT_LS = {'raw_1s': '-', 'raw_1s_spectral': ':', 'raw_3s': '-.', 'raw_3s_spectral': '--'}
LS_NAMES = {'-': 'solid', ':': 'dotted', '-.': 'dash-dot', '--': 'dashed'}


def discover_vae_runs(root: Path, arch: str = 'conv') -> pd.DataFrame:
    """Finished VAE runs of one architecture under ``root`` (summary.json and latents_val.npz present, no smoke runs)."""
    rows = []
    for summary in sorted(Path(root).glob('*/summary.json')):
        run = summary.parent
        if (run / 'latents_val.npz').exists() and not run.name.endswith('_smoke'):
            config = json.loads((run / 'config.json').read_text())
            if config.get('model') == 'vae' and config.get('arch', 'conv') == arch:
                rows.append({'run': run, 'stem': run.name.rpartition('_s')[0], 'beta': float(config['beta']),
                             'decoder_variance': config['decoder_variance'], 'seed': int(config['seed']),
                             'inputs': config.get('inputs', 'raw_1s'), 'config': config})
    columns = ['run', 'stem', 'beta', 'decoder_variance', 'seed', 'inputs', 'config']
    return pd.DataFrame(rows, columns = columns).sort_values(['inputs', 'decoder_variance', 'beta', 'seed']).reset_index(drop = True)


def get_cell_label(inputs: str, variance: str, beta: float) -> str:
    """e.g. '3 s + spectral · fixed · β 128'."""
    return f'{INPUT_SHORT.get(inputs, inputs)} · {variance} · β {beta:g}'


def load_vae_run(run: Path, windows: pd.DataFrame) -> dict:
    """A run's summary, history, and validation export joined to its window labels."""
    data = np.load(run / 'latents_val.npz')
    labels = windows.iloc[data['rows']].reset_index(drop = True)
    labels['is_train'] = False
    return {'summary': json.loads((run / 'summary.json').read_text()), 'config': json.loads((run / 'config.json').read_text()),
            'history': pd.DataFrame(json.loads((run / 'history.json').read_text())), 'labels': labels,
            'z': data['z'], 'logvar': data['logvar'], 'kl': data['kl'], 'nll': data['nll']}


def get_kl_per_dim(z: npt.NDArray, logvar: npt.NDArray) -> npt.NDArray[np.float64]:
    """Mean KL(q(z_j | x) || N(0, 1)) of each latent dimension over windows, (N, D) x 2 -> (D,); sums to the mean KL."""
    z, logvar = z.astype(np.float64), logvar.astype(np.float64)
    return (0.5 * (z ** 2 + np.exp(logvar) - 1 - logvar)).mean(axis = 0)


def get_active(logvar: npt.NDArray, rows: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Active dimensions: median posterior variance over the selected windows below ``ACTIVE_SIGMA2``. (N, D) -> (D,)."""
    return np.median(np.exp(logvar[rows]), axis = 0) < ACTIVE_SIGMA2


def get_width_by_task(labels: pd.DataFrame, logvar: npt.NDArray, active: npt.NDArray[np.bool_]) -> pd.Series:
    """Per task: the median over windows of the mean posterior variance across the active dimensions."""
    width = np.exp(logvar[:, active]).mean(axis = 1) if active.any() else np.full(len(logvar), np.nan)   # (N,)
    return pd.Series(width).groupby(labels['task'].astype(str).to_numpy()).median()


def get_cell_table(runs: dict[Path, dict], meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per seed and per cell (decoder variance, beta; mean ± SD over seeds): training summaries of the validation set."""
    rows = []
    for r in meta.itertuples():
        s = runs[r.run]['summary']
        rows.append({'inputs': r.inputs, 'decoder_variance': r.decoder_variance, 'beta': r.beta, 'seed': r.seed, 'best epoch': s['best_epoch'],
                     'val NLL / window': s['val_nll'], 'val KL / window': s['val_kl'], 'val MSE': s['val_mse'],
                     'active dims (of 32)': s['active_dims'], 'minutes': s['minutes']})
    per_seed = pd.DataFrame(rows).set_index(CELL_COLS + ['seed'])
    stats = per_seed.groupby(level = CELL_COLS).agg(['mean', 'std'])
    fmt = {'best epoch': '{:.0f}', 'val NLL / window': '{:.1f}', 'val KL / window': '{:.2f}', 'val MSE': '{:.4f}',
           'active dims (of 32)': '{:.1f}', 'minutes': '{:.1f}'}
    cells = pd.DataFrame({c: stats[(c, 'mean')].map(f.format) + ' ± ' + stats[(c, 'std')].fillna(0).map(f.format)
                          for c, f in fmt.items()})
    cells.insert(0, 'seeds', per_seed.groupby(level = CELL_COLS).size())
    return cells, per_seed


def plot_curves(runs: dict[Path, dict], meta: pd.DataFrame) -> None:
    """Validation NLL and KL per epoch: one panel pair per (input, decoder variance), one line per beta and seed."""
    groups = list(dict.fromkeys(zip(meta['inputs'], meta['decoder_variance'])))
    fig, axes = plt.subplots(len(groups), 2, figsize = (12, 3.1 * len(groups)), squeeze = False)
    for row, (inputs, variance) in zip(axes, groups):
        part = meta[(meta['inputs'] == inputs) & (meta['decoder_variance'] == variance)]
        betas = sorted(part['beta'].unique())
        shades = dict(zip(betas, np.linspace(0.35, 1.0, len(betas))))       # lightness encodes beta within a group
        for r in part.itertuples():
            h = runs[r.run]['history']
            for ax, col in zip(row, ('val_nll', 'val_kl')):
                ax.plot(h['epoch'], h[col], color = VARIANCE_C[variance], alpha = shades[r.beta], lw = 1.2,
                        label = f'β {r.beta:g}' if (r.seed == part['seed'].min() and col == 'val_nll') else None)
        name = f'{INPUT_SHORT.get(inputs, inputs)}, {variance} decoder variance'
        row[0].set(xlabel = 'epoch', ylabel = 'validation NLL per window', title = f'{name}: reconstruction')
        row[1].set(xlabel = 'epoch', ylabel = 'validation KL per window', title = f'{name}: KL')
        row[0].legend(fontsize = 8, frameon = False, title = 'darker = larger β', title_fontsize = 8)
    plt.tight_layout(); plt.show()


def plot_active_and_kl(per_seed: pd.DataFrame, save: Path | None = None) -> None:
    """Active dimensions and validation KL against beta (log axis), mean ± SD over seeds, one line per decoder variance."""
    stats = per_seed.groupby(level = CELL_COLS)[['active dims (of 32)', 'val KL / window']].agg(['mean', 'std'])
    fig, axes = plt.subplots(1, 2, figsize = (11, 3.8))
    for ax, col in zip(axes, ['active dims (of 32)', 'val KL / window']):
        for inputs, variance in dict.fromkeys(zip(stats.index.get_level_values(0), stats.index.get_level_values(1))):
            part = stats.loc[(inputs, variance)]
            ax.errorbar(part.index, part[(col, 'mean')], yerr = part[(col, 'std')].fillna(0), color = VARIANCE_C[variance],
                        ls = INPUT_LS.get(inputs, '-'), marker = 'o', ms = 5, lw = 2, capsize = 3,
                        label = f'{variance} decoder variance, {INPUT_SHORT.get(inputs, inputs)}')
        ax.set_xscale('log'); ax.set_xticks(sorted(per_seed.index.get_level_values('beta').unique()))
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set(xlabel = 'β', ylabel = col); ax.grid(alpha = .3)
    axes[0].legend(fontsize = 8, frameon = False)
    styles = ', '.join(f'{LS_NAMES[INPUT_LS.get(i, "-")]} {INPUT_SHORT.get(i, i)}' for i in dict.fromkeys(stats.index.get_level_values(0)))
    fig.suptitle(f'Validation participants, familiar windows for active dimensions; mean ± SD over seeds ({styles})', fontsize = 10)
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, dpi = 160)
    plt.show()


def plot_heatmap(grid: pd.DataFrame, title: str, label: str, fmt: str = '{:.2f}', cmap: str = 'Blues',
                 save: Path | None = None) -> None:
    """A labelled heatmap with values written in the cells."""
    fig, ax = plt.subplots(figsize = (max(6, 0.42 * grid.shape[1] + 3), 0.42 * grid.shape[0] + 1.8))
    values = grid.to_numpy(dtype = float)
    im = ax.imshow(values, aspect = 'auto', cmap = cmap)
    hi = np.nanmax(values) if np.isfinite(values).any() else 1
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, fmt.format(values[i, j]), ha = 'center', va = 'center', fontsize = 6.5,
                        color = 'white' if values[i, j] > 0.6 * hi else '#222')
    ax.set_xticks(range(grid.shape[1]), grid.columns, rotation = 90, fontsize = 7.5)
    ax.set_yticks(range(grid.shape[0]), grid.index, fontsize = 8)
    ax.set_title(title, fontsize = 9.5)
    fig.colorbar(im, ax = ax, shrink = .8, label = label); plt.tight_layout()
    if save is not None:
        fig.savefig(save, dpi = 160)
    plt.show()


def plot_umap_by_beta(views: dict[tuple[str, float], dict], labels: pd.DataFrame, column: str, palette: dict[str, str],
                      background: str, title: str, save: Path | None = None) -> None:
    """One UMAP panel per (decoder variance, beta): rows are the variances, columns the betas, all on the same windows."""
    variances = list(dict.fromkeys(k[0] for k in views))
    betas = sorted({k[1] for k in views})
    fig, axes = plt.subplots(len(variances), len(betas), figsize = (3.2 * len(betas), 3.3 * len(variances)), squeeze = False)
    values = labels[column].astype(str).to_numpy()
    rest = ~np.isin(values, list(palette))
    for r, variance in enumerate(variances):
        for c, beta in enumerate(betas):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if (variance, beta) not in views:
                ax.axis('off')
            else:
                pts = views[(variance, beta)][('UMAP', 2)]
                ax.scatter(pts[rest, 0], pts[rest, 1], s = 1, color = background, alpha = .5, rasterized = True)
                for lab, color in palette.items():
                    m = values == lab
                    ax.scatter(pts[m, 0], pts[m, 1], s = 1.4, color = color, alpha = .6, rasterized = True)
                ax.set_title(f'{variance} · β {beta:g}', fontsize = 9)
    handles = [plt.Line2D([], [], marker = 'o', ls = '', color = col, label = lab, markersize = 6) for lab, col in palette.items()]
    fig.legend(handles = handles, loc = 'upper center', bbox_to_anchor = (.5, .955), ncol = min(len(handles), 6), fontsize = 8.5,
               frameon = False)
    fig.suptitle(title, y = .995, fontsize = 11)
    plt.tight_layout(rect = (0, 0, 1, .91))
    if save is not None:
        fig.savefig(save, dpi = 160)
    plt.show()

