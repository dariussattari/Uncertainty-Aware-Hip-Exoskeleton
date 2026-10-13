"""Helpers for the Nature autoencoder latent-space notebook: load runs, project latents, draw the label lenses.

A *view* is one way of placing windows in 2-D or 3-D: the latent units themselves (the highest-variance units, raw
values), PCA, or UMAP of the standardized latent. A *lens* is one labelling of the same windows: task, task family,
controller class, movement direction, or the Science Robotics mode the task shares.
"""
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', message = '.*n_jobs value.*')     # umap: a fixed seed means one thread, by design
warnings.filterwarnings('ignore', message = 'IProgress not found')
import umap  # noqa: E402

VIEWS = ('latent units', 'PCA', 'UMAP')
BACKGROUND = '#d9d9d9'
FAMILY_C = {'walking': '#2a78d6', 'stairs / curbs / step-ups': '#eb6834', 'squat / lift / sit / lunge': '#1baf7a',
            'jump / run / cut / shuffle': '#eda100', 'standing / unstructured': '#4a3aa7'}
CLASS_C = {'ID-safe': '#287da8', 'OOD-safe': '#dd791a', 'OOD-dangerous': '#a32d2d'}
DIRECTION_C = {'up': '#c44e00', 'down': '#2a78d6', 'level': '#2e9e6b'}
# ascent dark, descent light within one hue per terrain: direction reads as lightness, terrain as hue (palette validated)
SCIROB_C = {'LG': '#2266cc', 'RA': '#a3470b', 'RD': '#ec9446', 'SA': '#0f7d55', 'SD': '#56c08e', 'ST': '#6a4fc2', 'TR': '#b88a00'}
# tasks within one family panel (validated for colour-vision deficiency); panels never share a task, so hues may repeat
PANEL_C = ('#2266cc', '#e0572b', '#0f9d6e', '#b0479a', '#d99a00', '#3fb0e0', '#6a4fc2', '#94a01c')
SYMBOLS_3D = ('circle', 'square', 'diamond', 'cross', 'circle-open', 'square-open', 'diamond-open', 'x')


def get_task_styles(families: dict[str, tuple[str, ...]]) -> dict[str, dict[str, str]]:
    """Per task: its 2-D panel colour, its 3-D colour (the family's) and its 3-D marker (its place in the family)."""
    return {task: {'panel': PANEL_C[i], 'family': FAMILY_C[family], 'symbol': SYMBOLS_3D[i]}
            for family, tasks in families.items() for i, task in enumerate(tasks)}


def load_run(run_dir: Path, windows: pd.DataFrame) -> dict:
    """A pulled run: its summary, history, and the exported latents joined to their labels in ``windows`` (the run's table)."""
    data = np.load(run_dir / 'latents.npz')
    labels = windows.iloc[data['rows']].reset_index(drop = True)
    labels['is_train'] = data['is_train']
    labels['recon_mse'] = data['recon_mse']
    return {'summary': json.loads((run_dir / 'summary.json').read_text()), 'config': json.loads((run_dir / 'config.json').read_text()),
            'history': pd.DataFrame(json.loads((run_dir / 'history.json').read_text())),
            'labels': labels, 'z': data['z']}


def get_aligned_z(run: dict, keys: pd.DataFrame) -> npt.NDArray[np.float32]:
    """The run's latents for the windows named by ``keys`` (rec, end), in that order; every key must be exported."""
    idx = run['labels'][['rec', 'end']].reset_index().merge(keys[['rec', 'end']], on = ['rec', 'end'], how = 'right')
    if idx['index'].isna().any():
        raise KeyError(f'{int(idx["index"].isna().sum())} windows are missing from this run\'s export')
    return run['z'][idx['index'].to_numpy(dtype = np.int64)]


def get_effective_directions(z: npt.NDArray[np.float32], standardize: bool = True) -> float:
    """Participation ratio (sum ev)^2 / sum ev^2 of the active units' correlation spectrum: how many independent
    directions the latent spreads windows along (1 when every unit carries the same signal, the unit count when none
    are correlated). ``standardize=False`` uses the covariance instead, weighting units by their variance: what a
    distance in the raw latent (LOF) sees."""
    active = z.std(axis = 0) > 1e-6
    za = z[:, active].astype(np.float64)
    ev = np.clip(np.linalg.eigvalsh(np.corrcoef(za.T) if standardize else np.cov(za.T)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def get_top_unit_correlation(z: npt.NDArray[np.float32]) -> float:
    """|r| between the two highest-variance latent units: what the "latent units" view plots against each other."""
    top = np.argsort(z.var(axis = 0))[::-1][:2]
    return float(abs(np.corrcoef(z[:, top].T)[0, 1]))


def get_novelty_units(task_means: pd.DataFrame, id_tasks: tuple[str, ...], ratio: float = 5.0) -> pd.Series:
    """Per latent unit (columns of ``task_means``, rows are tasks): active, strongest on a task outside ``id_tasks``,
    and at least ``ratio`` times its strongest mean on any ``id_tasks`` task."""
    peak, id_max = task_means.max(), task_means.loc[list(id_tasks)].max()
    return (peak > 1e-6) & (peak >= ratio * id_max) & ~task_means.idxmax().isin(id_tasks)


def plot_unit_heatmaps(runs: dict[str, dict], families: dict[str, tuple[str, ...]], id_tasks: tuple[str, ...],
                       title: str) -> None:
    """Per model: each latent unit's mean per task over the test participants' in-task windows, scaled to its maximum.

    Rows are labelled filter · time step (from the run's latent shape) and ordered by the task they respond to most. * marks a *novelty unit*: its
    strongest task is one the model never trained on, at 5 times or more its strongest training-task (ID-safe) mean.
    Units that never fire on these windows are labelled unused.
    """
    order_tasks = [t for ts in families.values() for t in ts]
    fig, axes = plt.subplots(1, len(runs), figsize = (7 * len(runs), 6.2), squeeze = False)
    for ax, (name, r) in zip(axes[0], runs.items()):
        test = ~r['labels']['is_train'].to_numpy()
        means = pd.DataFrame(r['z'][test]).groupby(r['labels']['task'].astype(str).to_numpy()[test]).mean().reindex(order_tasks).T
        peak = means.max(axis = 1)                                          # (32,)
        norm = means.div(peak.where(peak > 0, 1), axis = 0)
        alive = peak > 1e-6
        novelty = get_novelty_units(means.T, id_tasks)
        order = np.lexsort((-norm.max(axis = 1).to_numpy(), norm.to_numpy().argmax(axis = 1)))
        im = ax.imshow(norm.to_numpy()[order], aspect = 'auto', cmap = 'Blues', vmin = 0, vmax = 1)
        ax.set_xticks(range(len(order_tasks)), order_tasks, rotation = 90, fontsize = 7)
        steps = int(r['config'].get('latent', '4x8').split('x')[1])           # time steps per filter
        marks = {u: ' *' if novelty[u] else ('  unused' if not alive[u] else '') for u in norm.index}
        ax.set_yticks(range(len(norm)), [f'f{u // steps} · t{u % steps}{marks[u]}' for u in norm.index[order]], fontsize = 6.5)
        for e in np.cumsum([len(ts) for ts in families.values()])[:-1] - .5:
            ax.axvline(e, color = 'w', lw = 1.5)
        ax.set_title(f'{name}\n* novelty unit: {int(novelty.sum())} of {int(alive.sum())} active', fontsize = 9.5)
        ax.set_ylabel('latent unit')
    fig.subplots_adjust(wspace = .45)
    fig.colorbar(im, ax = axes[0].tolist(), shrink = .6, label = 'mean activation / unit maximum')
    fig.suptitle(title, fontsize = 11)
    plt.show()


def get_balanced_sample(labels: pd.DataFrame, per_task: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Up to ``per_task`` test-participant windows of every task, so no task dominates a neighbourhood or an axis."""
    picks = []
    for _, rows in labels[~labels['is_train']].groupby('task', observed = True).groups.items():
        rows = np.asarray(rows)
        picks.append(rng.choice(rows, min(per_task, len(rows)), replace = False))
    return np.sort(np.concatenate(picks))


def get_views(z: npt.NDArray[np.float32], cache: Path, seed: int = 0) -> dict:
    """2-D and 3-D coordinates of every view. UMAP results are cached by the latent values themselves."""
    units = np.argsort(z.var(axis = 0))[::-1][:3]                     # the three highest-variance latent units
    zs = StandardScaler().fit_transform(z.astype(np.float64))
    views = {}
    for dims in (2, 3):
        views[('latent units', dims)] = z[:, units[:dims]].astype(np.float32)
        views[('PCA', dims)] = PCA(n_components = dims, random_state = seed).fit_transform(zs).astype(np.float32)
        key = hashlib.sha256(zs.tobytes() + f'|umap|{dims}|{seed}'.encode()).hexdigest()[:16]
        path = cache / f'{key}.npy'
        if path.exists():
            points = np.load(path)
        else:
            points = umap.UMAP(n_components = dims, n_neighbors = 30, min_dist = 0.1,
                               random_state = seed).fit_transform(zs).astype(np.float32)
            cache.mkdir(parents = True, exist_ok = True)
            np.save(path, points)
        views[('UMAP', dims)] = points
    views['units'] = units
    return views


def plot_lens_grid(views: dict[str, dict], labels: pd.DataFrame, column: str, palette: dict[str, str], title: str) -> None:
    """Rows: models; columns: the three views in 2-D. Categories outside ``palette`` are drawn grey underneath."""
    names = list(views)
    fig, axes = plt.subplots(len(names), len(VIEWS), figsize = (4.4 * len(VIEWS), 4.0 * len(names)), squeeze = False)
    values = labels[column].astype(str).to_numpy()
    rest = ~np.isin(values, list(palette))
    for row, name in zip(axes, names):
        for ax, view in zip(row, VIEWS):
            pts = views[name][(view, 2)]
            ax.scatter(pts[rest, 0], pts[rest, 1], s = 1.2, color = BACKGROUND, alpha = .5, rasterized = True)
            for lab, color in palette.items():
                m = values == lab
                ax.scatter(pts[m, 0], pts[m, 1], s = 1.8, color = color, alpha = .6, rasterized = True)
            axis = f'units {views[name]["units"][0]} / {views[name]["units"][1]}' if view == 'latent units' else ''
            ax.set_title(f'{name} · {view}', fontsize = 9)
            ax.set_xlabel(axis, fontsize = 8); ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], marker = 'o', ls = '', color = c, label = l, markersize = 6) for l, c in palette.items()]
    if rest.any():
        handles.append(plt.Line2D([], [], marker = 'o', ls = '', color = BACKGROUND, label = 'other', markersize = 6))
    fig.suptitle(title, y = .995, fontsize = 12)
    fig.legend(handles = handles, loc = 'upper center', bbox_to_anchor = (.5, .965), ncol = min(len(handles), 8),
               fontsize = 8.5, frameon = False)
    plt.tight_layout(rect = (0, 0, 1, .93)); plt.show()


def plot_task_panels(points: npt.NDArray[np.float32], labels: pd.DataFrame, families: dict[str, tuple[str, ...]],
                     styles: dict[str, dict[str, str]], title: str) -> None:
    """One panel per family: that family's tasks in colour, every other window grey."""
    fig, axes = plt.subplots(1, len(families), figsize = (4.0 * len(families), 4.1))
    tasks = labels['task'].astype(str).to_numpy()
    for ax, (family, members) in zip(axes, families.items()):
        rest = ~np.isin(tasks, members)
        ax.scatter(points[rest, 0], points[rest, 1], s = 1, color = BACKGROUND, alpha = .4, rasterized = True)
        for task in members:
            m = tasks == task
            ax.scatter(points[m, 0], points[m, 1], s = 2, color = styles[task]['panel'], alpha = .7, label = task, rasterized = True)
        ax.set_title(family, fontsize = 9); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize = 7, markerscale = 4, loc = 'best', frameon = False, handletextpad = .1)
    fig.suptitle(title, fontsize = 11)
    plt.tight_layout(); plt.show()


def build_3d(settings: dict[str, dict], labels: pd.DataFrame, lenses: dict[str, tuple[str, dict[str, str]]],
             task_symbols: dict[str, str], title: str) -> go.Figure:
    """One 3-D scene; the dropdown shows one (input setting, view, lens) at a time. Hover: task, participant, direction.

    ``settings`` maps an input-setting label to that model's views of the same windows (``labels``). For the task lens
    ``palette`` maps each task to its family's colour and ``task_symbols`` to its marker.
    """
    fig, entries = go.Figure(), []
    hover = (labels['task'].astype(str) + ' · ' + labels['subject'].astype(str) + ' · ' + labels['direction'].astype(str)).to_numpy()
    for setting, views in settings.items():
        for view in VIEWS:
            pts = np.round(views[(view, 3)], 3)
            for lens, (column, palette) in lenses.items():
                values = labels[column].astype(str).to_numpy()
                first = len(fig.data)
                rest = ~np.isin(values, list(palette))
                groups = [('other', rest, BACKGROUND)] if rest.any() else []
                groups += [(lab, values == lab, color) for lab, color in palette.items()]
                for lab, m, color in groups:
                    symbol = task_symbols.get(lab, 'circle') if column == 'task' else 'circle'
                    fig.add_trace(go.Scatter3d(x = pts[m, 0], y = pts[m, 1], z = pts[m, 2], mode = 'markers', name = lab,
                                               visible = not entries, text = hover[m], hovertemplate = '%{text}<extra></extra>',
                                               marker = dict(size = 2.4, color = color, symbol = symbol,
                                                             opacity = .35 if lab == 'other' else .75)))
                prefix = f'{setting} · ' if len(settings) > 1 else ''
                entries.append((f'{prefix}{view} · {lens}', range(first, len(fig.data))))
    buttons = []
    for label, idx in entries:
        visible = np.zeros(len(fig.data), dtype = bool)
        visible[list(idx)] = True
        buttons.append(dict(label = label, method = 'update', args = [{'visible': visible.tolist()}, {'title.text': f'{title}: {label}'}]))
    fig.update_layout(title = dict(text = f'{title}: {entries[0][0]}', x = .02), height = 700, template = 'plotly_white',
                      margin = dict(l = 0, r = 0, t = 90, b = 0), legend = dict(itemsizing = 'constant', font = dict(size = 10)),
                      scene = dict(xaxis_title = '1', yaxis_title = '2', zaxis_title = '3'),
                      updatemenus = [dict(buttons = buttons, direction = 'down', x = .02, y = 1.08, xanchor = 'left', yanchor = 'top')])
    return fig
