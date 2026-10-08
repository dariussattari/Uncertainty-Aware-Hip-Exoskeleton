"""Interactive 2-D / 3-D views (Plotly) of sensor windows and learned latents.

Three projections of the same rows sit side by side:

- ``umap``   nonlinear, neighbourhood-preserving; distances and cluster areas are NOT measurements.
- ``pca``    linear, fitted on the same reference rows; axes carry the explained variance.
- ``latent`` two or three raw latent coordinates, no transformation at all: the only view in
             which a distance is a true latent distance, but it shows 2-3 of 32 dimensions.

Read-only presentation code: nothing here trains a model or writes into a run directory.
Every task gets one fixed colour and one fixed marker symbol, regardless of whether a
protocol calls it in-distribution or out-of-distribution.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IPython.display import Markdown, display

from _exploration import ROOT, read_json

TASKS = ('LG', 'RA', 'RD', 'SA', 'SD', 'TR', 'ST')  # fixed order: colours never cycle
TASK_NAMES = dict(LG='level ground', RA='ramp ascent', RD='ramp descent', SA='stair ascent',
                  SD='stair descent', TR='transitions', ST='standing')
# Validated categorical palette, assigned in fixed task order (not by ID/OOD role).
TASK_COLORS = dict(LG='#2a78d6', RA='#eb6834', RD='#1baf7a', SA='#eda100',
                   SD='#e87ba4', TR='#008300', ST='#4a3aa7')
# Secondary encoding so identity is never colour alone; the 3-D subset is what scatter3d supports.
TASK_SYMBOLS_2D = dict(LG='circle', RA='square', RD='diamond', SA='cross',
                       SD='x', TR='triangle-up', ST='star')
TASK_SYMBOLS_3D = dict(LG='circle', RA='square', RD='diamond', SA='cross',
                       SD='x', TR='circle-open', ST='square-open')
# One-hue sequential ramp (light -> dark blue) for continuous values such as variance.
SEQUENTIAL = [[0, '#cde2fb'], [.25, '#86b6ef'], [.5, '#3987e5'], [.75, '#1c5cab'], [1, '#0d366b']]

PROJECTIONS = ('umap', 'pca', 'latent')
PROJECTION_NAMES = {'umap': 'UMAP', 'pca': 'PCA', 'latent': 'latent coordinates'}

HERE = Path(__file__).resolve().parent
CACHE = HERE / '.cache'            # cached UMAP embeddings (ignored by git)
INTERACTIVE = HERE / 'interactive'  # standalone HTML exports (ignored by git)
PLOTLY_CONFIG = {'scrollZoom': True, 'displaylogo': False,
                 'modeBarButtonsToAdd': ['select2d', 'lasso2d', 'resetScale2d']}
# Browsers allow only a handful of live WebGL contexts per page (Chromium dropped the oldest
# after eight in testing; Chrome allows 16), and JupyterLab is ONE page for every open notebook.
# A dropped context leaves a scene with its legend but a blank plot. So 2-D maps are drawn as SVG
# (no WebGL); only 3-D scenes need WebGL, one per projection panel, and the views of a run share
# one figure through buttons rather than multiplying scenes.
WEBGL_2D = False          # True: Scattergl (smoother with >20k points, but costs a context)
# Store only the Plotly MIME bundle (rendered by JupyterLab and VS Code); adding the HTML copy
# doubles every figure's size and the VAE notebook alone reached 100 MB. Viewers without a
# Plotly renderer have the standalone HTML exports under interactive/ instead.
pio.renderers.default = 'plotly_mimetype'
warnings.filterwarnings('ignore', message='IProgress not found')  # umap's tqdm import noise


@dataclass
class Embedding:
    """UMAP coordinates for a subset of rows, with the rows they came from."""
    points: np.ndarray   # (n, dims)
    index: np.ndarray    # row positions into the source feature/metadata arrays
    dims: int
    fitted_on: str
    params: dict
    cache_path: Path | None = None


def stratified_sample(modes, max_points, seed=0, floor=2000):
    """Sample proportionally per task but keep at least `floor` rows of every task.

    Small tasks (standing has 621 test windows, transitions 1,994) keep all rows; the
    large tasks are thinned. Rows are returned in their original order.
    """
    modes = np.asarray(modes)
    if max_points is None or max_points >= len(modes):
        return np.arange(len(modes))
    rng = np.random.default_rng(seed)
    chosen = []
    for task in TASKS:
        rows = np.flatnonzero(modes == task)
        if not len(rows):
            continue
        n = min(len(rows), max(floor, round(max_points * len(rows) / len(modes))))
        chosen.append(rng.choice(rows, n, replace=False))
    return np.sort(np.concatenate(chosen)) if chosen else np.arange(0)


# ----------------------------------------------------------------------------- projections

def _fingerprint(features, fit_index, plot_index, dims, params, cache_key):
    sample = np.asarray(features[np.linspace(0, len(features) - 1, min(len(features), 64)).astype(int)])
    h = hashlib.sha256()
    for part in (cache_key.encode(), json.dumps(params, sort_keys=True).encode(), str(dims).encode(),
                 str(features.shape).encode(), np.ascontiguousarray(sample).tobytes(),
                 np.asarray(fit_index).tobytes(), np.asarray(plot_index).tobytes()):
        h.update(part)
    return h.hexdigest()[:16]


def umap_embedding(features, fit_index, plot_index, dims=2, *, cache_key, seed=0,
                   n_neighbors=15, min_dist=.1, metric='euclidean', use_cache=True, fitted_on=''):
    """Fit UMAP on `features[fit_index]`, then transform `features[plot_index]`.

    Fitting on a reference subset and transforming the rest mirrors the PCA discipline
    already used in the AE figures. Results are cached under `.cache/` because UMAP is
    stochastic and slow; delete the cache to recompute.
    """
    import umap
    if dims not in (2, 3):
        raise ValueError('dims must be 2 or 3')
    fit_index = np.asarray(fit_index, dtype=np.int64)
    plot_index = np.asarray(plot_index, dtype=np.int64)
    if not len(fit_index) or not len(plot_index):
        raise ValueError('Empty reference or plotting index.')
    params = dict(n_neighbors=int(n_neighbors), min_dist=float(min_dist), metric=metric, seed=int(seed))
    key = _fingerprint(features, fit_index, plot_index, dims, params, cache_key)
    path = CACHE / f'{cache_key}_{dims}d_{key}.npz'
    if use_cache and path.exists():
        with np.load(path) as saved:
            return Embedding(saved['points'], saved['index'], dims, str(saved['fitted_on']), params, path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # random_state disables parallelism; that is intended
        reducer = umap.UMAP(n_components=dims, random_state=seed, **{k: v for k, v in params.items() if k != 'seed'})
        reducer.fit(np.asarray(features[fit_index], dtype=np.float32))
        same = len(fit_index) == len(plot_index) and np.array_equal(fit_index, plot_index)
        points = reducer.embedding_ if same else reducer.transform(np.asarray(features[plot_index], dtype=np.float32))
    points = np.asarray(points, dtype=np.float32)
    if use_cache:
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(path, points=points, index=plot_index, fitted_on=fitted_on)
    return Embedding(points, plot_index, dims, fitted_on, params, path if use_cache else None)


def _compact(points, decimals=3):
    """Round plotted coordinates: invisible on screen, much shorter in the saved notebook JSON."""
    return np.round(np.asarray(points, dtype=np.float64), decimals).astype(np.float32)


def latent_axis_label(k, width):
    """Name a raw coordinate; a 32-wide latent is the AE/VAE's 4 filters x 8 time steps."""
    return f'z[{k}]  (filter {k // 8}, t {k % 8})' if width == 32 else f'dim {k}'


def choose_latent_dims(features, fit_index, dims, latent_dims=None):
    """Explicit dimensions, or the `dims` highest-variance ones on the reference rows."""
    if latent_dims is not None:
        chosen = tuple(int(k) for k in latent_dims)
        if len(chosen) < dims or len(set(chosen)) < len(chosen) or max(chosen) >= features.shape[1] or min(chosen) < 0:
            raise ValueError(f'latent_dims must name at least {dims} distinct coordinates below {features.shape[1]}')
        return chosen[:dims]
    var = np.var(np.asarray(features[fit_index], dtype=np.float64), axis=0)
    return tuple(int(k) for k in np.argsort(-var, kind='stable')[:dims])


def project(features, fit_index, plot_index, dims, method, *, cache_key, seed=0, fitted_on='',
            latent_dims=None, pca_variance_scale=1.0, **umap_kwargs):
    """Coordinates of `features[plot_index]` under one projection -> (points, axis labels)."""
    if method == 'umap':
        emb = umap_embedding(features, fit_index, plot_index, dims, cache_key=cache_key, seed=seed,
                             fitted_on=fitted_on, **umap_kwargs)
        return _compact(emb.points), [f'UMAP {i + 1}' for i in range(dims)]
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dims, random_state=seed).fit(np.asarray(features[fit_index], dtype=np.float32))
        points = pca.transform(np.asarray(features[plot_index], dtype=np.float32)).astype(np.float32)
        return _compact(points), [f'PC{i + 1} ({v * pca_variance_scale:.1%})' for i, v in enumerate(pca.explained_variance_ratio_)]
    if method == 'latent':
        ks = choose_latent_dims(features, fit_index, dims, latent_dims)
        points = np.asarray(features[plot_index], dtype=np.float32)[:, list(ks)]
        return _compact(points), [latent_axis_label(k, features.shape[1]) for k in ks]
    raise ValueError(f'unknown projection {method!r}; use one of {PROJECTIONS}')


# ----------------------------------------------------------------------------- figures

def hover_text(meta, columns=('subject', 'trial'), values=None):
    """Compact per-point hover strings (kept short to limit notebook size)."""
    parts = [meta[c].astype(str) for c in columns if c in meta]
    text = parts[0].str.cat(parts[1:], sep=' · ') if parts else pd.Series([''] * len(meta), index=meta.index)
    if values:
        for name, v in values.items():
            text = text + f'<br>{name}: ' + pd.Series(np.asarray(v)).map('{:.4g}'.format).to_numpy()
    return text.to_numpy()


def _base_layout(title, dims, axes=None):
    axes = axes or ([f'axis {i + 1}' for i in range(dims)])
    layout = dict(title=dict(text=title, x=.02), template='plotly_white', height=640,
                  margin=dict(l=40, r=20, t=70, b=40), hovermode='closest',
                  legend=dict(title='Task (click to hide, double-click to isolate)', itemsizing='constant'))
    if dims == 2:
        layout.update(dragmode='pan', xaxis=dict(title=axes[0], zeroline=False, gridcolor='#eeeeee'),
                      yaxis=dict(title=axes[1], zeroline=False, gridcolor='#eeeeee'))
    else:
        layout.update(scene=dict(xaxis_title=axes[0], yaxis_title=axes[1], zaxis_title=axes[2],
                                 aspectmode='cube', dragmode='turntable'))
    return layout


def _trace(points, name, color, symbol, text, dims, size, opacity, legendgroup=None, webgl=None, **marker_extra):
    marker = dict(color=color, symbol=symbol, size=size, opacity=opacity,
                  **{k: v for k, v in marker_extra.items() if v is not None})
    common = dict(mode='markers', name=name, marker=marker, text=text, legendgroup=legendgroup,
                  hovertemplate='%{text}<extra>' + name + '</extra>')
    if dims == 2:
        marker.setdefault('line', dict(width=0))
        cls = go.Scattergl if (WEBGL_2D if webgl is None else webgl) else go.Scatter
        return cls(x=points[:, 0], y=points[:, 1], **common)
    return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], **common)


def scatter_by_task(points, modes, text=None, *, title='', axes=None, size=None, opacity=.75, webgl=None):
    """One trace per task in fixed order; every task keeps its own colour and symbol."""
    points = np.asarray(points); modes = np.asarray(modes); dims = points.shape[1]
    text = np.asarray(text) if text is not None else np.full(len(points), '', dtype=object)
    size = size or (4 if dims == 2 else 2.6)
    symbols = TASK_SYMBOLS_2D if dims == 2 else TASK_SYMBOLS_3D
    fig = go.Figure(layout=_base_layout(title, dims, axes))
    for task in TASKS:
        sel = modes == task
        if not sel.any():
            continue
        fig.add_trace(_trace(points[sel], f'{task} — {TASK_NAMES[task]} (n={int(sel.sum()):,})',
                             TASK_COLORS[task], symbols[task], text[sel], dims, size, opacity,
                             legendgroup=task, webgl=webgl))
    return fig


def scatter_by_value(points, values, modes, text=None, *, title='', axes=None, value_name='value',
                     log10=False, size=None, opacity=.8, value_range=None):
    """Continuous colouring (one-hue ramp); task identity stays available in the hover text."""
    points = np.asarray(points); dims = points.shape[1]
    v = np.asarray(values, dtype=float)
    if log10:
        v = np.log10(np.clip(v, np.finfo(float).tiny, None)); value_name = f'log10 {value_name}'
    lo, hi = (float(v.min()), float(v.max())) if value_range is None else value_range
    modes = np.asarray(modes)
    base = np.asarray(text) if text is not None else np.full(len(points), '', dtype=object)
    text = np.array([f'{m} — {TASK_NAMES.get(m, m)}<br>{t}<br>{value_name}: {x:.4g}'
                     for m, t, x in zip(modes, base, v)], dtype=object)
    size = size or (4 if dims == 2 else 2.6)
    fig = go.Figure(layout=_base_layout(title, dims, axes))
    fig.add_trace(_trace(points, value_name, v, 'circle', text, dims, size, opacity,
                         colorscale=SEQUENTIAL, cmin=lo, cmax=hi, colorbar=dict(title=value_name), showscale=True))
    fig.update_layout(showlegend=False)
    return fig


def _copy(trace):
    return trace.__class__(trace.to_plotly_json())


def compose_2d(views, axes, *, title):
    """One SVG figure: projections as side-by-side panels, views (if several) as buttons.

    `views` = {view name: {projection: Figure}}, `axes` = {projection: [x label, y label]}.
    Legend entries are grouped per task across panels, so hiding a task hides it everywhere.
    """
    projs = list(axes); names = list(views)
    fig = make_subplots(rows=1, cols=len(projs), horizontal_spacing=.06,
                        subplot_titles=[PROJECTION_NAMES.get(p, p) for p in projs])
    groups = []
    for vi, name in enumerate(names):
        idx = []
        for pi, p in enumerate(projs):
            for tr in views[name][p].data:
                tr = _copy(tr)
                tr.visible = (vi == 0)
                if tr.showlegend is not False:
                    tr.showlegend = (pi == 0)
                if getattr(tr.marker, 'showscale', None):
                    tr.marker.showscale = (pi == len(projs) - 1)
                fig.add_trace(tr, row=1, col=pi + 1); idx.append(len(fig.data) - 1)
        groups.append(idx)
    for pi, p in enumerate(projs):
        fig.update_xaxes(title_text=axes[p][0], zeroline=False, gridcolor='#eeeeee', row=1, col=pi + 1)
        fig.update_yaxes(title_text=axes[p][1], zeroline=False, gridcolor='#eeeeee', row=1, col=pi + 1)
    base = _base_layout(title, 2); base.pop('xaxis'); base.pop('yaxis'); base['height'] = 560
    fig.update_layout(**base, showlegend=views[names[0]][projs[0]].layout.showlegend is not False)
    if len(names) > 1:
        buttons = []
        for vi, name in enumerate(names):
            visible = [i in set(groups[vi]) for i in range(len(fig.data))]
            src = views[name][projs[0]]
            buttons.append(dict(label=name, method='update',
                                args=[{'visible': visible},
                                      {'title.text': src.layout.title.text, 'showlegend': src.layout.showlegend is not False}]))
        fig.update_layout(updatemenus=[dict(type='buttons', direction='right', showactive=True,
                                            x=0, xanchor='left', y=1.14, yanchor='top', buttons=buttons)],
                          margin=dict(t=120))
    return fig


def compose_3d(views, axes, *, title):
    """One figure with a WebGL scene per projection side by side; views (if several) as buttons.

    Each scene is its own WebGL context, so a three-projection figure costs three of the
    browser's per-page budget (about 8-16). Keep only the notebook you are reading open.
    """
    projs = list(axes); names = list(views)
    fig = make_subplots(rows=1, cols=len(projs), horizontal_spacing=.02,
                        specs=[[{'type': 'scene'}] * len(projs)],
                        subplot_titles=[PROJECTION_NAMES.get(p, p) for p in projs])
    groups = []
    for vi, name in enumerate(names):
        idx = []
        for pi, p in enumerate(projs):
            for tr in views[name][p].data:
                tr = _copy(tr)
                tr.visible = (vi == 0)
                if tr.showlegend is not False:
                    tr.showlegend = (pi == 0)
                if getattr(tr.marker, 'showscale', None):
                    tr.marker.showscale = (pi == len(projs) - 1)
                fig.add_trace(tr, row=1, col=pi + 1); idx.append(len(fig.data) - 1)
        groups.append(idx)
    for pi, p in enumerate(projs):
        fig.update_layout({f'scene{pi + 1 if pi else ""}': dict(
            xaxis_title=axes[p][0], yaxis_title=axes[p][1], zaxis_title=axes[p][2],
            aspectmode='cube', dragmode='turntable')})
    base = _base_layout(title, 3); base.pop('scene'); base['height'] = 560
    fig.update_layout(**base, showlegend=views[names[0]][projs[0]].layout.showlegend is not False)
    if len(names) > 1:
        buttons = []
        for vi, name in enumerate(names):
            visible = [i in set(groups[vi]) for i in range(len(fig.data))]
            src = views[name][projs[0]]
            buttons.append(dict(label=name, method='update',
                                args=[{'visible': visible},
                                      {'title.text': src.layout.title.text, 'showlegend': src.layout.showlegend is not False}]))
        fig.update_layout(updatemenus=[dict(type='buttons', direction='right', showactive=True,
                                            x=0, xanchor='left', y=1.14, yanchor='top', buttons=buttons)],
                          margin=dict(t=120))
    return fig


def compose_grid(figs, *, cols=3, axes=('UMAP 1', 'UMAP 2'), title, row_height=360):
    """Many 2-D panels (one per model/run) in a grid, one shared legend, one WebGL context."""
    labels = list(figs); rows = -(-len(labels) // cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=labels, horizontal_spacing=.05, vertical_spacing=.11)
    for i, label in enumerate(labels):
        r, c = divmod(i, cols)
        for tr in figs[label].data:
            tr = _copy(tr)
            if tr.showlegend is not False:
                tr.showlegend = (i == 0)
            fig.add_trace(tr, row=r + 1, col=c + 1)
        fig.update_xaxes(title_text=axes[0] if r == rows - 1 else None, zeroline=False, gridcolor='#eeeeee', row=r + 1, col=c + 1)
        fig.update_yaxes(title_text=axes[1] if c == 0 else None, zeroline=False, gridcolor='#eeeeee', row=r + 1, col=c + 1)
    base = _base_layout(title, 2); base.pop('xaxis'); base.pop('yaxis'); base['height'] = 90 + row_height * rows
    fig.update_layout(**base)
    fig.update_annotations(font_size=11)
    return fig


def compose(views, axes, dims, *, title):
    return compose_2d(views, axes, title=title) if dims == 2 else compose_3d(views, axes, title=title)


def show(fig, name=None, save_html=False):
    """Render interactively (scroll = zoom, drag = pan/rotate) and optionally export HTML."""
    fig.show(config=PLOTLY_CONFIG)
    if save_html and name:
        INTERACTIVE.mkdir(exist_ok=True)
        path = INTERACTIVE / f'{name}.html'
        fig.write_html(path, include_plotlyjs='cdn', config=PLOTLY_CONFIG)
        print('Standalone interactive copy:', path.relative_to(ROOT))


def feature_views(features, meta, fit_index, *, plot_index=None, dims=(2, 3), projections=('umap', 'pca'),
                  title, cache_key, seed=0, max_points=12_000, hover_values=None, save_html=False,
                  fitted_on='', show_figures=True, latent_dims=None, pca_variance_scale=1.0, **umap_kwargs):
    """Task-coloured views of one feature matrix under several projections, in 2-D and/or 3-D.

    Returns {dims: {'figure': Figure, 'points': {projection: array}, 'index': rows, 'axes': {...}}}.
    """
    modes = meta['mode'].to_numpy()
    if plot_index is None:
        plot_index = stratified_sample(modes, max_points, seed)
    plot_index = np.asarray(plot_index, dtype=np.int64)
    sub = meta.iloc[plot_index]
    values = {k: np.asarray(v)[plot_index] for k, v in (hover_values or {}).items()}
    text = hover_text(sub, values=values)
    results = {}
    for d in dims:
        figs, axes, points = {}, {}, {}
        for p in projections:
            pts, labels = project(features, fit_index, plot_index, d, p, cache_key=cache_key, seed=seed,
                                  fitted_on=fitted_on, latent_dims=latent_dims,
                                  pca_variance_scale=pca_variance_scale, **umap_kwargs)
            figs[p] = scatter_by_task(pts, modes[plot_index], text, axes=labels, title=title)
            axes[p] = labels; points[p] = pts
        fig = compose({'windows': figs}, axes, d,
                      title=f'{title} — {d}D ({len(plot_index):,} windows; fitted on {fitted_on or "the plotted rows"})')
        if show_figures:
            show(fig, f'{cache_key}_views_{d}d', save_html)
        results[d] = dict(figure=fig, points=points, index=plot_index, axes=axes)
    return results


def task_counts(meta, index=None):
    modes = meta['mode'].to_numpy() if index is None else meta['mode'].to_numpy()[index]
    return pd.DataFrame({'task': TASKS, 'name': [TASK_NAMES[t] for t in TASKS],
                         'windows': [int(np.sum(modes == t)) for t in TASKS]}).set_index('task')


# ----------------------------------------------------------------------------- data sources

def sensor_windows(split='test', root=None, max_points=12_000, seed=0, pca_components=50):
    """All-task sensor windows from the mode-complete set, standardized and PCA-reduced.

    Uses `data/processed_all/AE_GAN` (all seven tasks, no ID/OOD partition on disk) and its
    scaler, which was fitted on LG/RA/RD training windows only. A stratified sample is read
    from the memory-mapped array; each 16×200 window is flattened to 3,200 values and reduced
    to `pca_components` PCs (fitted on the sample) before UMAP, the usual UMAP preprocessing.
    The `pca` view's first components are exactly those PCs; `explained` rescales their labels
    to shares of the raw window variance.
    """
    from sklearn.decomposition import PCA
    base = Path(root) if root else ROOT / 'data/processed_all/AE_GAN'
    meta = pd.read_parquet(base / f'{split}_meta.parquet')
    index = stratified_sample(meta['mode'].to_numpy(), max_points, seed)
    X = np.load(base / f'{split}_X.npy', mmap_mode='r')
    scaler = np.load(base / 'scaler.npz')
    windows = (np.asarray(X[index], dtype=np.float32) - scaler['mean'][None, :, None].astype(np.float32)) \
        / scaler['std'][None, :, None].astype(np.float32)
    flat = windows.reshape(len(index), -1)
    pca = PCA(n_components=min(pca_components, flat.shape[1], len(flat)), random_state=seed).fit(flat)
    return dict(features=pca.transform(flat).astype(np.float32), meta=meta.iloc[index].reset_index(drop=True),
                source_index=index, split=split, explained=float(pca.explained_variance_ratio_.sum()),
                channels=[str(c) for c in scaler['channels']], config=read_json(base / 'config.json'))


# ----------------------------------------------------------------------------- pelvis IMU variant

RAW = ROOT / 'data/molinaro-raw'
PELVIS_CHANNELS = ('pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z',
                   'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z')
SCALER_MODES = ('LG', 'RA', 'RD')   # the modes the processed set's scaler was fitted on
FEATURE_SETS = {'thigh': 'the processed 16 channels: hip encoder angle + velocity and thigh IMU, both legs',
                'pelvis': 'the 6 pelvis IMU channels only (accelerometer + gyroscope)',
                'thigh+pelvis': 'all 22 channels'}


def pelvis_scaler(base, raw_root=None):
    """Per-channel mean and std of the pelvis IMU, fitted the way the processed scaler was.

    The processed 16-channel scaler used the training participants' LG/RA/RD windows only, so
    that scaling never depends on which task a run holds out. The same trials are read here
    (every non-NaN row rather than every stride-10 window, which weights rows equally instead
    of ~20 times each; the difference is at the edges of a trial).
    """
    raw_root = Path(raw_root) if raw_root else RAW
    meta = pd.read_parquet(Path(base) / 'train_meta.parquet')
    trials = meta.loc[meta['mode'].isin(SCALER_MODES), ['subject', 'trial']].drop_duplicates()
    cols = list(PELVIS_CHANNELS)
    n, s, ss = 0, np.zeros(6), np.zeros(6)
    for r in trials.itertuples(index=False):
        x = pd.read_csv(raw_root / r.subject / r.trial / 'exo.csv', usecols=cols)[cols].to_numpy(np.float64)
        x = x[~np.isnan(x).any(axis=1)]
        n += len(x); s += x.sum(0); ss += (x * x).sum(0)
    if n == 0:
        raise ValueError('no pelvis rows found for the scaler trials')
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32), channels=cols,
                trials=int(len(trials)), rows=int(n))


def pelvis_windows(meta, raw_root=None, window=200):
    """(n, 6, window) raw pelvis IMU windows for metadata rows (subject, trial, end_idx).

    The processed windows are plain slices of the raw 200 Hz trials (rows end_idx-199 ... end_idx,
    no filtering), so the pelvis channels of the same window are the same slice of exo.csv.
    Windows containing NaN are returned as NaN; the caller decides what to drop.
    """
    raw_root = Path(raw_root) if raw_root else RAW
    meta = meta.reset_index(drop=True)
    cols = list(PELVIS_CHANNELS)
    out = np.full((len(meta), len(cols), window), np.nan, np.float32)
    for (subject, trial), g in meta.groupby(['subject', 'trial'], sort=False):
        raw = pd.read_csv(raw_root / subject / trial / 'exo.csv', usecols=cols)[cols].to_numpy(np.float32)
        ends = g['end_idx'].to_numpy(np.int64)
        if ends.max() >= len(raw) or ends.min() < window - 1:
            raise ValueError(f'{subject}/{trial}: end_idx outside the raw trial ({len(raw)} rows)')
        idx = (ends - window + 1)[:, None] + np.arange(window)[None, :]
        out[g.index.to_numpy()] = np.transpose(raw[idx], (0, 2, 1))
    return out


def sensor_window_sets(split='test', sets=('thigh', 'pelvis', 'thigh+pelvis'), root=None, raw_root=None,
                       max_points=12_000, seed=0, pca_components=50):
    """The same sampled windows under several channel sets, each standardized, flattened and PCA-reduced.

    One stratified sample is drawn exactly as `sensor_windows` draws it, so the rows of every set
    are the same windows in the same order; a window whose pelvis channels contain NaN is dropped
    from every set. Returns {set: bundle} with the keys `sensor_windows` returns plus `channels`.
    """
    from sklearn.decomposition import PCA
    unknown = set(sets) - set(FEATURE_SETS)
    if unknown:
        raise ValueError(f'unknown feature sets {sorted(unknown)}; choose from {list(FEATURE_SETS)}')
    base = Path(root) if root else ROOT / 'data/processed_all/AE_GAN'
    meta = pd.read_parquet(base / f'{split}_meta.parquet')
    index = stratified_sample(meta['mode'].to_numpy(), max_points, seed)
    scaler = np.load(base / 'scaler.npz')
    X = np.load(base / f'{split}_X.npy', mmap_mode='r')
    thigh = (np.asarray(X[index], dtype=np.float32) - scaler['mean'][None, :, None].astype(np.float32)) \
        / scaler['std'][None, :, None].astype(np.float32)
    thigh_channels = [str(c) for c in scaler['channels']]
    config = read_json(base / 'config.json')
    window = int(config.get('window_samples', X.shape[-1]))
    sub = meta.iloc[index].reset_index(drop=True)
    blocks = {'thigh': (thigh, thigh_channels)}
    dropped = 0
    if any(s != 'thigh' for s in sets):
        ps = pelvis_scaler(base, raw_root)
        pelvis = (pelvis_windows(sub, raw_root, window) - ps['mean'][None, :, None]) / ps['std'][None, :, None]
        bad = np.isnan(pelvis).any(axis=(1, 2))
        dropped = int(bad.sum())
        if dropped:
            keep = ~bad
            index, sub, thigh, pelvis = index[keep], sub[keep].reset_index(drop=True), thigh[keep], pelvis[keep]
            blocks['thigh'] = (thigh, thigh_channels)
        blocks['pelvis'] = (pelvis, ps['channels'])
        blocks['thigh+pelvis'] = (np.concatenate([thigh, pelvis], axis=1), thigh_channels + ps['channels'])
    out = {}
    for name in sets:
        windows, channels = blocks[name]
        flat = windows.reshape(len(windows), -1)
        pca = PCA(n_components=min(pca_components, flat.shape[1], len(flat)), random_state=seed).fit(flat)
        out[name] = dict(features=pca.transform(flat).astype(np.float32), meta=sub.copy(), source_index=index,
                         split=split, explained=float(pca.explained_variance_ratio_.sum()), channels=list(channels),
                         feature_set=name, dropped=dropped, config=config)
    return out


HELD_OUT = {'participant': 'subject', 'trial': 'trial'}


def knn_task_scores(points, meta, k=15, seed=0, held_out='participant', max_folds=5):
    """k-NN task classification in a coordinate space with whole recordings held out.

    Windows from one trial overlap by 95 %, so a random split would let a window label its own
    neighbours; folds are whole participants (`held_out='participant'`, one fold each) or whole
    trials (`held_out='trial'`, up to `max_folds` folds). A task recorded by fewer than two of the
    held-out units cannot be scored under that scheme and gets recall NaN; balanced accuracy
    averages the tasks that could be scored.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if held_out not in HELD_OUT:
        raise ValueError(f'held_out must be one of {list(HELD_OUT)}')
    points = np.asarray(points, dtype=np.float32)
    modes = meta['mode'].to_numpy()
    groups = meta[HELD_OUT[held_out]].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError(f'holding out {held_out}s needs at least two of them')
    pred = np.empty(len(modes), dtype=object)
    for tr, te in GroupKFold(n_splits=min(n_groups, max_folds)).split(points, modes, groups):
        clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(points[tr], modes[tr])
        pred[te] = clf.predict(points[te])
    units = pd.Series(groups).groupby(modes).nunique()
    recall = {t: (float(np.mean(pred[modes == t] == t)) if units.get(t, 0) >= 2 else float('nan'))
              for t in TASKS if np.any(modes == t)}
    scored = [v for v in recall.values() if np.isfinite(v)]
    return dict(accuracy=float(np.mean(pred == modes)), balanced_accuracy=float(np.mean(scored)) if scored else float('nan'),
                recall=recall, folds=int(min(n_groups, max_folds)), held_out=held_out)


def silhouette_by_task(points, meta, max_points=5000, seed=0):
    """Silhouette of the task labelling in a coordinate space, on a random subset (O(n^2))."""
    from sklearn.metrics import silhouette_score
    points = np.asarray(points, dtype=np.float32)
    modes = meta['mode'].to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(points), min(max_points, len(points)), replace=False)
    if len(np.unique(modes[pick])) < 2:
        return float('nan')
    return float(silhouette_score(points[pick], modes[pick]))


def separability(named, meta, k=15, seed=0, held_out=('participant', 'trial')):
    """Task separability of several feature sets on the same rows.

    `named` maps a set name to {'features': (n, p) PCA features, 'umap': {dims: (n, dims) points}}.
    Returns (summary, per_task). `summary` has one row per (feature set, held-out unit): k-NN
    accuracy and balanced accuracy in the PCA space, k-NN accuracy in each UMAP embedding, and the
    UMAP silhouettes (fold-free, so equal across the units). `per_task` has the PCA-space recall of
    every task, tasks as rows and (held-out unit, feature set) as columns.
    """
    rows, recall = {}, {}
    for name, item in named.items():
        if len(item['features']) != len(meta):
            raise ValueError(f'{name}: {len(item["features"])} rows of features for {len(meta)} metadata rows')
        umap = item.get('umap') or {}
        sil = {d: silhouette_by_task(pts, meta, seed=seed) for d, pts in sorted(umap.items())}
        for unit in held_out:
            base = knn_task_scores(item['features'], meta, k, seed, unit)
            row = {f'kNN accuracy, PCA-{item["features"].shape[1]}': base['accuracy'],
                   'balanced accuracy, PCA': base['balanced_accuracy']}
            for d, pts in sorted(umap.items()):
                row[f'kNN accuracy, UMAP-{d}D'] = knn_task_scores(pts, meta, k, seed, unit)['accuracy']
                row[f'silhouette, UMAP-{d}D'] = sil[d]
            rows[(name, unit)] = row
            recall[(unit, name)] = base['recall']
    summary = pd.DataFrame(rows).T
    summary.index.names = ['feature set', 'held out']
    per_task = pd.DataFrame(recall)
    per_task.columns.names = ['held out', 'feature set']
    per_task.index.name = 'task'
    return summary, per_task

def ae_latents(run):
    """Saved AE latents (`latents_test.npz` + parquet) from an existing evaluation run."""
    run = Path(run)
    if not (run / 'latents_test.npz').exists():
        return None
    with np.load(run / 'latents_test.npz') as z:
        data = {k: z[k] for k in z.files}
    meta = pd.read_parquet(run / 'latents_test.parquet')
    if len(meta) != len(data['Z']):
        raise ValueError(f'{run}: latent/metadata lengths differ.')
    return dict(Z=data['Z'], meta=meta, labels=data['labels'], psi_smooth=data.get('psi_smooth'))


def ae_latent_views(run, *, dims=(2, 3), projections=PROJECTIONS, latent_dims=None, max_points=12_000,
                    seed=0, reference_points=15_000, save_html=False, show_figures=True, title='AE latent (32-D)'):
    """AE latent maps: UMAP and PCA fitted on in-distribution latents, all tasks projected into them,
    plus the raw latent coordinates.

    This mirrors `ae_latent_space.png`, which fits PCA on ID latents only, so the question
    "do held-out tasks land off the learned manifold?" is asked the same way. Colours are
    by task, never by ID/OOD role.
    """
    bundle = ae_latents(run)
    if bundle is None:
        print(f'No saved latents in {run}; nothing to embed.'); return None
    rng = np.random.default_rng(seed)
    id_rows = np.flatnonzero(bundle['labels'] == 0)
    fit = np.sort(rng.choice(id_rows, min(reference_points, len(id_rows)), replace=False))
    hover = {'LOF score': bundle['psi_smooth']} if bundle['psi_smooth'] is not None else None
    return feature_views(bundle['Z'], bundle['meta'], fit, dims=dims, projections=projections, title=title,
                         cache_key=f'ae_{Path(run).name}', seed=seed, max_points=max_points,
                         hover_values=hover, save_html=save_html, latent_dims=latent_dims,
                         fitted_on=f'{len(fit):,} in-distribution latents', show_figures=show_figures)


def vae_latents(run):
    """Posterior means, log variances (if saved) and the training reference means."""
    run = Path(run)
    if not (run / 'latents_test.npz').exists():
        return None
    with np.load(run / 'latents_test.npz') as z:
        data = {k: z[k] for k in z.files}
    meta = pd.read_parquet(run / 'latents_test.parquet')
    reference = np.load(run / 'train_reference_mu.npy') if (run / 'train_reference_mu.npy').exists() else None
    return dict(mu=data['Z'], logvar=data.get('logvar'), meta=meta, labels=data['labels'],
                lof=data.get('smooth_latent_lof'), reference=reference)


def sample_latents(mu, logvar, n_samples, seed=0):
    """Reparameterized draws z = mu + exp(logvar/2) * eps, shape (n_samples, n, d)."""
    rng = np.random.default_rng(seed)
    sigma = np.exp(.5 * np.asarray(logvar, dtype=np.float32))
    return mu[None] + sigma[None] * rng.standard_normal((n_samples, *mu.shape), dtype=np.float32)


def vae_latent_views(run, *, dims=(2, 3), projections=PROJECTIONS, latent_dims=None, max_points=8_000,
                     seed=0, reference_points=15_000, sample_windows=1_500, samples_per_window=4,
                     cloud_windows=2, cloud_samples=150, save_html=False, show_figures=True):
    """Three views of one trained VAE run under each projection, in 2-D and/or 3-D:

    1. **means**    posterior means μ(x), coloured by task.
    2. **samples**  z ~ q(z|x): several reparameterized draws per window, plus dense "clouds"
                    around a few windows per task to show the local spread of q(z|x).
    3. **variance** the means coloured by the average σ²(x) over the 32 latent dimensions
                    (log10). This is the learned variance itself; a cloud's footprint in a
                    projection is only a distorted picture of it.

    UMAP and PCA are fitted on the training reference means and everything else is transformed
    into that fixed map; the latent view is the raw coordinates. In 2-D the projections are
    side-by-side panels with view buttons, in 3-D one WebGL scene per panel.
    """
    bundle = vae_latents(run)
    if bundle is None:
        print(f'No trained VAE representation in {run}. Enable training above, then rerun.'); return None
    mu, logvar, meta = bundle['mu'], bundle['logvar'], bundle['meta']
    modes = meta['mode'].to_numpy()
    rng = np.random.default_rng(seed)
    reference = bundle['reference'] if bundle['reference'] is not None else mu[bundle['labels'] == 0]
    fitted_on = 'training reference means' if bundle['reference'] is not None else 'in-distribution test means'
    # Stack reference + test means + samples into one feature matrix so a single fit serves all views.
    plot = stratified_sample(modes, max_points, seed)
    ref_rows = np.sort(rng.choice(len(reference), min(reference_points, len(reference)), replace=False))
    blocks = [reference[ref_rows], mu[plot]]
    offsets = {'reference': (0, len(ref_rows)), 'mu': (len(ref_rows), len(ref_rows) + len(plot))}
    sampled_rows = clouds = None
    if logvar is None:
        print('This run predates variance saving: sampled-z and variance views are unavailable. Retrain to add them.')
    else:
        sampled_rows = stratified_sample(modes, sample_windows, seed + 1, floor=400)
        z = sample_latents(mu[sampled_rows], logvar[sampled_rows], samples_per_window, seed).reshape(-1, mu.shape[1])
        start = sum(len(b) for b in blocks); blocks.append(z); offsets['z'] = (start, start + len(z))
        clouds = []
        for task in TASKS:
            rows = np.flatnonzero(modes == task)
            for r in rng.choice(rows, min(cloud_windows, len(rows)), replace=False):
                clouds.append((task, int(r)))
        cloud_z = np.concatenate([sample_latents(mu[[r]], logvar[[r]], cloud_samples, seed + r)[:, 0] for _, r in clouds])
        start = sum(len(b) for b in blocks); blocks.append(cloud_z); offsets['cloud'] = (start, start + len(cloud_z))
    features = np.concatenate(blocks).astype(np.float32)
    fit = np.arange(*offsets['reference']); everything = np.arange(len(features))
    key = f'vae_{Path(run).name}'
    sub = meta.iloc[plot]
    hover = hover_text(sub, values={'LOF': bundle['lof'][plot]} if bundle['lof'] is not None else None)
    var = np.exp(logvar[plot]).mean(1) if logvar is not None else None
    results = {}
    for d in dims:
        views = {'means': {}}; axes = {}; points = {}
        if logvar is not None:
            views['samples'] = {}; views['variance'] = {}
        for p in projections:
            pts, labels = project(features, fit, everything, d, p, cache_key=key, seed=seed,
                                  fitted_on=fitted_on, latent_dims=latent_dims)
            axes[p] = labels; points[p] = pts
            a, b = offsets['mu']; mu_pts = pts[a:b]
            views['means'][p] = scatter_by_task(mu_pts, modes[plot], hover, axes=labels,
                                                title=f'VAE posterior means μ(x) — {d}D ({len(plot):,} windows; fitted on {fitted_on})')
            if logvar is None:
                continue
            a, b = offsets['z']; z_pts = pts[a:b]
            z_modes = np.tile(modes[sampled_rows], samples_per_window)
            fig = scatter_by_task(z_pts, z_modes, None, axes=labels, opacity=.45,
                                  title=f'Sampled latent vectors z ~ q(z|x) — {d}D '
                                        f'({samples_per_window} draws × {len(sampled_rows):,} windows; clouds = {cloud_samples} draws around one window)')
            a, b = offsets['cloud']; cloud_pts = pts[a:b]
            for i, (task, r) in enumerate(clouds):
                seg = cloud_pts[i * cloud_samples:(i + 1) * cloud_samples]
                label = f'{meta.iloc[r]["subject"]} · {meta.iloc[r]["trial"]}'
                fig.add_trace(_trace(seg, f'cloud: {task} {label} ({cloud_samples} draws)', TASK_COLORS[task], 'circle',
                                     np.full(len(seg), '', dtype=object),
                                     d, 5 if d == 2 else 3, .5, legendgroup=f'cloud {task} {r}',
                                     line=dict(width=1, color='black') if d == 2 else None))
            views['samples'][p] = fig
            views['variance'][p] = scatter_by_value(mu_pts, var, modes[plot], hover_text(sub), axes=labels, log10=True,
                                                    value_name='mean posterior σ²',
                                                    title=f'Posterior means coloured by learned variance — {d}D')
        fig = compose(views, axes, d, title=views['means'][projections[0]].layout.title.text)
        if show_figures:
            show(fig, f'{key}_views_{d}d', save_html)
        results[d] = dict(figure=fig, views=views, points=points, axes=axes, offsets=offsets, plot_rows=plot)
    if logvar is not None:
        var_all = np.exp(logvar).mean(1)
        table = pd.DataFrame({'task': TASKS, 'name': [TASK_NAMES[t] for t in TASKS],
                              'windows': [int(np.sum(modes == t)) for t in TASKS],
                              'median mean σ²': [float(np.median(var_all[modes == t])) if np.any(modes == t) else np.nan for t in TASKS],
                              'median ‖μ‖': [float(np.median(np.linalg.norm(mu[modes == t], axis=1))) if np.any(modes == t) else np.nan for t in TASKS]})
        display(Markdown('**Learned posterior variance by task** (exact values from all saved test windows, not from any projection):'))
        display(table.set_index('task').round(4))
    return results


def compose_3d_selector(figs, *, title, axes=('UMAP 1', 'UMAP 2', 'UMAP 3')):
    """ONE WebGL scene showing one of several 3-D figures at a time, chosen from a dropdown.

    Used where many runs must be compared: a scene per run would cost a WebGL context each.
    """
    labels = list(figs)
    fig = go.Figure(layout=_base_layout(title, 3, list(axes)))
    groups = []
    for i, label in enumerate(labels):
        idx = []
        for tr in figs[label].data:
            tr = _copy(tr); tr.visible = (i == 0)
            fig.add_trace(tr); idx.append(len(fig.data) - 1)
        groups.append(set(idx))
    buttons = [dict(label=label, method='update',
                    args=[{'visible': [j in groups[i] for j in range(len(fig.data))]},
                          {'title.text': f'{title}<br><sup>{label}</sup>'}])
               for i, label in enumerate(labels)]
    fig.update_layout(title_text=f'{title}<br><sup>{labels[0]}</sup>',
                      updatemenus=[dict(type='dropdown', direction='down', showactive=True, x=0, xanchor='left',
                                        y=1.12, yanchor='top', buttons=buttons)], margin=dict(t=120))
    return fig


def _run_label(run):
    """'β=4 · val AUROC 0.706 · 18 active dims' from a run directory (controls named as such)."""
    report = read_json(run / 'report.json'); cfg = report.get('config', {})
    val = read_json(run / 'eval_val.json').get('scores', {}).get('latent_lof', {}).get('auroc')
    head = f"β={cfg.get('beta'):g}" if cfg.get('sample', True) else f"β=0 no sampling (lr {cfg.get('lr')}, batch {cfg.get('batch_size')})"
    parts = [head]
    if val is not None:
        parts.append(f'val AUROC {val:.3f}')
    with np.load(run / 'latents_val.npz') as z:
        if 'logvar' in z.files and cfg.get('sample', True):
            var = np.exp(z['logvar'][z['labels'] == 0].astype(np.float64))
            parts.append(f'{int((np.median(var, 0) < .9).sum())} active dims')
    return ' · '.join(parts)


def sweep_mean_views(runs, *, dims=(2, 3), max_points=4_000, seed=0, reference_points=15_000,
                     cols=3, save_html=False, show_figures=True, name='vae_sweep_means'):
    """UMAP of the posterior means of SEVERAL runs side by side, coloured by task.

    Each panel is its own embedding: UMAP fitted on that run's training reference means, its
    test means transformed in. Coordinates are therefore not comparable across panels (each map
    has its own orientation and scale); what is comparable is structure: how far apart the tasks
    sit, and which ones overlap. 2-D: a grid of panels (WebGL scatter, sharing one context);
    3-D: one scene with a run selector.
    """
    runs = [Path(r) for r in runs]
    loaded = {}
    for run in runs:
        bundle = vae_latents(run)
        if bundle is None or bundle['reference'] is None:
            print(f'skipping {run.name}: no saved latents'); continue
        loaded[run] = bundle
    if not loaded:
        print('No runs with saved latents.'); return None
    rng = np.random.default_rng(seed)
    results = {}
    for d in dims:
        figs = {}
        for run, bundle in loaded.items():
            mu, meta = bundle['mu'], bundle['meta']; modes = meta['mode'].to_numpy()
            plot = stratified_sample(modes, max_points, seed, floor=600)
            ref = bundle['reference']; ref_rows = np.sort(rng.choice(len(ref), min(reference_points, len(ref)), replace=False))
            features = np.concatenate([ref[ref_rows], mu[plot]]).astype(np.float32)
            emb = umap_embedding(features, np.arange(len(ref_rows)), np.arange(len(ref_rows), len(features)), d,
                                 cache_key=f'vae_{run.name}_sweepgrid', seed=seed, fitted_on='training reference means')
            hover = hover_text(meta.iloc[plot], values={'LOF': bundle['lof'][plot]} if bundle['lof'] is not None else None)
            figs[_run_label(run)] = scatter_by_task(emb.points, modes[plot], hover, axes=[f'UMAP {i + 1}' for i in range(d)],
                                                    title=_run_label(run), size=3 if d == 2 else 2.2, opacity=.6, webgl=(d == 2))
        cap = 'all' if max_points is None else f'≤{max_points:,}'
        title = f'VAE posterior means per run — UMAP {d}D, each panel its own embedding ({len(figs)} runs, {cap} test windows each)'
        if d == 2:
            fig = compose_grid(figs, cols=cols, title=title)
        else:
            fig = compose_3d_selector(figs, title=title)
        if show_figures:
            show(fig, f'{name}_{d}d', save_html)
        results[d] = dict(figure=fig, panels=list(figs))
    return results
