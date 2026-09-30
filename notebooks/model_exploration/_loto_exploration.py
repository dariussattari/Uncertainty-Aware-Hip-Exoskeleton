"""Read-only LOTO distributions and exploratory threshold what-ifs.

Use saved, already-smoothed scores; never refit a model or persist a new threshold.
"""
from dataclasses import dataclass
from pathlib import Path
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from _exploration import ROOT, LOTO, read_json

TASKS = ('LG', 'RA', 'RD', 'SA', 'SD', 'TR', 'ST')
HOLDOUTS = ('SA', 'ST', 'TR')
NAMES = dict(LG='level ground', RA='ramp ascent', RD='ramp descent',
             SA='stair ascent', SD='stair descent', TR='transitions', ST='standing')
BLUE, ORANGE, PURPLE = '#287da8', '#dd791a', '#8956ad'


@dataclass
class LotoScores:
    key: str
    held_out: str
    score: str
    values: np.ndarray
    labels: np.ndarray
    modes: np.ndarray
    threshold: float
    run: Path


def threshold_rates(bundle, thresholds):
    """Exact percentages for score > threshold, including ties (no KDE estimates)."""
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
    if not np.isfinite(thresholds).all():
        raise ValueError('Thresholds must be finite.')
    result = {'threshold': thresholds}
    groups = {'OOD recall (%)': bundle.labels == 1,
              'ID false alarms (%)': bundle.labels == 0}
    groups.update({f'{task} flagged (%)': bundle.modes == task for task in TASKS})
    for name, mask in groups.items():
        values = np.sort(bundle.values[mask])
        result[name] = (100 * (len(values) - np.searchsorted(values, thresholds, side='right'))
                        / len(values) if len(values) else np.full(len(thresholds), np.nan))
    return pd.DataFrame(result)


def load_loto_scores(key, held_out, score='primary', run=None):
    """Load a run and check task alignment against labels and saved per-task metrics."""
    if held_out not in HOLDOUTS:
        raise ValueError(f'Unknown holdout: {held_out}')
    if run is None:
        if key not in LOTO:
            raise FileNotFoundError(f'No registered LOTO run directory for {key}.')
        run = ROOT / 'ml/LOTO' / held_out / LOTO[key]
    run = Path(run)
    result = read_json(run / 'eval_test.json')
    if not result:
        raise FileNotFoundError(f'No saved evaluation: {run}')
    if result.get('held_out', held_out) != held_out:
        raise ValueError(f'{run}: saved held_out disagrees with requested holdout.')
    if 'scores' in result:
        if score == 'primary':
            score = result.get('primary_score', 'epistemic')
        metrics = result['scores'][score]
        field = f'smooth_{score}'
        threshold = float(metrics['threshold'])
    else:
        if score != 'primary':
            raise ValueError(f'{key} exposes only the primary saved score here.')
        metrics = result['metrics']
        field = 'psi_smooth' if key in ('autoencoder', 'gan') else 'scores'
        threshold = float(result['threshold'])
    stem = 'latents_test' if key in ('autoencoder', 'vae') else 'scores_test'
    with np.load(run / f'{stem}.npz', allow_pickle=False) as saved:
        values = np.asarray(saved[field], dtype=float)
        labels = saved['labels'].copy()
    metadata = run / f'{stem}.parquet'
    if metadata.exists():
        meta = pd.read_parquet(metadata)
    else:
        # Ensemble evaluation writes ID rows, then OOD rows, in original row order.
        data_root = Path(os.environ.get('HIPEXO_LOTO_DATA') or ROOT / 'data/processed_all')
        family = 'ensemble' if key == 'ensemble' else 'AE_GAN'
        meta = pd.read_parquet(data_root / family / 'test_meta.parquet')
        held = meta['mode'].eq(held_out)
        meta = pd.concat([meta.loc[~held], meta.loc[held]], ignore_index=True)
    modes = meta['mode'].to_numpy()
    if values.ndim != 1 or labels.shape != values.shape or len(meta) != len(values):
        raise ValueError(f'{run}: score/label/metadata lengths do not match.')
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError(f'{run}: non-finite scores or threshold.')
    if not set(modes).issubset(TASKS) or not np.array_equal(labels, (modes == held_out).astype(int)):
        raise ValueError(f'{run}: labels must mark ONLY {held_out} as OOD.')
    if not (np.any(labels == 0) and np.any(labels == 1)):
        raise ValueError(f'{run}: both ID and OOD windows are required.')
    bundle = LotoScores(key, held_out, score, values, labels, modes, threshold, run)
    computed = threshold_rates(bundle, [threshold]).iloc[0]
    for column, expected in [('OOD recall (%)', metrics['recall']),
                             ('ID false alarms (%)', 100 - metrics['specificity'])]:
        if not np.isclose(computed[column], expected, atol=1e-6):
            raise ValueError(f'{run}: reconstructed {column} differs from saved evaluation.')
    per_mode = result.get('per_mode', [])
    if isinstance(per_mode, dict):
        per_mode = per_mode[score]
    elif 'scores' in result and score != result.get('primary_score', 'epistemic'):
        per_mode = []  # VAE stores per-mode summaries for its primary score only.
    for row in per_mode:
        task = row['mode']
        sample = values[modes == task]
        median = row.get('median_psi', row.get('median'))
        flagged = row.get('flagged_%', row.get('flagged_LOF_%'))
        if (len(sample) != row['windows'] or row['kind'] != ('OOD' if task == held_out else 'ID')
                or (median is not None and not np.isclose(np.median(sample), median, rtol=1e-6))
                or (flagged is not None and not np.isclose(computed[f'{task} flagged (%)'], flagged))):
            raise ValueError(f'{run}: {task} metadata does not reproduce saved per-task results.')
    return bundle


def plot_loto_scores(bundle, candidate=None):
    """One holdout: task distributions and exact empirical threshold-response curves."""
    candidate = bundle.threshold if candidate is None else float(candidate)
    threshold_rates(bundle, [candidate])  # Validate before drawing.
    use_log = (bundle.key != 'gan' and np.all(bundle.values > 0)
               and min(candidate, bundle.threshold) > 0)
    transform = np.log10 if use_log else np.asarray
    xlabel = r'$\log_{10}(\widetilde\Psi)$' if use_log else r'Smoothed score $\widetilde\Psi$'
    fig, (ax, sweep) = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw={'width_ratios': [1.2, 1]})
    for i, task in enumerate(TASKS):
        values = bundle.values[bundle.modes == task]
        color = ORANGE if task == bundle.held_out else BLUE
        if not len(values):
            continue
        # Deterministic quantile thinning affects density drawing only, never rates/quartiles.
        points = transform(np.quantile(values, np.linspace(0, 1, min(1500, len(values)))))
        if np.ptp(points) > 0:
            violin = ax.violinplot([points], positions=[i], orientation='horizontal', widths=.76,
                                    showextrema=False, bw_method=.18)
            violin['bodies'][0].set(facecolor=color, edgecolor=color, alpha=.65)
        low, median, high = transform(np.quantile(values, [.25, .5, .75]))
        ax.plot([low, high], [i, i], color='#222222', linewidth=3, solid_capstyle='butt')
        ax.plot(median, i, '|', color='black', markersize=11, markeredgewidth=1.8)
    ticks = [f'{t} — {NAMES[t]}' + (' (held out)' if t == bundle.held_out else '') for t in TASKS]
    ax.set(yticks=range(len(TASKS)), yticklabels=ticks, xlabel=xlabel,
           title='Test score spread by task')
    ax.invert_yaxis()
    for tick, task in zip(ax.get_yticklabels(), TASKS):
        tick.set_color(ORANGE if task == bundle.held_out else BLUE)
    # Full unique score support: the staircase is not estimated from the violins.
    minimum = np.nextafter(bundle.values.min(), -np.inf)
    grid = np.unique(np.r_[minimum, bundle.values, bundle.threshold, candidate])
    rates = threshold_rates(bundle, grid)
    sweep.step(transform(grid), rates['OOD recall (%)'], where='post', color=ORANGE, label='Held-out recall')
    sweep.step(transform(grid), rates['ID false alarms (%)'], where='post', color=BLUE, label='ID false alarms')
    markers = [(bundle.threshold, 'black', '--', f'Saved τ = {bundle.threshold:.6g}')]
    if candidate != bundle.threshold:
        markers.append((candidate, PURPLE, ':', f'What-if τ = {candidate:.6g}'))
    for threshold, color, style, label in markers:
        for axis in (ax, sweep):
            axis.axvline(transform(threshold), color=color, ls=style, lw=1.4)
        point = threshold_rates(bundle, [threshold]).iloc[0]
        sweep.scatter([transform(threshold)] * 2,
                      [point['OOD recall (%)'], point['ID false alarms (%)']],
                      c=[ORANGE, BLUE], edgecolors=color, s=38, zorder=4)
    legend = [Line2D([], [], color=c, ls=s, label=label) for _, c, s, label in markers]
    ax.legend(handles=legend, loc='best', fontsize=9)
    sweep.set(xlabel=xlabel + ' threshold', ylabel='Windows flagged (%)', ylim=(-2, 102),
              title='What changes when the threshold moves?')
    sweep.legend(fontsize=9)
    for axis in (ax, sweep):
        axis.grid(axis='x', alpha=.18)
    fig.suptitle(f'{bundle.key} · {bundle.score} | Held out: {bundle.held_out} — {NAMES[bundle.held_out]}', fontsize=13)
    fig.text(.5, .025, 'Orange = held-out OOD; blue = the six ID tasks. Black bar = IQR; tick = median. '
             'Flag rule: score > threshold.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    return fig


def loto_score_explorer(key, score='primary', thresholds=None, run_dirs=None):
    """Render all three holdouts; optional thresholds are exploratory, never saved."""
    thresholds = thresholds or {}
    if set(thresholds) - set(HOLDOUTS):
        raise ValueError('Threshold keys must be SA, ST, or TR.')
    bundles = {}
    for task in HOLDOUTS:
        display(Markdown(f'### Held out: {task} — {NAMES[task]}'))
        try:
            if run_dirs is not None and task not in run_dirs:
                raise FileNotFoundError(f'No run directory supplied for {task}.')
            bundle = load_loto_scores(key, task, score, None if run_dirs is None else run_dirs[task])
        except FileNotFoundError as exc:
            print(f'Unavailable — {exc} No plot or performance is inferred.')
            continue
        bundles[task] = bundle
        candidate = thresholds.get(task)
        fig = plot_loto_scores(bundle, candidate)
        plt.show()
        plt.close(fig)
        # Modest local changes help expose tightly clustered scores (e.g. GAN standing).
        candidates = bundle.threshold * np.array([.5, .75, .9, .95, .99, 1, 1.01, 1.05, 1.1, 1.25])
        candidates = np.unique(np.r_[candidates, [] if candidate is None else [candidate]])
        table = threshold_rates(bundle, candidates)
        table.insert(0, 'point', ['saved' if t == bundle.threshold else
                                 'what-if' if candidate is not None and t == candidate else ''
                                 for t in candidates])
        display(table[['point', 'threshold', 'OOD recall (%)', 'ID false alarms (%)']].style.format(
            {'threshold': '{:.8g}', 'OOD recall (%)': '{:.2f}', 'ID false alarms (%)': '{:.2f}'}))
        selected = bundle.threshold if candidate is None else candidate
        row = threshold_rates(bundle, [selected]).iloc[0]
        task_table = pd.DataFrame([{'task': t, 'role': 'held-out OOD' if t == task else 'ID',
                                   'windows': int(np.sum(bundle.modes == t)),
                                   'flagged (%)': row[f'{t} flagged (%)']} for t in TASKS])
        display(Markdown(f'**Per-task flags at τ = {selected:.8g}** '
                         '(recall on the held-out task; false alarms on each ID task).'))
        display(task_table.style.format({'flagged (%)': '{:.2f}'}).apply(
            lambda row: [f'color: {ORANGE if row["task"] == task else BLUE}'] * len(row), axis=1))
    return bundles
