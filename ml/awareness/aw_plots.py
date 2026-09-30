"""Score spread by task, for every score, with no thresholds.

    python ml/awareness/aw_plots.py --holdout DOWN     # results/<holdout>/spread_<group>.png

One violin per task per score, in fixed task order. Tasks held out in this run are drawn in
orange, the trained-on tasks in blue, so each panel reads like the LOTO explorer's figure:
does the score separate the tasks the model never saw from the ones it did? Black bars are
the interquartile range, the tick the median. Positive scores with a wide range are shown on
a log10 axis. Nothing here decides anything; it is the raw material for choosing what to fuse.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aw_paths import RESULTS
from splits import held_out_modes

TASKS = ('LG', 'RA', 'RD', 'SA', 'SD', 'TR', 'ST')
NAMES = dict(LG='level ground', RA='ramp ascent', RD='ramp descent', SA='stair ascent',
             SD='stair descent', TR='transitions', ST='standing')
BLUE, ORANGE = '#287da8', '#dd791a'
GROUPS = {   # figure name -> score-name prefixes
    'novelty': ('novelty_', 'recon_'),
    'regime': ('regime_',),
    'forecast': ('forecast_plain', 'forecast_regime'),
    'experts': ('expert_',),
    'baselines': ('ae_', 'gan_', 'corr_', 'forecast_full', 'prob_', 'gaitphase_'),
}


def _use_log(v):
    v = v[np.isfinite(v)]
    return len(v) > 0 and np.all(v > 0) and (np.quantile(v, .99) / max(np.quantile(v, .01), 1e-12)) > 100


def spread_panel(ax, values, modes, out_modes, title):
    use_log = _use_log(values)
    tf = np.log10 if use_log else np.asarray
    finite = values[np.isfinite(values)]
    lo_lim, hi_lim = np.quantile(finite, [.002, .998]) if len(finite) else (0, 1)
    for i, task in enumerate(TASKS):
        v = values[modes == task]; v = v[np.isfinite(v)]
        if not len(v):
            continue
        color = ORANGE if task in out_modes else BLUE
        pts = tf(np.quantile(v, np.linspace(0, 1, min(1500, len(v)))))
        if np.ptp(pts) > 0:
            body = ax.violinplot([pts], positions=[i], orientation='horizontal', widths=.8, showextrema=False, bw_method=.18)
            body['bodies'][0].set(facecolor=color, edgecolor=color, alpha=.65)
        lo, med, hi = tf(np.quantile(v, [.25, .5, .75]))
        ax.plot([lo, hi], [i, i], color='#222222', lw=2.5, solid_capstyle='butt')
        ax.plot(med, i, '|', color='black', ms=9, mew=1.6)
    ax.set_yticks(range(len(TASKS)), [f'{t} {"(held out)" if t in out_modes else ""}'.strip() for t in TASKS], fontsize=8)
    for tick, task in zip(ax.get_yticklabels(), TASKS):
        tick.set_color(ORANGE if task in out_modes else BLUE)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9, loc='left')
    if hi_lim > lo_lim:   # axis spans the central 99.6 % of all windows; a few extreme values are clipped, not dropped
        pad = .04 * (tf(hi_lim) - tf(lo_lim)) if use_log else .04 * (hi_lim - lo_lim)
        ax.set_xlim(tf(lo_lim) - pad, tf(hi_lim) + pad)
    ax.set_xlabel('log10 score (axis: central 99.6 %)' if use_log else 'score (axis: central 99.6 %)', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='x', alpha=.18)


def spread_grid(ss, names, out_path, cols=3, title=None):
    names = [n for n in names if n in ss.scores]
    if not names:
        return None
    out_modes = held_out_modes(ss.holdout)
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.9 * rows), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    for ax, name in zip(axes.flat, names):
        spread_panel(ax, ss.scores[name], ss.meta_for(name)['mode'].to_numpy(), out_modes, name)
    fig.suptitle(title or f'{ss.holdout} holdout: score spread by task (orange = never seen in training)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def all_groups(ss, results=RESULTS):
    out = Path(results) / ss.holdout
    written = {}
    for group, prefixes in GROUPS.items():
        names = [n for n in ss.scores if n.startswith(prefixes)]
        path = spread_grid(ss, names, out / f'spread_{group}.png', title=f'{ss.holdout} holdout: {group} scores by task')
        if path:
            written[group] = path
    return written


def main(argv=None):
    from aw_scores import collect
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--holdout', required=True)
    parser.add_argument('--no-kv', action='store_true')
    a = parser.parse_args(argv)
    ss = collect(a.holdout, recompute_kv=not a.no_kv)
    for group, path in all_groups(ss).items():
        print('wrote', path)


if __name__ == '__main__':
    main()
