"""Figures for the Sci-Rob replay (D03): accuracy per mode, example traces, calibration, timing alignment, device.

    .venv/bin/python deployment/plot_replay.py [--subjects dev|final|all]

Reads what ``analyze_replay.py`` reads (the replay streams, the Jetson and reference predictions) and writes
deployment/docs/figures/D03_<subjects>_*.png. Same scoring rules as the analysis (B1 at t - 10, trq_est at t, common
samples, the dev-derived moment sign). Colors: the dataviz reference palette, validated for these three series.
"""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_replay as ar  # noqa: E402
import scirob_replay as sr  # noqa: E402

FIGURES = Path(__file__).resolve().parent / 'docs' / 'figures'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SERIES = {'B1 calibrated': '#2a78d6', 'Online trq_est': '#eb6834', 'B1 nominal': '#1baf7a'}   # fixed order
MODES = ('LG', 'RA', 'RD', 'SA', 'SD')
MODE_NAMES = {'LG': 'Level\nwalking', 'RA': 'Ramp\nascent', 'RD': 'Ramp\ndescent', 'SA': 'Stair\nascent', 'SD': 'Stair\ndescent'}
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200, 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, bbox_inches = 'tight', facecolor = SURFACE)
    plt.close(fig)
    print(f'wrote {(FIGURES / name).relative_to(FIGURES.parents[2])}')


def plot_modes(rows: pd.DataFrame, tag: str) -> None:
    """Grouped bars: hip R² per mode, three estimators, common samples."""
    table = ar.score(rows, ['mode']).set_index('mode').reindex(list(MODES))
    columns = {'B1 nominal': 'B1 nominal r2', 'B1 calibrated': 'B1 calibrated r2', 'Online trq_est': 'online trq_est r2'}
    order = ['B1 nominal', 'B1 calibrated', 'Online trq_est']
    fig, ax = plt.subplots(figsize = (7.2, 3.4))
    width = 0.26
    for i, name in enumerate(order):
        values = table[columns[name]].to_numpy()
        x = np.arange(len(MODES)) + (i - 1) * (width + 0.02)
        ax.bar(x, values, width, color = SERIES[name], label = name, zorder = 3)
        for xi, v in zip(x, values):
            ax.annotate(f'{v:.2f}', (xi, v), xytext = (0, 2 if v >= 0 else -9), textcoords = 'offset points', ha = 'center',
                        fontsize = 7, color = MUTED)
    ax.axhline(0, color = MUTED, lw = 0.8)
    ax.set_xticks(range(len(MODES)), [f'{MODE_NAMES[m]}\nn={int(table.loc[m, "n"]):,}' for m in MODES])
    ax.set_ylabel('Hip moment R² (vs inverse dynamics)')
    ax.set_ylim(min(-0.3, np.nanmin(table[list(columns.values())].to_numpy()) - 0.1), 1)
    ax.grid(axis = 'x', visible = False)
    ax.legend(ncol = 3, loc = 'upper right', fontsize = 8)
    ax.set_title('B1 trails the exo\'s own estimator on every mode; calibration helps most on stairs', loc = 'left', fontsize = 9.5,
                 color = INK, fontweight = 'bold')
    save(fig, f'D03_{tag}_r2_per_mode.png')


def plot_traces(subject: str, tag: str, sign: float) -> None:
    """A few seconds of level walking and stair descent, left leg: ground truth vs the three estimators."""
    data = np.load(sr.OUT / f'{subject}.npz')
    pred = np.load(ar.PRED / 'jetson' / f'{subject}.npz')
    fig, axes = plt.subplots(1, 2, figsize = (9.6, 3.2), sharey = True)
    for ax, mode in zip(axes, ('LG', 'SD')):
        best = None
        for k in np.flatnonzero(data['mode'] == mode):                      # the trial with the most ground truth
            s, n = data['start'][k], data['length'][k]
            cover = np.isfinite(data['moment'][s:s + n, 0]).mean()
            best = (k, cover) if best is None or cover > best[1] else best
        k = best[0]
        s, n = int(data['start'][k]), int(data['length'][k])
        moment = sign * data['moment'][s:s + n, 0]
        ok = np.flatnonzero(np.isfinite(moment) & np.isfinite(pred['pred_calibrated'][s:s + n, 0]))
        a = ok[0] + 10 if len(ok) else 300
        span = slice(a, min(a + 4 * sr.nt.RATE_HZ, n))                         # 4 s
        t = np.arange(n)[span] / sr.nt.RATE_HZ
        b1 = {v: pred[f'pred_{v}'][s:s + n, 0] for v in sr.VARIANTS}
        lagged = np.r_[np.full(ar.LAG, np.nan), moment[:-ar.LAG]]           # B1 at t estimates the moment at t - 10
        ax.plot(t, lagged[span], color = INK, lw = 2.2, label = 'Inverse dynamics (B1 target, t − 50 ms)')
        for name, y in (('B1 nominal', b1['nominal']), ('B1 calibrated', b1['calibrated'])):
            yy = y[span]
            ax.plot(t[np.isfinite(yy)], yy[np.isfinite(yy)], color = SERIES[name], lw = 1.6, label = name)
        ax.plot(t, sign * data['trq_est'][s:s + n, 0][span], color = SERIES['Online trq_est'], lw = 1.6, label = 'Online trq_est (t)')
        ax.set_title(f'{MODE_NAMES[mode].replace(chr(10), " ")}, {subject} left leg\n{data["trial"][k]}', loc = 'left',
                     fontsize = 8.5, color = INK)
        ax.set_xlabel('Time in trial (s)')
    axes[0].set_ylabel('Hip moment (N·m/kg, flexion +)')
    axes[1].legend(fontsize = 7.5, loc = 'upper left', bbox_to_anchor = (1.01, 1))
    fig.suptitle('Example traces: B1 follows walking; on stair descent it misses the stance moment', x = 0.01, ha = 'left',
                 fontsize = 9.5, fontweight = 'bold', color = INK)
    fig.tight_layout()
    save(fig, f'D03_{tag}_example_traces.png')


def plot_calibration(rows: pd.DataFrame, tag: str) -> None:
    """Slope chart: per participant and leg, B1 R² nominal -> calibrated."""
    table = ar.score(rows, ['subject', 'leg'])
    fig, axes = plt.subplots(1, 2, figsize = (6.4, 3.2), sharey = True)
    for ax, leg, title in zip(axes, ('l', 'r'), ('Left leg (IMU rolled 23–36°)', 'Right leg (rolled ~6°)')):
        part = table[table['leg'] == leg].sort_values('B1 calibrated r2')
        label_y, gap = [], 0.022                                             # end labels at least ``gap`` apart in R²
        for _, r in part.iterrows():
            a, b = r['B1 nominal r2'], r['B1 calibrated r2']
            ax.plot([0, 1], [a, b], color = MUTED, lw = 1.2, zorder = 2)
            ax.scatter([0], [a], s = 42, color = SERIES['B1 nominal'], zorder = 3, edgecolor = SURFACE, linewidth = 1.5)
            ax.scatter([1], [b], s = 42, color = SERIES['B1 calibrated'], zorder = 3, edgecolor = SURFACE, linewidth = 1.5)
            y = max(b, label_y[-1] + gap) if label_y else b
            label_y.append(y)
            ax.annotate(f"{r['subject']} {b - a:+.2f}", (1, b), xytext = (1.08, y), textcoords = 'data', va = 'center',
                        fontsize = 7.5, color = MUTED)
        ax.set_xticks([0, 1], ['Nominal\n(axis map)', 'Calibrated\n(per person)'])
        ax.set_xlim(-0.3, 1.6)
        ax.grid(axis = 'x', visible = False)
        ax.set_title(title, loc = 'left', fontsize = 9, color = INK)
    axes[0].set_ylabel('B1 hip R²')
    fig.suptitle('Per-person IMU calibration helps the rolled left IMU, not the right', x = 0.01, ha = 'left', fontsize = 9.5,
                 fontweight = 'bold', color = INK)
    fig.tight_layout()
    save(fig, f'D03_{tag}_calibration.png')


def plot_lag(subjects: list[str], sign: float, tag: str) -> None:
    """Correlation with the moment shifted by -30..30 samples; B1's trained lag marked."""
    scan = ar.get_lag_scan(subjects, sign)
    ms = scan['lag_samples'] * 1000 / sr.nt.RATE_HZ
    fig, ax = plt.subplots(figsize = (6.4, 3.2))
    for name, col in (('B1 calibrated', 'B1 calibrated'), ('Online trq_est', 'online trq_est')):
        i = int(scan[col].idxmax())
        ax.plot(ms, scan[col], color = SERIES[name], lw = 2, marker = 'o', ms = 5, mec = SURFACE, mew = 1.5,
                label = f'{name} (peak r {scan[col][i]:.3f} at {ms[i]:.0f} ms)')
    ax.axvline(ar.LAG * 1000 / sr.nt.RATE_HZ, color = MUTED, lw = 1, ls = '--')
    ax.annotate('B1 trained to estimate\n50 ms behind (scoring rule)', (ar.LAG * 1000 / sr.nt.RATE_HZ, 0.55), xytext = (5, 0),
                textcoords = 'offset points', fontsize = 7.5, color = MUTED)
    ax.set_xlabel('Lag: prediction at t vs moment at t − lag (ms)')
    ax.set_ylabel('Correlation r with inverse dynamics')
    ax.legend(loc = 'lower left', fontsize = 7.5, bbox_to_anchor = (0.1, 0.0))
    ax.set_title('On Sci-Rob, B1 lines up best at 0–25 ms, not the 50 ms it was trained for', loc = 'left', fontsize = 9.5,
                 fontweight = 'bold', color = INK)
    save(fig, f'D03_{tag}_lag_scan.png')


def plot_device(subjects: list[str], tag: str) -> None:
    """Jetson step latency against the 10 ms tick, and |Jetson - PyTorch reference| per prediction."""
    steps = np.concatenate([np.load(ar.PRED / 'jetson' / f'{s}.npz')['step_ms'] for s in subjects])
    diffs = []
    for s in subjects:
        ref, jet = np.load(ar.PRED / 'reference' / f'{s}.npz'), np.load(ar.PRED / 'jetson' / f'{s}.npz')
        for v in sr.VARIANTS:
            a = ref[f'pred_{v}']
            diffs.append(np.abs(a - jet[f'pred_{v}'])[np.isfinite(a)])
    diffs = np.concatenate(diffs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (9.6, 3.2))
    ax1.hist(steps, bins = np.arange(0, 17.5, 0.1), color = SERIES['B1 calibrated'], zorder = 3)
    ax1.set_yscale('log')
    ax1.axvline(10, color = INK, lw = 1.2, ls = '--')
    ax1.annotate('10 ms tick\n(100 Hz)', (10, 3), xytext = (4, 0), textcoords = 'offset points', fontsize = 7.5)
    ax1.annotate(f'median {np.median(steps):.2f} ms\np99 {np.percentile(steps, 99):.2f} ms\n'
                 f'{int((steps > 10).sum())} of {len(steps):,} > 10 ms', (0.97, 0.62), xycoords = 'axes fraction', ha = 'right',
                 va = 'top', fontsize = 7.5, color = MUTED)
    ax1.set_xlabel('Step time on the Jetson (ms): window, copy, engine, copy back')
    ax1.set_ylabel('Steps (log)')
    ax1.set_title('Latency per 100 Hz step, two legs', loc = 'left', fontsize = 9, color = INK)
    bins = np.logspace(-9, -2, 71)
    ax2.hist(np.clip(diffs, 1e-9, None), bins = bins, color = SERIES['Online trq_est'], zorder = 3)
    ax2.set_xscale('log')
    ax2.axvline(5e-3, color = INK, lw = 1.2, ls = '--')
    ax2.annotate('tolerance 5e-3', (5e-3, ax2.get_ylim()[1] * 0.8), xytext = (-4, 0), textcoords = 'offset points', ha = 'right',
                 fontsize = 7.5)
    ax2.annotate(f'max {diffs.max():.1e} N·m/kg\nover {len(diffs):,} predictions', (0.03, 0.95), xycoords = 'axes fraction',
                 va = 'top', fontsize = 7.5, color = MUTED)
    ax2.set_xlabel('|Jetson engine − PyTorch reference| (N·m/kg)')
    ax2.set_ylabel('Predictions')
    ax2.set_title('The device reproduces the reference', loc = 'left', fontsize = 9, color = INK)
    fig.tight_layout()
    save(fig, f'D03_{tag}_device.png')


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split('\n')[0])
    parser.add_argument('--subjects', default = 'dev', choices = ('dev', 'final', 'all'))
    args = parser.parse_args()
    FIGURES.mkdir(parents = True, exist_ok = True)
    dev = ar.get_subjects('dev')
    sign = ar.get_sign(pd.concat([ar.get_rows(s, 'reference') for s in dev]))
    subjects = ar.get_subjects(args.subjects)
    rows = pd.concat([ar.get_rows(s, 'jetson') for s in subjects], ignore_index = True)
    rows[['trq_est', 'moment_t', 'moment_lag']] *= sign
    plot_modes(rows, args.subjects)
    plot_traces(subjects[0], args.subjects, sign)
    plot_calibration(rows, args.subjects)
    plot_lag(dev, sign, args.subjects)
    plot_device(subjects, args.subjects)


if __name__ == '__main__':
    main()
