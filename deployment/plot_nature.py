"""Figures: B1 vs the Nature Estimator vs Ground Truth on the Nature test participants (Phase 3), per task and as traces.

    .venv/bin/python deployment/plot_nature.py [ml/nature/estimator/runs/hip_thigh_aug_s2]

B1: the run's saved test predictions (``preds_test.npz``; the TensorRT engine on the Jetson reproduces them, D02).
Nature Estimator: the study's estimator as recorded during collection (the dataset's ``*_torque_estimated``), hip sign
flipped to OpenSim flexion-positive and Phase 3 only, as ``score_moment.py`` does. Ground Truth: the dataset's inverse
dynamics (``Joint_Moments_Filt``), the hip 50 ms behind each window's end, as B1 was trained. Phase 3 test participants
(BT19, BT22) were already scored in E41; this pools windows per task (E41 averages per participant first). Writes
deployment/docs/figures/D02_nature_*.png.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'deployment'))
sys.path.insert(0, str(REPO / 'ml' / 'nature' / 'estimator'))
from plot_replay import INK, MUTED, SURFACE, save  # noqa: E402  (same style as D03)
from score_moment import DEPLOYED_SIGN  # noqa: E402

WINDOWS = REPO / 'data' / 'processed_nature' / 'moment' / 'windows.parquet'
B1_COLOR, NATURE_COLOR = '#2a78d6', '#eb6834'


def get_r2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(1 - np.sum((pred - target) ** 2) / np.sum((target - target.mean()) ** 2))


def load(run: Path) -> pd.DataFrame:
    """One row per Phase 3 test window with B1, the Nature Estimator and Ground Truth all defined."""
    d = np.load(run / 'preds_test.npz')
    w = pd.read_parquet(WINDOWS).iloc[d['rows']].reset_index(drop = True)
    df = w[['subject', 'phase', 'task', 'rec', 'end', 'leg', 'paper_train']].assign(
        b1 = d['pred'][:, 0], nature = DEPLOYED_SIGN[0] * d['estimate'][:, 0], truth = d['target'][:, 0])
    df = df[(df['phase'] == 'phase3') & df[['b1', 'nature', 'truth']].notna().all(axis = 1)]
    df['task'] = df['task'].astype(str)
    return df.reset_index(drop = True)


def plot_tasks(df: pd.DataFrame) -> None:
    rows = []
    for task, g in df.groupby('task'):
        rows.append({'task': task, 'n': len(g), 'B1': get_r2(g['b1'].to_numpy(), g['truth'].to_numpy()),
                     'Nature Estimator': get_r2(g['nature'].to_numpy(), g['truth'].to_numpy()), 'trained': bool(g['paper_train'].iloc[0])})
    t = pd.DataFrame(rows).sort_values('B1')
    fig, ax = plt.subplots(figsize = (7.6, 7.4))
    y = np.arange(len(t))
    h = 0.38
    ax.barh(y + h / 2, t['B1'].clip(lower = -1), h, color = B1_COLOR, label = 'B1', zorder = 3)
    ax.barh(y - h / 2, t['Nature Estimator'].clip(lower = -1), h, color = NATURE_COLOR, label = 'Nature Estimator', zorder = 3)
    labels = [f"{r.task.replace('_', ' ')}{'' if r.trained else ' *'}" for r in t.itertuples()]
    ax.set_yticks(y, labels, fontsize = 7.5)
    ax.axvline(0, color = MUTED, lw = 0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel('Hip moment R² vs Ground Truth (clipped at −1)')
    ax.grid(axis = 'y', visible = False)
    pooled_b1 = get_r2(df['b1'].to_numpy(), df['truth'].to_numpy())
    pooled_n = get_r2(df['nature'].to_numpy(), df['truth'].to_numpy())
    ax.legend(loc = 'lower left', fontsize = 8, title = f'pooled: B1 {pooled_b1:.2f}, Nature Est. {pooled_n:.2f}', title_fontsize = 7.5)
    ax.annotate('* task not in B1\'s training set', (0.0, -0.085), xycoords = 'axes fraction', fontsize = 7.5, color = MUTED)
    ax.set_title('Nature data (Phase 3 test participants): B1 beats the Nature Estimator on most tasks', loc = 'left',
                 fontsize = 9.5, fontweight = 'bold', color = INK)
    save(fig, 'D02_nature_r2_per_task.png')


def plot_traces(df: pd.DataFrame) -> None:
    """Window-by-window time series (every 5th sample, 40 Hz) of one recording per task, right leg."""
    fig, axes = plt.subplots(1, 3, figsize = (12, 3.2), sharey = True)
    for ax, task in zip(axes, ('normal_walk', 'stairs', 'squats')):
        g = df[(df['task'] == task) & (df['leg'] == 1)]
        rec = g.groupby('rec').size().idxmax()
        g = g[g['rec'] == rec].sort_values('end')
        end = g['end'].to_numpy()
        gap = np.r_[False, np.diff(end) > 5]                                    # windows missing: break the line, don't bridge
        t = np.insert((end - end[0]) / 200, np.flatnonzero(gap), np.nan)
        series = {k: np.insert(g[k].to_numpy(float), np.flatnonzero(gap), np.nan) for k in ('truth', 'b1', 'nature')}
        keep = ~(t > 6)
        ax.plot(t[keep], series['truth'][keep], color = INK, lw = 2.2, label = 'Ground Truth (inverse dynamics)')
        ax.plot(t[keep], series['b1'][keep], color = B1_COLOR, lw = 1.6, label = 'B1')
        ax.plot(t[keep], series['nature'][keep], color = NATURE_COLOR, lw = 1.6, label = 'Nature Estimator')
        ax.set_title(f"{task.replace('_', ' ')}, {g['subject'].iloc[0]} right leg", loc = 'left', fontsize = 9, color = INK)
        ax.set_xlabel('Time (s)')
    axes[0].set_ylabel('Hip moment (N·m/kg, flexion +)')
    axes[-1].legend(fontsize = 7.5, loc = 'upper left', bbox_to_anchor = (1.01, 1))
    fig.suptitle('Nature data: example traces against Ground Truth', x = 0.01, ha = 'left', fontsize = 9.5, fontweight = 'bold', color = INK)
    fig.tight_layout()
    save(fig, 'D02_nature_example_traces.png')


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'ml' / 'nature' / 'estimator' / 'runs' / 'hip_thigh_aug_s2'
    df = load(run)
    print(f'{len(df):,} Phase 3 test windows, {df["task"].nunique()} tasks, people {sorted(df["subject"].astype(str).unique())}')
    plot_tasks(df)
    plot_traces(df)


if __name__ == '__main__':
    main()
