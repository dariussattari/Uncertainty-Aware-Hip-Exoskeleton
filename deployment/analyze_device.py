"""Analysis and figures for D04 (device timing and stability), from the overnight battery's outputs pulled to the Mac.

    .venv/bin/python deployment/analyze_device.py

Reads data/replay/scirob/{soak/soak.npz, latency_budget.json, predictions/{jitter_*, jetson, jetson_fp16},
overnight_logs/tegrastats.log} and the dev reference predictions; prints the D04 numbers and writes
deployment/docs/figures/D04_*.png.

Soak stalls are split by cause: the harness saves its timing arrays every 10 minutes inside the loop (soak_test.py), so a
run of missed ticks that starts within ``SAVE_WINDOW_TICKS`` after a 10-minute boundary is the harness, not the
inference path; everything else is the device's own jitter.
"""
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_replay as ar  # noqa: E402
import scirob_replay as sr  # noqa: E402
from plot_replay import FIGURES, GRID, INK, MUTED, SURFACE, save  # noqa: E402  (same figure style as D03)

ROOT = sr.OUT
CONDITIONS = {'baseline': 'jitter_baseline', 'GC frozen': 'jitter_gcfreeze', 'pinned core 3': 'jitter_cpu', 'both': 'jitter_both'}
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']                  # dataviz reference palette, validated for 4
TICKS_PER_SAVE = 60_000                                                  # 600 s / 10 ms
SAVE_WINDOW_TICKS = 200


def get_soak() -> dict:
    d = np.load(ROOT / 'soak' / 'soak.npz')
    step, late, missed = d['step_ms'], d['late_ms'], d['missed']
    idx = np.flatnonzero(missed)
    starts = idx[np.r_[True, np.diff(idx) > 1]] if len(idx) else idx
    harness_run = (starts % TICKS_PER_SAVE) < SAVE_WINDOW_TICKS
    in_harness = np.zeros(len(step), bool)
    for s in starts[harness_run]:
        end = s
        for e in range(s, min(s + 100_000, len(step))):                     # the stall and its catch-up
            if not missed[e]:
                break
            end = e
        in_harness[s:end + 1] = True
    peak = np.array([late[s:s + 60].max() for s in starts])
    return {'step': step, 'late': late, 'missed': missed, 'starts': starts, 'harness_run': harness_run, 'peak': peak,
            'in_harness': in_harness, 'rss': d['rss']}


def get_tegrastats() -> pd.DataFrame:
    rows = []
    pattern = re.compile(r'^(\S+ \S+) RAM (\d+)/\d+MB.*GR3D_FREQ (\d+)%.*cpu@([\d.]+)C.*gpu@([\d.]+)C.*tj@([\d.]+)C.*VDD_IN (\d+)mW')
    for line in (ROOT / 'overnight_logs' / 'tegrastats.log').read_text().splitlines():
        m = pattern.match(line)
        if m:
            rows.append((m[1], int(m[2]), int(m[3]), float(m[4]), float(m[5]), float(m[6]), int(m[7])))
    df = pd.DataFrame(rows, columns = ['time', 'ram_mb', 'gpu_pct', 'cpu_c', 'gpu_c', 'tj_c', 'power_mw'])
    df['time'] = pd.to_datetime(df['time'], format = '%m-%d-%Y %H:%M:%S')
    return df


def get_steps(folder: str) -> np.ndarray:
    return np.concatenate([np.load(ROOT / 'predictions' / folder / f'{s}.npz')['step_ms'] for s in sr.DEV])


def describe(step: np.ndarray) -> dict[str, float]:
    return {'steps': len(step), 'median': np.median(step), 'p99': np.percentile(step, 99), 'p99.9': np.percentile(step, 99.9),
            'p99.99': np.percentile(step, 99.99), '>5 ms': int((step > 5).sum()), '>10 ms': int((step > 10).sum()), 'max': step.max()}


def plot_soak(soak: dict) -> None:
    hours = soak['starts'] / 360_000
    fig, ax = plt.subplots(figsize = (7.2, 3.3))
    h = soak['harness_run']
    ax.scatter(hours[h], soak['peak'][h], s = 22, color = COLORS[1], label = f'harness checkpoint save, every 10 min ({h.sum()} runs)',
               zorder = 3, edgecolor = SURFACE, linewidth = 1)
    ax.scatter(hours[~h], soak['peak'][~h], s = 22, color = COLORS[0], label = f'device jitter ({(~h).sum()} runs)', zorder = 3,
               edgecolor = SURFACE, linewidth = 1)
    ax.axhline(10, color = INK, lw = 1, ls = '--')
    ax.annotate('10 ms tick', (1.0, 10), xycoords = ('axes fraction', 'data'), xytext = (2, 3), textcoords = 'offset points', fontsize = 7.5)
    ax.set_yscale('log')
    ax.set_xlabel('Hours into the soak')
    ax.set_ylabel('Worst tick start delay in the run (ms, log)')
    ax.legend(fontsize = 8, loc = 'upper left')
    ax.set_title('Missed ticks: the big ones are the test harness saving every 10 min, not inference', loc = 'left', fontsize = 9.5,
                 fontweight = 'bold', color = INK)
    save(fig, 'D04_soak_stalls.png')


def plot_jitter(conditions: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize = (7.2, 3.3))
    for (name, step), color in zip(conditions.items(), COLORS):
        x = np.sort(step)
        ccdf = 1 - np.arange(len(x)) / len(x)
        ax.step(x, ccdf, where = 'post', color = color, lw = 1.8, label = f'{name}: {int((step > 5).sum())} steps > 5 ms, {int((step > 10).sum())} > 10 ms')
    ax.set_yscale('log')
    ax.set_xlim(1.5, 20)
    ax.axvline(10, color = INK, lw = 1, ls = '--')
    ax.set_xlabel('Step time (ms): window, copy, engine, copy back')
    ax.set_ylabel('Share of steps slower (log)')
    ax.legend(fontsize = 7.5, loc = 'upper right')
    ax.set_title('Jitter is the same with or without GC freezing and core pinning', loc = 'left', fontsize = 9.5, fontweight = 'bold',
                 color = INK)
    save(fig, 'D04_jitter_conditions.png')


def plot_budget(budget: dict) -> None:
    keys = ['B1 engine (two legs)', 'LSTM-VAE encoder, fp32', 'TCN-VAE encoder, fp32', 'novelty scores (LOF + Mahalanobis + variance)',
            'regime classifier (183 trees)', 'causal median + u + regime']
    labels = ['B1 engine (both legs)', 'LSTM-VAE encoder', 'TCN-VAE encoder', 'Novelty scores (LOF ...)', 'Regime classifier', 'Smoothing, u, regime']
    med = [budget[k]['median'] for k in keys]
    p99 = [budget[k]['p99'] for k in keys]
    fig, ax = plt.subplots(figsize = (7.2, 3.0))
    y = np.arange(len(keys))[::-1]
    colors = [COLORS[0]] + [COLORS[1]] * 5
    ax.barh(y, med, color = colors, height = 0.6, zorder = 3)
    for yi, m, p in zip(y, med, p99):
        ax.annotate(f'{m:.2f} ms (p99 {p:.2f})', (m, yi), xytext = (4, 0), textcoords = 'offset points', va = 'center', fontsize = 7.5,
                    color = MUTED)
    ax.set_yticks(y, labels)
    ax.axvline(10, color = INK, lw = 1, ls = '--')
    ax.annotate('10 ms tick', (10, y[0] + 0.45), xytext = (3, 0), textcoords = 'offset points', fontsize = 7.5)
    total = sum(med[1:])
    ax.set_xlabel(f'Median time per call, batch 1 (ms). Awareness total {total:.1f} ms per 50 ms window')
    ax.set_xlim(0, 14)
    ax.grid(axis = 'y', visible = False)
    ax.set_title('B1 fits a 10 ms tick; the awareness layer needs its own 50 ms cadence', loc = 'left', fontsize = 9.5,
                 fontweight = 'bold', color = INK)
    save(fig, 'D04_latency_budget.png')


def plot_thermal(teg: pd.DataFrame, soak_start: pd.Timestamp) -> None:
    hours = (teg['time'] - teg['time'].iloc[0]).dt.total_seconds() / 3600
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (9.6, 3.0))
    smooth = lambda s: s.rolling(60, min_periods = 1).median()            # noqa: E731  (1 min)
    ax1.plot(hours, smooth(teg['tj_c']), color = COLORS[1], lw = 1.6, label = 'junction (tj)')
    ax1.plot(hours, smooth(teg['gpu_c']), color = COLORS[0], lw = 1.2, label = 'GPU')
    ax1.set_ylabel('Temperature (°C, 1 min median)')
    ax1.legend(fontsize = 8, loc = 'upper right')
    ax2.plot(hours, smooth(teg['power_mw']) / 1000, color = COLORS[0], lw = 1.6)
    ax2.set_ylabel('Board input power (W, 1 min median)')
    soak_h = (soak_start - teg['time'].iloc[0]).total_seconds() / 3600
    for ax in (ax1, ax2):
        ax.axvline(soak_h, color = MUTED, lw = 1, ls = ':')
        ax.annotate('soak starts', (soak_h, ax.get_ylim()[1]), xytext = (3, -10), textcoords = 'offset points', fontsize = 7.5,
                    color = MUTED)
        ax.set_xlabel('Hours since the battery started')
    ax1.set_title(f'Temperature: max tj {teg["tj_c"].max():.1f} °C', loc = 'left', fontsize = 9, color = INK)
    ax2.set_title(f'Power: median {teg["power_mw"].median() / 1000:.1f} W', loc = 'left', fontsize = 9, color = INK)
    fig.tight_layout()
    save(fig, 'D04_thermal_power.png')


def main() -> None:
    FIGURES.mkdir(parents = True, exist_ok = True)
    pd.set_option('display.width', 200)
    soak = get_soak()
    step, missed = soak['step'], soak['missed']
    own = missed & ~soak['in_harness']
    hours = np.arange(len(step)) // 360_000
    print(f"SOAK: {len(step):,} ticks (12 h), missed {int(missed.sum())} ({missed.mean():.1e})")
    print(f"  harness-save runs: {int(soak['harness_run'].sum())}, peak delay {soak['peak'][soak['harness_run']].min():.0f}"
          f"-{soak['peak'][soak['harness_run']].max():.0f} ms, {int(soak['in_harness'].sum())} missed ticks")
    print(f"  device's own misses: {int(own.sum())} ({own.mean():.1e}) in {int((~soak['harness_run']).sum())} runs; "
          f"steps > 10 ms {int((step > 10).sum())} ({(step > 10).mean():.1e}), > 5 ms {int((step > 5).sum())}")
    print(f"  step median {np.median(step):.2f}, p99 {np.percentile(step, 99):.2f}, p99.99 {np.percentile(step, 99.99):.2f}, max {step.max():.1f} ms")
    per_hour = pd.DataFrame({'hour': range(12), 'step p99': [np.percentile(step[hours == h], 99) for h in range(12)],
                             'step p99.99': [np.percentile(step[hours == h], 99.99) for h in range(12)],
                             "device's own misses": [int(own[hours == h].sum()) for h in range(12)]})
    print(per_hour.round(2).to_string(index = False))
    rss = soak['rss']
    print(f"  resident memory: {rss[1, 1]:.0f} MB at 1 min, {rss[-1, 1]:.0f} MB at 12 h (max {rss[:, 1].max():.0f})")

    conditions = {name: get_steps(folder) for name, folder in CONDITIONS.items()}
    print('\nJITTER (dev replay, calibrated, flat out):')
    print(pd.DataFrame({k: describe(v) for k, v in conditions.items()}).T.round(2).to_string())

    fp16, fp32 = get_steps('jetson_fp16'), np.concatenate(
        [np.load(ROOT / 'predictions' / 'jitter_baseline' / f'{s}.npz')['step_ms'] for s in sr.DEV])
    diffs = []
    for s in sr.DEV:
        a = np.load(ROOT / 'predictions' / 'jetson' / f'{s}.npz')['pred_calibrated']
        b = np.load(ROOT / 'predictions' / 'jetson_fp16' / f'{s}.npz')['pred_calibrated']
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            raise AssertionError(f'{s}: fp16 and fp32 replays predicted on different samples')
        diffs.append(np.abs(a - b)[np.isfinite(a)])
    d = np.concatenate(diffs)
    print(f'\nFP16 vs FP32 on the dev replay: {len(d):,} predictions, max |diff| {d.max():.2e}, p99.9 {np.percentile(d, 99.9):.2e} N m/kg; '
          f'step median {np.median(fp16):.2f} vs {np.median(fp32):.2f} ms, p99 {np.percentile(fp16, 99):.2f} vs {np.percentile(fp32, 99):.2f}')
    dev = ar.get_subjects('dev')
    sign = ar.get_sign(pd.concat([ar.get_rows(s, 'reference') for s in dev]))
    for source in ('jetson', 'jetson_fp16'):
        rows = pd.concat([ar.get_rows(s, source) for s in dev], ignore_index = True)
        rows[['trq_est', 'moment_t', 'moment_lag']] *= sign
        rows['pooled'] = 'all'
        t = ar.score(rows, ['pooled'])
        print(f"  {source}: B1 calibrated R² {t['B1 calibrated r2'].iloc[0]:.4f}, RMSE {t['B1 calibrated rmse'].iloc[0]:.4f}")

    budget = pd.read_json(ROOT / 'latency_budget.json').to_dict()
    print('\nLATENCY BUDGET (median / p99 ms):')
    for k, v in budget.items():
        print(f"  {k:58s} {v['median']:6.2f} {v.get('p99', float('nan')):6.2f}")

    teg = get_tegrastats()
    soak_start = pd.Timestamp('2026-10-27 19:33:34')
    s = teg[teg['time'] >= soak_start]
    print(f"\nTEGRASTATS ({len(teg):,} s): tj max {teg['tj_c'].max():.1f} °C, during soak median {s['tj_c'].median():.1f} "
          f"(first hour {s[s['time'] < soak_start + pd.Timedelta('1h')]['tj_c'].median():.1f}, last hour "
          f"{s[s['time'] > s['time'].max() - pd.Timedelta('1h')]['tj_c'].median():.1f}); power median {teg['power_mw'].median() / 1000:.2f} W, "
          f"max {teg['power_mw'].max() / 1000:.2f} W; RAM used {teg['ram_mb'].min()}-{teg['ram_mb'].max()} MB of 7620")

    plot_soak(soak)
    plot_jitter(conditions)
    plot_budget(budget)
    plot_thermal(teg, soak_start)


if __name__ == '__main__':
    main()
