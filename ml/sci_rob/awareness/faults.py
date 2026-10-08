"""Fault injection: an OOD-dangerous condition the dataset does not contain, scored through the
whole awareness pipeline built for a holdout.

    python ml/sci_rob/awareness/faults.py --holdout DOWN            # print the summary
    python ml/sci_rob/awareness/faults.py --holdout DOWN --write    # results/<holdout>/faults_*.{parquet,csv,md,png}

Four cells of the novelty x health grid are built on the same participants' test windows:

* **ID-safe**        level-ground windows, untouched (familiar, healthy)
* **ID-dangerous**   the same level-ground windows with a sensor fault injected into the *input* only
* **OOD-safe**       ramp-descent windows, untouched (unfamiliar under DOWN, but ordinary gait)
* **OOD-dangerous**  the same ramp-descent windows with the same fault

Faults are applied in raw sensor units, persistently across every window of a trial so the
causal median filter behaves as it would in deployment, and the forecast target (the future
encoder reading) is left untouched, so realized error measures what the corrupted input does to
the prediction. Every score of the pipeline is recorded for every window: LOF on the encoder
latent, reconstruction NLL, the regime probe's probabilities and entropy, the unrouted forecast
ensemble's disagreement and realized error, and the regime experts' routed error. The question
is whether faults land somewhere a safe unfamiliar regime does not: high novelty *and* unhealthy
forecasts, where ramp descent is high novelty with healthy forecasts.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from aw_paths import RESULTS, RUNS, run_dir
from evaluation.common import smooth_by_trial

SMOOTH = 10
LEFT_IMU = ['thigh_accel_x_l', 'thigh_accel_y_l', 'thigh_accel_z_l', 'thigh_gyro_x_l', 'thigh_gyro_y_l', 'thigh_gyro_z_l']
RIGHT_IMU = [c.replace('_l', '_r') for c in LEFT_IMU]
LEFT_ALL = ['enc_angle_l', 'enc_velo_l', *LEFT_IMU]


def _idx(channels, names):
    return [channels.index(n) for n in names]


def make_faults(channels, mean, std, rng):
    """name -> function(x_raw (n, 16, 200)) -> corrupted copy. Persistent within a window."""
    gyro_l = _idx(channels, [c for c in LEFT_IMU if 'gyro' in c])
    imu_lr = _idx(channels, LEFT_IMU + RIGHT_IMU)
    enc_l = channels.index('enc_angle_l')
    left = _idx(channels, LEFT_ALL)

    def dead_left_imu(x):        # sensor disconnected: the six left-thigh IMU channels read zero
        y = x.copy(); y[:, _idx(channels, LEFT_IMU), :] = 0.0; return y

    def encoder_offset(x):       # encoder slip: +0.5 rad on the left hip angle
        y = x.copy(); y[:, enc_l, :] += 0.5; return y

    def frozen_right_imu(x):     # sensor freeze: right-thigh IMU stuck at the window's first sample
        y = x.copy(); y[:, _idx(channels, RIGHT_IMU), :] = y[:, _idx(channels, RIGHT_IMU), :1]; return y

    def imu_noise(x):            # electrical noise: Gaussian at twice each IMU channel's own spread
        y = x.copy(); sel = imu_lr
        y[:, sel, :] += rng.standard_normal(y[:, sel, :].shape).astype(np.float32) * (2 * std[sel])[None, :, None]; return y

    def gyro_spikes(x):          # glitches: ten 3-sample spikes of ±20 rad/s on the left gyros
        y = x.copy()
        for i in range(len(y)):
            for t in rng.integers(0, y.shape[-1] - 3, size=10):
                y[i, gyro_l, t:t + 3] += 20.0 * rng.choice([-1, 1])
        return y

    def left_leg_delay(x):       # desynchronized left leg: its eight channels lag by 100 ms
        y = x.copy(); y[:, left, 20:] = x[:, left, :-20]; y[:, left, :20] = x[:, left, :1]; return y

    # encoder_offset is defined but not injected: a constant angle bias is a calibration error, not a
    # dynamic fault (the future encoder reading would carry the same bias), so it is out of scope here.
    return {'dead_left_imu': dead_left_imu, 'frozen_right_imu': frozen_right_imu,
            'imu_noise': imu_noise, 'gyro_spikes': gyro_spikes, 'left_leg_delay': left_leg_delay}


class Pipeline:
    """Every model of the awareness layer for one holdout, applied to arbitrary standardized windows."""

    def __init__(self, holdout, out_root=RUNS, device=None):
        from models.vae import ConvVAE
        from vae_train import VAEConfig
        from models.autoencoder import LatentLOF
        from evaluation.ensemble import load_checkpoint
        from training.common import pick_device
        from forecast import make_builder
        self.holdout = holdout; self.device = pick_device(device)
        enc = run_dir(holdout, 'encoder', out_root)
        cfg = VAEConfig(**json.loads((enc / 'report.json').read_text())['config'])
        self.encoder = ConvVAE().to(self.device).eval()
        self.encoder.load_state_dict(torch.load(enc / 'final.pt', map_location=self.device, weights_only=True)['model'])
        self.lof = LatentLOF(fit_samples=None, seed=cfg.seed).fit(np.load(enc / 'train_reference_mu.npy'))
        probe = joblib.load(run_dir(holdout, 'regime', out_root) / 'logistic.joblib')
        self.probe, self.classes = probe['pipeline'], probe['classes']
        self.data = make_builder(holdout, 'plain')(batch_size=1024)
        self.forecast, _ = load_checkpoint(run_dir(holdout, 'forecast_plain', out_root) / 'final.pt', self.data, self.device)
        self.experts = {}
        for regime in self.classes:
            ck = run_dir(holdout, f'expert_{regime}', out_root) / 'final.pt'
            if ck.exists():
                self.experts[regime], _ = load_checkpoint(ck, self.data, self.device)

    @torch.inference_mode()
    def _ensemble(self, model, x, y, m):
        out = {'A': [], 'E': []}
        for i in range(0, len(x), 1024):
            xb = torch.from_numpy(x[i:i + 1024]).to(self.device); yb = torch.from_numpy(y[i:i + 1024]).to(self.device)
            mb = torch.from_numpy(m[i:i + 1024]).to(self.device)
            pred = model(xb); fbar = pred.mean(1); w = mb / mb.sum(1, keepdim=True).clamp(min=1)
            out['E'].append((((fbar - yb) ** 2) * w).sum(1).cpu().numpy())
            out['A'].append((pred.var(1, unbiased=False) * w).sum(1).cpu().numpy())
        return np.concatenate(out['A']), np.concatenate(out['E'])

    @torch.inference_mode()
    def score(self, x, y, m):
        """x standardized (n, 16, 200); y, m forecast targets and masks. Returns raw (unsmoothed) scores."""
        from models.vae import loss_terms
        mus, nlls = [], []
        for i in range(0, len(x), 1024):
            xb = torch.from_numpy(x[i:i + 1024]).to(self.device)
            mu, lv = self.encoder.encode(xb)
            nll, _ = loss_terms(xb, self.encoder.decoder(mu), mu, lv)
            mus.append(mu.flatten(1).cpu().numpy()); nlls.append(nll.cpu().numpy())
        mu = np.concatenate(mus)
        P = self.probe.predict_proba(mu)
        scores = {'novelty_lof32': self.lof.score(mu), 'recon_nll': np.concatenate(nlls),
                  'regime_entropy': -(np.clip(P, 1e-12, 1) * np.log(np.clip(P, 1e-12, 1))).sum(1),
                  'regime_confidence': P.max(1)}
        for j, c in enumerate(self.classes):
            scores[f'regime_p_{c}'] = P[:, j]
        A, E = self._ensemble(self.forecast, x, y, m)
        scores['forecast_disagreement'] = A; scores['forecast_error'] = E
        if self.experts:
            errs = np.stack([self._ensemble(self.experts[c], x, y, m)[1] for c in self.classes], 1)
            scores['expert_routed_error'] = errs[np.arange(len(x)), P.argmax(1)]
            scores['expert_min_error'] = errs.min(1)
        return scores


def windows(pipe, split_obj, fwd, rows):
    x, y, m = pipe.data.batch(split_obj, rows, fwd)
    return x.numpy(), y.numpy(), m.numpy()


def select_trials(meta, mode, max_windows, seed=0):
    """Whole trials of one mode, up to about max_windows windows, so the causal filter sees full trials."""
    rng = np.random.default_rng(seed)
    trials = meta.loc[meta['mode'] == mode].groupby(['subject', 'trial']).size()
    order = rng.permutation(len(trials)); keep, total = [], 0
    for i in order:
        if total >= max_windows:
            break
        keep.append(trials.index[i]); total += int(trials.iloc[i])
    key = pd.MultiIndex.from_frame(meta[['subject', 'trial']])
    return np.flatnonzero(key.isin(keep) & (meta['mode'] == mode).to_numpy())


def run(holdout, out_root=RUNS, max_windows=12_000, seed=0, device=None, id_mode='LG', safe_mode='RD'):
    pipe = Pipeline(holdout, out_root, device)
    data = pipe.data
    rng = np.random.default_rng(seed)
    faults = make_faults(data.channels, data.mean[0, :, 0], data.std[0, :, 0], rng)
    id_rows = select_trials(data.test.meta, id_mode, max_windows, seed)
    safe_rows = select_trials(data.ood['test'].meta, safe_mode, max_windows, seed)
    x_id, y_id, m_id = windows(pipe, data.test, data._fwd_for('test'), id_rows)
    x_safe, y_safe, m_safe = windows(pipe, data.ood['test'], data._fwd['ood_test'], safe_rows)
    meta_id = data.test.meta.iloc[id_rows].reset_index(drop=True)
    meta_safe = data.ood['test'].meta.iloc[safe_rows].reset_index(drop=True)
    mean, std = data.mean.astype(np.float32), data.std.astype(np.float32)
    conditions = {f'{id_mode} (ID-safe)': (x_id, y_id, m_id, meta_id), f'{safe_mode} (OOD-safe)': (x_safe, y_safe, m_safe, meta_safe)}
    raw_id, raw_safe = x_id * std + mean, x_safe * std + mean      # back to sensor units for the fault definitions
    for name, fn in faults.items():
        conditions[f'{id_mode} + {name} (ID-dangerous)'] = (((fn(raw_id) - mean) / std).astype(np.float32), y_id, m_id, meta_id)
        conditions[f'{safe_mode} + {name} (OOD-dangerous)'] = (((fn(raw_safe) - mean) / std).astype(np.float32), y_safe, m_safe, meta_safe)
    frames = []
    for cond, (x, y, m, meta) in conditions.items():
        print(f'  scoring {cond}: {len(x):,} windows', flush=True)
        raw = pipe.score(x, y, m)
        smooth = {k: smooth_by_trial(v, meta, SMOOTH, 'median') for k, v in raw.items()}
        frames.append(pd.concat([meta[['subject', 'trial', 'mode', 'end_idx']].assign(condition=cond), pd.DataFrame(smooth)], axis=1))
    return pd.concat(frames, ignore_index=True)


def summarize(df, id_label, safe_label):
    from sklearn.metrics import roc_auc_score
    scores = [c for c in df.columns if c not in ('subject', 'trial', 'mode', 'end_idx', 'condition')]
    rows = []
    base_id = df[df.condition == id_label]; base_safe = df[df.condition == safe_label]
    for cond, g in df.groupby('condition', sort=False):
        row = {'condition': cond, 'windows': len(g), 'cell': cond[cond.rfind('(') + 1:-1]}
        for s in scores:
            row[f'median {s}'] = float(np.median(g[s]))
            if cond != id_label:
                v = np.r_[base_id[s], g[s]]; lab = np.r_[np.zeros(len(base_id)), np.ones(len(g))]
                row[f'AUROC vs ID-safe: {s}'] = float(roc_auc_score(lab, v))
            if cond != safe_label:
                v = np.r_[base_safe[s], g[s]]; lab = np.r_[np.zeros(len(base_safe)), np.ones(len(g))]
                row[f'AUROC vs OOD-safe: {s}'] = float(roc_auc_score(lab, v))
        rows.append(row)
    return pd.DataFrame(rows)


def figures(df, holdout, out, id_label, safe_label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conds = list(dict.fromkeys(df.condition))
    colors = {c: ('#287da8' if c == id_label else '#dd791a' if c == safe_label else '#a32d2d' if c.endswith('(ID-dangerous)') else '#4a3aa7') for c in conds}
    keys = ['novelty_lof32', 'recon_nll', 'regime_entropy', 'forecast_disagreement', 'forecast_error'] + \
           [k for k in ('expert_routed_error', 'expert_min_error') if k in df]
    fig, axes = plt.subplots(2, 4, figsize=(20, 11)); axes = axes.flat
    for ax, k in zip(axes, keys):
        v_all = df[k].to_numpy(); use_log = np.all(v_all > 0) and (np.quantile(v_all, .99) / max(np.quantile(v_all, .01), 1e-12) > 100)
        tf = np.log10 if use_log else np.asarray
        for i, c in enumerate(conds):
            v = df.loc[df.condition == c, k].to_numpy(); v = v[np.isfinite(v)]
            pts = tf(np.quantile(v, np.linspace(0, 1, min(1500, len(v)))))
            if np.ptp(pts) > 0:
                body = ax.violinplot([pts], positions=[i], orientation='horizontal', widths=.8, showextrema=False, bw_method=.18)
                body['bodies'][0].set(facecolor=colors[c], edgecolor=colors[c], alpha=.65)
            lo, med, hi = tf(np.quantile(v, [.25, .5, .75]))
            ax.plot([lo, hi], [i, i], color='#222', lw=2.5); ax.plot(med, i, '|', color='k', ms=9, mew=1.6)
        ax.set_yticks(range(len(conds)), conds, fontsize=8); ax.invert_yaxis()
        ax.set_title(k, fontsize=9, loc='left'); ax.set_xlabel('log10 score' if use_log else 'score', fontsize=8); ax.grid(axis='x', alpha=.18)
        lo_lim, hi_lim = np.quantile(v_all, [.002, .998])
        if hi_lim > lo_lim:
            pad = .04 * (tf(hi_lim) - tf(lo_lim)); ax.set_xlim(tf(lo_lim) - pad, tf(hi_lim) + pad)
    for ax in list(axes)[len(keys):]:
        ax.axis('off')
    fig.suptitle(f'{holdout} holdout: blue = ID-safe, orange = OOD-safe, red = ID-dangerous (faulted level ground), purple = OOD-dangerous (faulted ramp descent)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(out / 'faults_spread.png', dpi=130); plt.close(fig)
    # the four cells in the novelty x health plane, one column per fault; health = routed expert error
    # (the regime has its say) on the top row and reconstruction error on the bottom row
    faults = [c[len(id_label.split(' ')[0]) + 3:c.rfind(' (')] for c in conds if c.endswith('(ID-dangerous)')]
    ykeys = [k for k in ('expert_routed_error', 'recon_nll') if k in df]
    cell_style = {'ID-safe': ('#287da8', 'o'), 'OOD-safe': ('#dd791a', 's'), 'ID-dangerous': ('#a32d2d', '^'), 'OOD-dangerous': ('#4a3aa7', 'D')}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(ykeys), len(faults), figsize=(4.2 * len(faults), 4.2 * len(ykeys)), squeeze=False, sharex=True)
    id_mode, safe_mode = id_label.split(' ')[0], safe_label.split(' ')[0]
    for j, fault in enumerate(faults):
        cells = {'ID-safe': id_label, 'OOD-safe': safe_label,
                 'ID-dangerous': f'{id_mode} + {fault} (ID-dangerous)', 'OOD-dangerous': f'{safe_mode} + {fault} (OOD-dangerous)'}
        for i, yk in enumerate(ykeys):
            ax = axes[i, j]
            for cell, cond in cells.items():
                g = df[df.condition == cond]; pick = rng.choice(len(g), min(1200, len(g)), replace=False)
                color, marker = cell_style[cell]
                ax.scatter(np.log10(g['novelty_lof32'].to_numpy()[pick]), np.log10(g[yk].to_numpy()[pick]), s=6, alpha=.35,
                           color=color, marker=marker, label=cell, rasterized=True)
            ax.grid(alpha=.2)
            if i == 0:
                ax.set_title(fault, fontsize=10)
            if j == 0:
                ax.set_ylabel(f'log10 {yk}')
            if i == len(ykeys) - 1:
                ax.set_xlabel('log10 novelty (LOF on z)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9, markerscale=3, frameon=False)
    fig.suptitle(f'{holdout} holdout: the four cells per fault. Health = routed expert error (top) or reconstruction error (bottom); novelty = LOF', fontsize=11)
    fig.tight_layout(rect=(0, .06, 1, .95)); fig.savefig(out / 'faults_quadrants.png', dpi=130); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--holdout', required=True)
    parser.add_argument('--max-windows', type=int, default=12_000)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--device', default=None)
    a = parser.parse_args(argv)
    df = run(a.holdout, max_windows=a.max_windows, device=a.device)
    id_label, safe_label = 'LG (ID-safe)', 'RD (OOD-safe)'
    summ = summarize(df, id_label, safe_label)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    for against in ('ID-safe', 'OOD-safe'):
        show = ['condition', 'windows'] + [c for c in summ if c.startswith(f'AUROC vs {against}: ') and any(k in c for k in ('lof32', 'recon', 'forecast_disagreement', 'routed'))]
        print(f'-- AUROC against {against}'); print(summ[show].rename(columns=lambda c: c.replace(f'AUROC vs {against}: ', '')).round(3).to_string(index=False))
    if a.write:
        out = RESULTS / a.holdout; out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / 'faults_scores.parquet', index=False); summ.to_csv(out / 'faults_summary.csv', index=False)
        from vae_sweep import markdown_table
        (out / 'faults_summary.md').write_text('# Injected sensor faults through the awareness pipeline\n\n' + markdown_table(summ) + '\n')
        figures(df, a.holdout, out, id_label, safe_label); print('wrote', out)


if __name__ == '__main__':
    main()
