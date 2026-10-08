"""Pelvis-IMU ablation: the same encoder, LOF and regime probes on 22 input channels instead of 16.

    python ml/sci_rob/awareness/pelvis_ablation.py --holdout DOWN            # print the comparison
    python ml/sci_rob/awareness/pelvis_ablation.py --holdout DOWN --write    # results/<holdout>/pelvis_ablation.{csv,md,png}

The raw trials carry a pelvis-mounted IMU (six channels) that the 16-channel processed set omits.
Notebook 09 showed it adds almost no task structure to the raw windows; this is the direct test
for novelty detection: ``encoder_pelvis`` is ``encoder`` retrained on ``processed_all_pelvis``
(the 16 channels with identical scaling plus the six pelvis channels), same holdout, same
settings, same seed, and ``regime_pelvis`` the probes on its latent. Every score is compared on
the same test windows: LOF on the 32-D latent and on 12/16 PCs of it, reconstruction NLL, and
the probes' in-distribution accuracy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aw_paths import RESULTS, RUNS, run_dir
from splits import held_out_modes, trained_modes
from evaluation.common import smooth_by_trial

SMOOTH = 10
LOF_DIMS = (12, 16)
TASKS = ('LG', 'RA', 'RD', 'SA', 'SD', 'TR', 'ST')
STAGES = {'16 channels (thigh + encoders)': 'encoder', '22 channels (+ pelvis IMU)': 'encoder_pelvis'}


def encoder_scores(run: Path):
    """Test-window scores of one encoder run, causally filtered, with the PCA-LOF variants recomputed."""
    from sklearn.decomposition import PCA
    from models.autoencoder import LatentLOF
    with np.load(run / 'latents_test.npz') as z:
        Z, labels = z['Z'], z['labels']
        scores = {'LOF, 32-D latent': z['smooth_latent_lof'], 'reconstruction NLL': z['smooth_reconstruction_nll']}
    meta = pd.read_parquet(run / 'latents_test.parquet')
    ref = np.load(run / 'train_reference_mu.npy')
    for k in LOF_DIMS:
        pca = PCA(n_components=k, random_state=0).fit(ref)
        lof = LatentLOF(fit_samples=None, seed=0).fit(pca.transform(ref))
        scores[f'LOF, PCA-{k}'] = smooth_by_trial(lof.score(pca.transform(Z)), meta, SMOOTH, 'median')
    return scores, meta, labels


def run_facts(run: Path) -> dict:
    report = json.loads((run / 'report.json').read_text())
    cfg = report['config']
    hist = report['history']
    facts = {'input channels': None, 'epochs run': len(hist), 'best epoch': report['best_epoch'] + 1,
             'best val loss': float(min(h['val_loss'] for h in hist)), 'seed': cfg['seed']}
    data_root = cfg.get('data_root')
    config_json = (Path(data_root) / 'AE_GAN' / 'config.json') if data_root else None
    if config_json and config_json.exists():
        facts['input channels'] = json.loads(config_json.read_text())['n_channels']
    else:
        facts['input channels'] = 22 if 'pelvis' in run.name else 16
    timing = run / 'timing.json'
    if timing.exists():
        facts['hours'] = json.loads(timing.read_text())['hours']
    return facts


def compare(holdout: str, out_root=RUNS, stages=STAGES):
    """(summary, per_mode, facts): AUROC per score and per held-out mode for each encoder, plus run facts."""
    from sklearn.metrics import roc_auc_score
    out_modes = held_out_modes(holdout)
    rows, per_mode, facts, base_meta = [], [], {}, None
    for label, stage in stages.items():
        run = run_dir(holdout, stage, out_root)
        if not (run / 'latents_test.npz').exists():
            print(f'  {label}: no run at {run}'); continue
        scores, meta, labels = encoder_scores(run)
        if base_meta is None:
            base_meta = meta
        elif not meta[['subject', 'trial', 'end_idx']].equals(base_meta[['subject', 'trial', 'end_idx']]):
            raise ValueError(f'{stage}: test windows differ from the 16-channel run; the ablation is not window-aligned')
        modes = meta['mode'].to_numpy(); ood = labels.astype(bool)
        facts[label] = run_facts(run)
        reg = run_dir(holdout, 'regime' if stage == 'encoder' else stage.replace('encoder', 'regime'), out_root)
        for probe in ('logistic', 'mlp'):
            sj = reg / f'{probe}_summary.json'
            if sj.exists():
                s = json.loads(sj.read_text())
                facts[label][f'{probe} probe ID accuracy'] = s['test_id']['accuracy']
        for name, v in scores.items():
            row = {'encoder': label, 'score': name, 'AUROC OOD vs ID': float(roc_auc_score(ood, v))}
            for t in out_modes:
                sel = (~ood) | (modes == t)
                row[f'AUROC {t} vs ID'] = float(roc_auc_score(ood[sel], v[sel]))
            rows.append(row)
            per_mode.extend({'encoder': label, 'score': name, 'mode': t, 'kind': 'OOD' if t in out_modes else 'ID',
                             'median': float(np.median(v[modes == t])), 'values': v[modes == t]} for t in TASKS if np.any(modes == t))
    summary = pd.DataFrame(rows)
    return summary, per_mode, facts


def figure(per_mode, holdout, path, score='LOF, 32-D latent'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    encoders = list(dict.fromkeys(r['encoder'] for r in per_mode))
    fig, axes = plt.subplots(1, len(encoders), figsize=(6.2 * len(encoders), 4.4), sharex=True)
    axes = np.atleast_1d(axes)
    out_modes = held_out_modes(holdout)
    allv = np.log10(np.concatenate([r['values'] for r in per_mode if r['score'] == score]))
    lo, hi = np.quantile(allv, [.002, .998]); pad = .04 * (hi - lo)
    for ax, enc in zip(axes, encoders):
        rows = [r for r in per_mode if r['encoder'] == enc and r['score'] == score]
        for i, r in enumerate(rows):
            v = np.log10(r['values']); color = '#dd791a' if r['mode'] in out_modes else '#287da8'
            pts = np.quantile(v, np.linspace(0, 1, min(1500, len(v))))
            if np.ptp(pts) > 0:
                body = ax.violinplot([pts], positions=[i], orientation='horizontal', widths=.8, showextrema=False, bw_method=.18)
                body['bodies'][0].set(facecolor=color, edgecolor=color, alpha=.65)
            q1, med, q3 = np.quantile(v, [.25, .5, .75])
            ax.plot([q1, q3], [i, i], color='#222', lw=2.5); ax.plot(med, i, '|', color='k', ms=9, mew=1.6)
        ax.set_yticks(range(len(rows)), [f"{r['mode']} ({'held out' if r['mode'] in out_modes else 'trained'})" for r in rows], fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(lo - pad, hi + pad); ax.grid(axis='x', alpha=.18)
        ax.set_title(enc, fontsize=10, loc='left'); ax.set_xlabel(f'log10 {score}', fontsize=9)
    fig.suptitle(f'{holdout} holdout: novelty score by task, 16 versus 22 input channels (blue = trained on, orange = held out)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(path, dpi=130); plt.close(fig)


def write(summary, per_mode, facts, holdout, results=RESULTS):
    from vae_sweep import markdown_table
    out = Path(results) / holdout; out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / 'pelvis_ablation.csv', index=False)
    facts_df = pd.DataFrame(facts).T; facts_df.index.name = 'encoder'
    facts_df.to_csv(out / 'pelvis_ablation_runs.csv')
    figure(per_mode, holdout, out / 'pelvis_ablation.png')
    text = (f'# Pelvis-IMU ablation under the `{holdout}` holdout (test split)\n\n'
            f'Held out: {", ".join(held_out_modes(holdout))}; trained on {", ".join(trained_modes(holdout))}. '
            'Same encoder (β = 0, no sampling, AE settings), same LOF, same seed, same test windows; the only change is '
            'six pelvis IMU channels appended to the input (their own scaler, the 16 shared channels scaled identically). '
            'AUROC is threshold-free; OOD is the positive class; scores are causally median-filtered within trials.\n\n'
            + markdown_table(summary, '{:.4f}') + '\n\n## Runs\n\n' + markdown_table(facts_df.reset_index(), '{:.4g}') + '\n')
    (out / 'pelvis_ablation.md').write_text(text)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--holdout', required=True)
    parser.add_argument('--out-root', default=str(RUNS))
    parser.add_argument('--write', action='store_true')
    a = parser.parse_args(argv)
    summary, per_mode, facts = compare(a.holdout, a.out_root)
    pd.set_option('display.width', 200)
    print(summary.round(4).to_string(index=False))
    print(pd.DataFrame(facts).T.to_string())
    if a.write and len(summary):
        print('wrote', write(summary, per_mode, facts, a.holdout))


if __name__ == '__main__':
    main()
