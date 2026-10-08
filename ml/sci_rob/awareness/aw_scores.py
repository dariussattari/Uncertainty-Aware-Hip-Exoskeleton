"""Collect every per-window score for one holdout into one place, with no thresholds.

    python ml/sci_rob/awareness/aw_scores.py --holdout DOWN            # print the summary
    python ml/sci_rob/awareness/aw_scores.py --holdout DOWN --write    # also results/<holdout>/scores_*.parquet, score_summary.{csv,md}

Scores come from three places and are all already causally median-filtered within trials:

* the awareness stages under ``runs/<holdout>/``: LOF on the encoder latent (32-D, and
  refitted on 12 and 16 PCs of the same training reference), reconstruction NLL, the regime
  probes' class probabilities / confidence / entropy, and for each forecast variant the
  branch disagreement plus the Krogh-Vedelsby terms recomputed from its checkpoint
  (ensemble error, mean branch error, their ratio);
* the six existing models' LOTO leaves under ``ml/sci_rob/LOTO/<holdout>/`` as they finish.

Every AE_GAN-family score is aligned row for row to the holdout's test windows (ID rows then
OOD rows, the order every evaluator writes). The gait-phase ensemble scores a smaller,
labelled window set and is kept with its own metadata. The summary reports per-mode medians
and AUROCs (OOD versus ID, and each held-out mode alone versus ID); AUROC is a ranking
statistic and involves no threshold.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from aw_paths import ROOT, RUNS, RESULTS, run_dir
from splits import DATA_ROOT, held_out_modes, trained_modes
from evaluation.common import smooth_by_trial

SMOOTH = 10
LOF_DIMS = (12, 16)
TASKS = ('LG', 'RA', 'RD', 'SA', 'SD', 'TR', 'ST')
BASELINES = {   # leaf -> (bundle stem, {score name: field})
    'vanilla/AE': ('latents_test', {'ae_lof': 'psi_smooth', 'ae_recon': 'recon_smooth'}),
    'vanilla/GAN': ('scores_test', {'gan_score': 'psi_smooth'}),
    'vanilla/Ensemble_synthetic': ('scores_test', {'corr_disagreement': 'scores'}),
    'vanilla/Ensemble_forecast': ('scores_test', {'forecast_full_disagreement': 'scores'}),
    'prob_ensemble/run': ('scores_test', {'prob_epistemic': 'smooth_epistemic', 'prob_aleatoric': 'smooth_aleatoric',
                                          'prob_total': 'smooth_total', 'prob_ratio': 'smooth_ratio'}),
    'vanilla/Ensemble_GP': ('scores_test', {'gaitphase_disagreement': 'scores'}),   # own window set
}


class ScoreSet:
    """name -> (values, meta); ``meta`` has mode/kind and is shared unless a score says otherwise."""

    def __init__(self, holdout, meta):
        self.holdout, self.meta, self.scores, self.metas, self.notes = holdout, meta, {}, {}, []

    def add(self, name, values, meta=None):
        values = np.asarray(values, dtype=float)
        m = self.meta if meta is None else meta
        if len(values) != len(m):
            raise ValueError(f'{name}: {len(values):,} values for {len(m):,} rows')
        self.scores[name] = values
        if meta is not None:
            self.metas[name] = meta

    def meta_for(self, name):
        return self.metas.get(name, self.meta)

    def aligned(self):
        """Scores sharing the holdout's main test-window order, as one DataFrame."""
        cols = {k: v for k, v in self.scores.items() if k not in self.metas}
        return pd.concat([self.meta.reset_index(drop=True), pd.DataFrame(cols)], axis=1)


# ------------------------------------------------------------------- awareness stages

def encoder_scores(holdout, ss, out_root=RUNS, lof_dims=LOF_DIMS):
    run = run_dir(holdout, 'encoder', out_root)
    if not (run / 'latents_test.npz').exists():
        ss.notes.append('encoder run missing'); return
    with np.load(run / 'latents_test.npz') as z:
        Z, labels = z['Z'], z['labels']
        ss.add('novelty_lof32', z['smooth_latent_lof'])
        ss.add('recon_nll', z['smooth_reconstruction_nll'])
    if not np.array_equal(labels, (ss.meta['kind'] == 'OOD').to_numpy().astype(int)):
        raise ValueError('encoder latents are not in the expected ID-then-OOD order')
    from sklearn.decomposition import PCA
    from models.autoencoder import LatentLOF
    ref = np.load(run / 'train_reference_mu.npy')
    for k in lof_dims:
        pca = PCA(n_components=k, random_state=0).fit(ref)
        lof = LatentLOF(fit_samples=None, seed=0).fit(pca.transform(ref))
        ss.add(f'novelty_lof_pca{k}', smooth_by_trial(lof.score(pca.transform(Z)), ss.meta, SMOOTH, 'median'))


def regime_scores(holdout, ss, out_root=RUNS, probes=('logistic', 'mlp')):
    run = run_dir(holdout, 'regime', out_root)
    for probe in probes:
        parts = [run / f'{probe}_test_{kind}.npz' for kind in ('id', 'ood')]
        if not all(p.exists() for p in parts):
            ss.notes.append(f'regime probe {probe} missing'); continue
        loaded = [np.load(p) for p in parts]
        classes = [str(c) for c in loaded[0]['classes']]
        P = np.concatenate([d['probs'] for d in loaded])
        for j, c in enumerate(classes):
            ss.add(f'regime_{probe}_p_{c}', P[:, j])
        ss.add(f'regime_{probe}_confidence', np.concatenate([d['confidence'] for d in loaded]))
        ss.add(f'regime_{probe}_entropy', np.concatenate([d['entropy'] for d in loaded]))


def forecast_scores(holdout, ss, out_root=RUNS, variants=None, recompute=True, device=None):
    if variants is None:   # every forecast_* run present: plain, regime, and horizon variants such as plain_h200
        variants = sorted(p.name[len('forecast_'):] for p in (Path(out_root) / holdout).glob('forecast_*') if p.is_dir())
    for variant in variants:
        run = run_dir(holdout, f'forecast_{variant}', out_root)
        if not (run / 'scores_test.npz').exists():
            ss.notes.append(f'forecast {variant} missing'); continue
        with np.load(run / 'scores_test.npz') as z:
            ss.add(f'forecast_{variant}_disagreement', z['scores'])
        kv = run / 'kv_test.npz'
        if not kv.exists() and recompute and (run / 'final.pt').exists():
            _decompose(holdout, variant, run, out_root, device)
        if kv.exists():
            with np.load(kv) as z:
                if not np.array_equal(z['labels'], (ss.meta['kind'] == 'OOD').to_numpy().astype(int)):
                    raise ValueError(f'{kv}: label order differs from the test-window order')
                ss.add(f'forecast_{variant}_error', z['E_s'])
                ss.add(f'forecast_{variant}_branch_error', z['Ebar_s'])
                ss.add(f'forecast_{variant}_ratio', z['ratio_s'])


def _decompose(holdout, variant, run, out_root, device=None):
    """Krogh-Vedelsby terms from the saved checkpoint, through the same holdout view it trained on."""
    from forecast import make_builder
    from evaluation.ensemble import load_checkpoint
    from evaluation.ambiguity import decompose
    from training.common import pick_device
    base, _, h = variant.partition('_h')                       # e.g. plain_h200 -> plain at 200 samples
    horizon = int(h) if h else None
    data = make_builder(holdout, base, run_dir(holdout, 'regime', out_root))(batch_size=1024, horizon=horizon)
    device = pick_device(device)
    model, threshold = load_checkpoint(run / 'final.pt', data, device)
    d = decompose(model, data, threshold, 'test', device)
    print(f'  {variant}: decomposition identity residual {d.residual:.2e}', flush=True)
    d.save(run / 'kv_test')


def expert_scores(holdout, ss, out_root=RUNS):
    run = run_dir(holdout, 'experts', out_root)
    if not (run / 'scores_test.npz').exists():
        ss.notes.append('regime experts missing (run experts.py score after pulling both checkpoints)'); return
    with np.load(run / 'scores_test.npz') as z:
        if not np.array_equal(z['labels'], (ss.meta['kind'] == 'OOD').to_numpy().astype(int)):
            raise ValueError('expert scores are not in the expected ID-then-OOD order')
        for k in z.files:
            if k.startswith('smooth_'):
                ss.add(k[len('smooth_'):], z[k])


# ------------------------------------------------------------------- existing baselines

def baseline_meta(holdout, family):
    """ID rows then OOD rows of the rebuilt test split, the order the ensemble evaluators write."""
    meta = pd.read_parquet(DATA_ROOT / family / 'test_meta.parquet')
    out = meta['mode'].isin(held_out_modes(holdout))
    meta = pd.concat([meta.loc[~out], meta.loc[out]], ignore_index=True)
    meta['kind'] = np.where(meta['mode'].isin(held_out_modes(holdout)), 'OOD', 'ID')
    return meta


def baseline_scores(holdout, ss, loto_root=ROOT / 'ml/sci_rob/LOTO'):
    for leaf, (stem, fields) in BASELINES.items():
        run = loto_root / holdout / leaf
        npz = run / f'{stem}.npz'
        if not npz.exists():
            ss.notes.append(f'baseline {leaf} missing'); continue
        with np.load(npz, allow_pickle=False) as z:
            own = None
            if 'Ensemble_GP' in leaf:
                own = baseline_meta(holdout, 'ensemble')
            n = len(z['labels'])
            if own is None and n != len(ss.meta):
                ss.notes.append(f'baseline {leaf}: {n:,} rows, expected {len(ss.meta):,}; skipped'); continue
            if own is not None and n != len(own):
                ss.notes.append(f'baseline {leaf}: {n:,} rows, expected {len(own):,}; skipped'); continue
            expect = ((own if own is not None else ss.meta)['kind'] == 'OOD').to_numpy().astype(int)
            if not np.array_equal(z['labels'], expect):
                ss.notes.append(f'baseline {leaf}: label order differs; skipped'); continue
            for name, field in fields.items():
                if field in z.files:
                    ss.add(name, z[field], own)


# ------------------------------------------------------------------- assembly and summary

def collect(holdout, out_root=RUNS, loto_root=ROOT / 'ml/sci_rob/LOTO', recompute_kv=True, device=None):
    enc = run_dir(holdout, 'encoder', out_root)
    if (enc / 'latents_test.parquet').exists():
        meta = pd.read_parquet(enc / 'latents_test.parquet')
    else:
        meta = baseline_meta(holdout, 'AE_GAN')
    meta = meta[[c for c in ('subject', 'trial', 'mode', 'kind', 'end_idx') if c in meta]].reset_index(drop=True)
    ss = ScoreSet(holdout, meta)
    encoder_scores(holdout, ss, out_root)
    regime_scores(holdout, ss, out_root)
    forecast_scores(holdout, ss, out_root, recompute=recompute_kv, device=device)
    expert_scores(holdout, ss, out_root)
    baseline_scores(holdout, ss, loto_root)
    return ss


def summarize(ss):
    from sklearn.metrics import roc_auc_score
    out = held_out_modes(ss.holdout)
    rows = []
    for name, v in ss.scores.items():
        meta = ss.meta_for(name); modes = meta['mode'].to_numpy(); ood = meta['kind'].to_numpy() == 'OOD'
        row = {'score': name, 'windows': len(v)}
        row.update({f'median {t}': float(np.median(v[modes == t])) if np.any(modes == t) else np.nan for t in TASKS})
        row['AUROC OOD vs ID'] = float(roc_auc_score(ood, v)) if ood.any() and (~ood).any() else np.nan
        for t in out:
            sel = (~ood) | (modes == t)
            row[f'AUROC {t} vs ID'] = float(roc_auc_score(ood[sel], v[sel])) if np.any(modes == t) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write(ss, summary, results=RESULTS):
    out = Path(results) / ss.holdout; out.mkdir(parents=True, exist_ok=True)
    ss.aligned().to_parquet(out / 'scores_test.parquet', index=False)
    for name in ss.metas:
        pd.concat([ss.metas[name].reset_index(drop=True), pd.DataFrame({name: ss.scores[name]})], axis=1) \
            .to_parquet(out / f'scores_test_{name}.parquet', index=False)
    summary.to_csv(out / 'score_summary.csv', index=False)
    from vae_sweep import markdown_table
    shown = summary.copy()
    text = (f'# Per-window scores under the `{ss.holdout}` holdout (test split)\n\n'
            f'Held out: {", ".join(held_out_modes(ss.holdout))}; trained on {", ".join(trained_modes(ss.holdout))}. '
            'Every score is causally median-filtered within trials. AUROC is a ranking statistic (no threshold); '
            'OOD is the positive class. Medians are in each score\'s own units.\n\n' + markdown_table(shown, '{:.4g}') + '\n')
    if ss.notes:
        text += '\nNot available yet: ' + '; '.join(ss.notes) + '\n'
    (out / 'score_summary.md').write_text(text)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--holdout', required=True)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--no-kv', action='store_true', help='do not recompute the forecast error terms')
    parser.add_argument('--device', default=None)
    a = parser.parse_args(argv)
    ss = collect(a.holdout, recompute_kv=not a.no_kv, device=a.device)
    summary = summarize(ss)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    cols = ['score', 'windows', 'AUROC OOD vs ID'] + [c for c in summary if c.startswith('AUROC ') and c != 'AUROC OOD vs ID']
    print(summary[cols].round(4).to_string(index=False))
    if ss.notes:
        print('not available yet:', '; '.join(ss.notes))
    if a.write:
        print('wrote', write(ss, summary))


if __name__ == '__main__':
    main()
