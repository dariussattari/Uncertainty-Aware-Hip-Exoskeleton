"""Post-hoc, no training: does LOF work better in fewer latent dimensions?

    python ml/VAE/posthoc_dims.py            # print
    python ml/VAE/posthoc_dims.py --write    # also results/posthoc_dims.{csv,md}

Two questions, both answered with LOF refitted on in-distribution TRAINING latents and scored on
the validation and test splits (validation is the selection column):

1. VAE runs: LOF on all 32 posterior-mean coordinates (reproduces the saved score), on the
   *active* dimensions only (median posterior sigma^2 < 0.9 on ID validation windows), and on
   PCA-reduced means with k in {8, 12, 16, 18, 24}.
2. The AE baseline: LOF on its 32-D latent and on PCA-reduced latents with the same k.

If the AE gains at 16-18 dimensions, beta=4's advantage over beta=1 is a dimension effect, not
a regularization effect. PCA and LOF are fitted on the same 50,000 training latents the
production LOF used (same seed), so the k=32 rows should reproduce the saved AUROCs.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from vae_paths import ROOT, RUNS
from vae_sweep import RESULTS, active_dimensions, markdown_table
from models.autoencoder import LatentLOF
from evaluation.common import smooth_by_trial

KS = (8, 12, 16, 18, 24)
SMOOTH = 10


def _auc(scores, labels, meta):
    return float(roc_auc_score(labels, smooth_by_trial(scores, meta, SMOOTH, 'median')))


def _score(reference, splits, transform=None, seed=0):
    """Fit LOF on (transformed) reference latents; AUROC on each split (already smoothed per trial)."""
    ref = reference if transform is None else transform(reference)
    lof = LatentLOF(fit_samples=None, seed=seed).fit(ref)
    return {name: _auc(lof.score(Z if transform is None else transform(Z)), lab, meta)
            for name, (Z, lab, meta) in splits.items()}


def _rows(model_name, reference, splits, active=None, seed=0):
    rows = [{'model': model_name, 'representation': 'all 32 dims', 'dims': reference.shape[1],
             **{f'{k} AUROC': v for k, v in _score(reference, splits, seed=seed).items()}}]
    if active is not None:
        sel = np.flatnonzero(active)
        rows.append({'model': model_name, 'representation': 'active dims only', 'dims': len(sel),
                     **{f'{k} AUROC': v for k, v in _score(reference, splits, lambda Z: Z[:, sel], seed=seed).items()}})
    for k in KS:
        pca = PCA(n_components=k, random_state=seed).fit(reference)
        rows.append({'model': model_name, 'representation': f'PCA-{k}', 'dims': k,
                     **{n: v for n, v in ((f'{s} AUROC', v) for s, v in _score(reference, splits, pca.transform, seed=seed).items())},
                     'variance kept': float(pca.explained_variance_ratio_.sum())})
    return rows


def vae_splits(run):
    out = {}
    for split in ('val', 'test'):
        with np.load(run / f'latents_{split}.npz') as z:
            out[split] = (z['Z'].astype(np.float32), z['labels'], pd.read_parquet(run / f'latents_{split}.parquet'))
    return out


def ae_reference_and_splits(run, seed=0):
    """Encode the AE's training (ID) and validation windows; test comes from the saved bundle."""
    import sys
    sys.path.insert(0, str(ROOT / 'ml/vanilla'))
    from dataset import AutoencoderData
    from evaluation.autoencoder import load_ae_run, extract_latents
    from training.common import pick_device
    from training.autoencoder import encode_split
    data = AutoencoderData(batch_size=1024); device = pick_device()
    model, lof, _ = load_ae_run(run, data, device)
    train_Z, _ = encode_split(model, data.loader('train'), device)
    rng = np.random.default_rng(seed)      # LatentLOF(fit_samples=50_000, seed=0) subsamples the same way
    if len(train_Z) > 50_000:
        train_Z = train_Z[rng.choice(len(train_Z), 50_000, replace=False)]
    Zv, _, _, labv, metav, _ = extract_latents(model, lof, data, 'val', device)
    with np.load(run / 'latents_test.npz') as z:
        test = (z['Z'].astype(np.float32), z['labels'], pd.read_parquet(run / 'latents_test.parquet'))
    return np.asarray(train_Z, dtype=np.float32), {'val': (np.asarray(Zv, dtype=np.float32), labv, metav), 'test': test}


def run_all(vae_runs=('original_standard_seed0', 'original_beta4_seed0'), ae_run='ml/vanilla/runs/autoencoder'):
    rows = []
    for name in vae_runs:
        run = RUNS / name
        if not (run / 'latents_val.npz').exists():
            print('skipping', name, '(no saved latents)'); continue
        t0 = time.time()
        n_active, active, _ = active_dimensions(run)
        rows += _rows(f'VAE {name}', np.load(run / 'train_reference_mu.npy').astype(np.float32), vae_splits(run), active)
        print(f'{name}: {n_active} active dims, {time.time() - t0:.0f}s', flush=True)
    ae = ROOT / ae_run
    if (ae / 'final.pt').exists():
        t0 = time.time()
        reference, splits = ae_reference_and_splits(ae)
        rows += _rows('AE ' + ae.name, reference, splits)
        print(f'AE: {time.time() - t0:.0f}s', flush=True)
    else:
        print('AE run not found; skipping the AE rows')
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true')
    a = parser.parse_args(argv)
    torch.set_num_threads(4)
    df = run_all()
    pd.set_option('display.width', 200)
    print(df.round(4).to_string(index=False))
    if a.write:
        RESULTS.mkdir(exist_ok=True)
        df.to_csv(RESULTS / 'posthoc_dims.csv', index=False)
        (RESULTS / 'posthoc_dims.md').write_text(
            '# LOF in fewer dimensions (post hoc, no retraining)\n\n'
            'LOF refitted on 50,000 in-distribution training latents (same seed as the production fit) and scored on '
            'validation and test with the usual per-trial median filter. Select on validation; test is reported only. '
            '"Active dims" are those with median posterior σ² < 0.9 on ID validation windows.\n\n'
            + markdown_table(df) + '\n')
        print('wrote', RESULTS / 'posthoc_dims.md')


if __name__ == '__main__':
    main()
