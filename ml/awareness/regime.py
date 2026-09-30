"""Stage B: regime probes on the encoder latent, trained on in-distribution task labels only.

    python ml/awareness/regime.py --holdout DOWN

Two probes read the same 32-D latent: a multinomial logistic regression (the linear probe,
which answers whether regime is linearly readable from the representation) and a small MLP
(the comparison). Both are trained on the training split's windows with their task labels
mapped to regimes (``aw_paths.REGIMES``); under the DOWN holdout only ``level`` and ``ascend``
exist, so a held-out descending window can never be named. What is recorded instead is what
the probe *does* with it: its class probabilities, confidence (max probability) and entropy,
for every window of every split, in the same row order as ``encoder.py`` wrote the latents.

Outputs, under ``runs/<holdout>/regime/``: ``<probe>_<split>_<kind>.npz`` with ``probs``,
``confidence``, ``entropy`` and ``classes``; ``<probe>.joblib``; ``<probe>_summary.json``
with ID accuracy per split and per-mode mean entropy and predicted-class shares.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aw_paths import REGIMES, RUNS, SPLITS, run_dir

PROBES = {
    'logistic': lambda seed: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
    'mlp': lambda seed: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True,
                                                                        max_iter=300, random_state=seed)),
}


def load_latents(encoded: Path, split: str, kind: str):
    mu = np.load(encoded / f'{split}_{kind}_mu.npy')
    meta = pd.read_parquet(encoded / f'{split}_{kind}_meta.parquet')
    if len(mu) != len(meta):
        raise ValueError(f'{split}/{kind}: latent and meta lengths differ')
    return mu, meta


def entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(1)


def fit_and_score(holdout: str, encoder_out: Path, out: Path, probes=PROBES, seed: int = 0,
                  max_train: int | None = None):
    encoded = Path(encoder_out) / 'encoded'
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    mu_tr, meta_tr = load_latents(encoded, 'train', 'id')
    y_tr = meta_tr['mode'].map(REGIMES).to_numpy()
    if max_train is not None and len(mu_tr) > max_train:
        pick = np.sort(np.random.default_rng(seed).choice(len(mu_tr), max_train, replace=False))
        mu_tr, y_tr = mu_tr[pick], y_tr[pick]
    classes = sorted(set(y_tr))
    # Integer-coded targets: scikit-learn 1.7's MLP early stopping cannot score string labels.
    code = {c: i for i, c in enumerate(classes)}
    y_tr_i = np.array([code[c] for c in y_tr])
    print(f'regime probes on {len(mu_tr):,} training latents; classes {classes}', flush=True)
    summaries = {}
    for name, make in probes.items():
        clf = make(seed).fit(mu_tr, y_tr_i)
        joblib.dump({'pipeline': clf, 'classes': classes}, out / f'{name}.joblib')
        assert list(clf.classes_) == list(range(len(classes)))
        summary = {'classes': classes, 'n_train': int(len(mu_tr))}
        for split, kind in SPLITS:
            mu, meta = load_latents(encoded, split, kind)
            P = clf.predict_proba(mu).astype(np.float32)
            conf, ent = P.max(1), entropy(P)
            pred = np.array(classes)[P.argmax(1)]
            truth = meta['mode'].map(REGIMES).to_numpy()
            np.savez_compressed(out / f'{name}_{split}_{kind}.npz', probs=P, confidence=conf, entropy=ent,
                                classes=np.array(classes))
            rows = {}
            for mode, g in meta.groupby('mode', sort=False):
                idx = g.index.to_numpy()
                rows[mode] = {'windows': int(len(idx)), 'mean_entropy': float(ent[idx].mean()),
                              'mean_confidence': float(conf[idx].mean()),
                              'predicted_share': {c: float(np.mean(pred[idx] == c)) for c in classes}}
            known = np.isin(truth, classes)
            summary[f'{split}_{kind}'] = {'n': int(len(mu)),
                                          'accuracy': float(np.mean(pred[known] == truth[known])) if known.any() else None,
                                          'by_mode': rows}
            acc = summary[f'{split}_{kind}']['accuracy']
            print(f'  {name:9s} {split}/{kind}: n={len(mu):,}'
                  + (f'  accuracy {acc:.4f}' if acc is not None else '')
                  + '  mean entropy by mode: ' + ' '.join(f'{m}={r["mean_entropy"]:.3f}' for m, r in rows.items()), flush=True)
        (out / f'{name}_summary.json').write_text(json.dumps(summary, indent=2))
        summaries[name] = summary
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--holdout', required=True)
    parser.add_argument('--out-root', default=str(RUNS))
    parser.add_argument('--probes', nargs='*', default=list(PROBES), choices=list(PROBES))
    parser.add_argument('--max-train', type=int, default=None, help='subsample training latents (speed)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--encoder-stage', default='encoder', help='which encoder run to read latents from')
    parser.add_argument('--stage', default='regime', help='output directory name under runs/<holdout>/')
    a = parser.parse_args(argv)
    fit_and_score(a.holdout, run_dir(a.holdout, a.encoder_stage, a.out_root), run_dir(a.holdout, a.stage, a.out_root),
                  {k: PROBES[k] for k in a.probes}, a.seed, a.max_train)


if __name__ == '__main__':
    main()
