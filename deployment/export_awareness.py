"""Fit the awareness layer as notebook 13 (E39) did and save one bundle per seed to deployment/artifacts/.

The training fit: novelty scores on the fit part of the LSTM-VAE's exported ID-safe training windows; the boosted-tree
regime classifier on the 14 training participants (without the novelty scores' own fit windows); the two u cutoffs on
the 4 validation participants. Each bundle is then run through ``awareness.py`` (window by window for the smoothing,
as on the device) and must reproduce, to 1e-9, the notebook's test-participant results in
``onboard/results/13_regime_scores.csv`` (train fit) and ``13_uncertainty_scores.csv``. Nothing is saved if it does not.

Needs the local data (data/processed_nature/ae, the runs' latents*.npz, the novelty cache). Run from the repo root:

    .venv/bin/python deployment/export_awareness.py
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

REPO = Path(__file__).resolve().parents[1]
ONBOARD = REPO / 'notebooks' / 'nature' / 'model_exploration' / 'onboard'
for folder in (REPO / 'ml' / 'common', ONBOARD, REPO / 'deployment'):
    sys.path.insert(0, str(folder))
import _novelty_scores as ns  # noqa: E402  (the notebook's helpers: data loading and the reference scores)
import _regime_classifier as rc  # noqa: E402
import awareness as aw  # noqa: E402
from models.autoencoder import LatentLOF  # noqa: E402

ROOT = REPO / 'notebooks' / 'nature' / 'model_exploration'
DATA = REPO / 'data' / 'processed_nature' / 'ae'
CACHE = REPO / 'data' / 'cache' / 'nature' / 'novelty'
RESULTS = ONBOARD / 'results'
OUT = REPO / 'deployment' / 'artifacts'
SEEDS = (0, 1, 2)
SPLITS = ('train', 'val', 'test')
ENCODER = ROOT / 'runs_train_class' / 'id_ood_safe' / 'vae' / 'tcn_vae_raw_3s_spectral_b128_fixed_s{seed}'
DETECTOR = ROOT / 'runs' / 'vae' / 'lstm_vae_raw_3s_b1_fixed_s{seed}'
CLASSIFIER = {'class_weight': 'balanced', 'max_iter': 300}                     # notebook 06's choice, random state = seed


def get_detector_latents(run: Path, table: pd.DataFrame, keys: pd.DataFrame, split: str
                         ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """The LSTM-VAE's z and logvar for the windows in ``keys``, from the export that holds that split. (N, 32) x 2."""
    export = np.load(run / {'train': 'latents_train.npz', 'val': 'latents_val.npz', 'test': 'latents.npz'}[split])
    pick = ~export['is_train'] if split == 'test' else np.ones(len(export['rows']), bool)   # latents.npz: test part
    labels = ns.get_labels(table, export['rows'][pick])
    return tuple(rc.get_aligned(export[k][pick], labels, keys) for k in ('z', 'logvar'))


def get_streamed(proba: npt.NDArray, labels: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Smoothed probabilities as the device computes them: each recording's windows pushed in time order through one
    ``CausalMedian``. (N, 3) -> (N, 3), rows in ``labels`` order."""
    out = np.empty_like(proba, dtype = np.float64)
    for _, idx in labels.groupby('rec', sort = False).groups.items():
        idx = np.asarray(idx)[np.argsort(labels['end'].to_numpy()[idx], kind = 'stable')]
        smoother = aw.CausalMedian()
        for i in idx:
            out[i] = smoother.push(int(labels['end'].iat[i]), proba[i])
    return out


def get_git_sha() -> str:
    sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd = REPO, capture_output = True, text = True, check = True)
    dirty = subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd = REPO).returncode != 0
    return sha.stdout.strip() + ('-dirty' if dirty else '')


def export_seed(seed: int, tables: dict[str, pd.DataFrame], rows: dict[str, pd.DataFrame], y: dict[str, npt.NDArray],
                expected: dict[str, pd.Series]) -> aw.AwarenessBundle:
    """Fit, check against notebook 13 and return one seed's bundle."""
    enc, det = Path(str(ENCODER).format(seed = seed)), Path(str(DETECTOR).format(seed = seed))
    keys = {s: r[['rec', 'end']] for s, r in rows.items()}

    # novelty fits on the fit part of the detector's exported training windows (ns.get_native_scores' protocol)
    export = np.load(det / 'latents.npz')
    fit_idx, _ = ns.get_fit_split(int(export['is_train'].sum()))
    z_fit, logvar_fit = export['z'][export['is_train']][fit_idx], export['logvar'][export['is_train']][fit_idx]
    lof = LatentLOF(n_neighbors = aw.LOF_NEIGHBORS, fit_samples = None, seed = ns.SPLIT_SEED).fit(z_fit.astype(np.float32)).lof
    novelty_fit = aw.fit_novelty(lof, z_fit, logvar_fit)

    # features for every split through awareness.py, checked against the notebook's cached novelty scores
    reference = rc.get_novelty(det, tables['raw_3s'], {s: keys[s] for s in ('val', 'test')}, list(aw.NOVELTY), CACHE)
    reference['train'] = rc.get_split_novelty(det, tables['raw_3s'], keys['train'], list(aw.NOVELTY), 'train', CACHE)
    tcn = rc.get_latents(enc, tables['raw_3s_spectral'], {s: keys[s] for s in ('val', 'test')})
    tcn['train'] = rc.get_split_latents(enc, tables['raw_3s_spectral'], keys['train'], 'train')
    x = {}
    for s in SPLITS:
        novelty = aw.score_novelty(novelty_fit, *get_detector_latents(det, tables['raw_3s'], keys[s], s))
        np.testing.assert_allclose(novelty, reference[s], rtol = 1e-6, err_msg = f'seed {seed}, {s}: novelty scores')
        x[s] = aw.get_features(tcn[s], novelty, rows[s][list(aw.RHYTHM)].to_numpy(dtype = np.float64))

    # the classifier on the training participants without the novelty fit windows (their scores are in-sample)
    fit_windows = rc.get_score_fit_windows(det, tables['raw_3s'])
    in_sample = keys['train'].merge(fit_windows.assign(hit = True), on = ['rec', 'end'], how = 'left')['hit'].notna().to_numpy()
    classifier = HistGradientBoostingClassifier(**CLASSIFIER, random_state = seed).fit(x['train'][~in_sample], y['train'][~in_sample])

    # cutoffs on validation; the test participants scored once, as on the device, against the notebook
    proba = {s: get_streamed(aw.get_proba(classifier, x[s]), rows[s]) for s in ('val', 'test')}
    np.testing.assert_allclose(proba['test'], rc.get_smoothed(aw.get_proba(classifier, x['test']), rows['test']), atol = 1e-12)
    cutoffs = rc.get_cutoffs(aw.get_uncertainty(proba['val']), y['val'])
    argmax = rc.get_regime_metrics(y['test'], np.array(aw.REGIMES)[proba['test'].argmax(axis = 1)])
    banded = rc.get_regime_metrics(y['test'], aw.get_regime(aw.get_uncertainty(proba['test']), cutoffs))
    got = {'cutoff 1': cutoffs[0], 'cutoff 2': cutoffs[1], **{f'u {k}': v for k, v in banded.items()},
           **{f'argmax {k}': v for k, v in argmax.items()}}
    want = {k: expected['u'].loc[seed, k] for k in ('cutoff 1', 'cutoff 2')}
    want.update({f'u {k}': expected['u'].loc[seed, k] for k in banded})
    want.update({f'argmax {k}': expected['argmax'].loc[seed, k] for k in argmax})
    for k in got:
        np.testing.assert_allclose(got[k], want[k], atol = 1e-9, err_msg = f'seed {seed}: {k}')
    print(f"seed {seed}: cutoffs {cutoffs}, test macro F1 {banded['macro F1']:.4f} (u), {argmax['macro F1']:.4f} (argmax); "
          'matches notebook 13')

    meta = {'seed': seed, 'encoder_run': str(enc.relative_to(REPO)), 'detector_run': str(det.relative_to(REPO)),
            'fit': 'train (14 participants), u cutoffs on val (4)', 'n_classifier_windows': int((~in_sample).sum()),
            'n_novelty_fit_windows': len(z_fit), 'test_macro_f1_u': banded['macro F1'], 'git_sha': get_git_sha(),
            'data_version': json.loads((det / 'config.json').read_text())['data_version'],
            'sklearn': sklearn.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'features': [f'tcn_z{i}' for i in range(aw.LATENT_DIM)] + list(aw.NOVELTY) + list(aw.RHYTHM)}
    return aw.AwarenessBundle(novelty = novelty_fit, classifier = classifier, cutoffs = cutoffs, meta = meta)


def main() -> None:
    t0 = time.time()
    meta = json.loads((DATA / 'meta.json').read_text())
    tables = {k: pd.read_parquet(DATA / meta['inputs'][k]['table']) for k in ('raw_3s', 'raw_3s_spectral')}
    rows = {s: rc.get_split_rows(tables['raw_3s'], s) for s in SPLITS}
    y = {s: r['controller_class'].astype(str).to_numpy() for s, r in rows.items()}
    scores = pd.read_csv(RESULTS / '13_regime_scores.csv')
    expected = {'argmax': scores[scores['fit'] == 'train'].set_index('seed'),
                'u': pd.read_csv(RESULTS / '13_uncertainty_scores.csv').set_index('seed')}
    bundles = [export_seed(seed, tables, rows, y, expected) for seed in SEEDS]       # every seed checked before saving
    OUT.mkdir(exist_ok = True)
    for bundle in bundles:
        path = OUT / f"awareness_s{bundle.meta['seed']}.pkl"
        aw.save_bundle(bundle, path)
        loaded = aw.load_bundle(path)
        x = np.zeros((1, aw.N_FEATURES))
        np.testing.assert_allclose(aw.get_proba(loaded.classifier, x), aw.get_proba(bundle.classifier, x))
        print(f'wrote {path.relative_to(REPO)}')
    print(f'{time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
