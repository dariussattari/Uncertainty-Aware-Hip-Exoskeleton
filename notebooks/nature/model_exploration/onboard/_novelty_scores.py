"""Novelty scores for every trained latent model, and how well they flag unfamiliar movement (05_Novelty_Metric_comparison).

Any run folder under ``runs/`` holding ``config.json``, ``summary.json`` and ``latents.npz`` (``rows``, ``rec``,
``end``, ``z``, ``recon_mse``, ``is_train``; see ``ml/nature/awareness/train_ae.py``) is picked up, so a new model
family only has to export that format. Scores are registered in ``SCORES``; each sees the run's exported training
windows, split once into a fit set and a calibration set, and returns scores where larger means more unfamiliar.

Protocol (fixed before any model was scored)
  fit / calibrate   the run's exported training windows (ID-safe tasks, training participants), split 75 / 25 with a
                    fixed seed: the score is fitted on the first part, its threshold set on the second
  threshold         the 99.5th percentile of calibration scores (the ankle paper's rule)
  test              in-task windows of the four test participants present in every run's export; familiar = the
                    ID-safe tasks, unfamiliar = every other task
  filter            the ankle paper's causal median over the last 0.5 s of windows of the same recording, applied
                    before the threshold

Held-out operating point (notebook 05, section 8; needs ``latents_val.npz`` from ``ml/nature/awareness/encode_split.py``)
  representation    the latent as exported ('raw'), every unit standardized by the fit set (None), or that standardized
                    latent projected to k principal components fitted on the fit set (``PCA_KS``)
  threshold         the filtered score that flags ``TARGET_FA_PCT`` % of the familiar windows of the validation
                    participants; k is chosen on the validation participants too, the test participants are scored once

Native metrics (section 7): every score a model family offers (``NATIVE``), fitted where needed on the fit part of the
exported training windows, oriented and thresholded on the validation participants, scored once on the test ones.

Training-class ablation (notebook 07, ``evaluate_dangerous``): the same native scores of models trained on ID-safe,
ID-safe + OOD-safe or OOD-safe movement, judged on OOD-dangerous against OOD-safe windows (ID is left to a heuristic);
oriented and thresholded (1% of OOD-safe windows) on the validation participants, scored once on the test ones.
"""
from collections.abc import Callable
import hashlib
import os
import json
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'ml' / 'common').is_dir())
if str(REPO / 'ml' / 'common') not in sys.path:
    sys.path.insert(0, str(REPO / 'ml' / 'common'))
from models.autoencoder import LOF_NEIGHBORS, THRESHOLD_PERCENTILE, LatentLOF  # noqa: E402  (the ankle paper's LOF)

FIT_SHARE = 0.75
SPLIT_SEED = 0
FILTER_S = 0.5
RATE_HZ = 200
INPUT_LABELS = {'raw_1s': '1 s raw', 'raw_1s_spectral': '1 s + spectral', 'raw_3s': '3 s raw', 'raw_3s_spectral': '3 s + spectral'}
INPUT_ORDER = ('raw_1s', 'raw_1s_spectral', 'raw_3s', 'raw_3s_spectral')         # the 2 x 2 input ablation: window x spectral
ARCH_LABELS = {'conv': 'VAE', 'lstm': 'LSTM-VAE', 'transformer': 'Transformer VAE', 'tcn': 'TCN-VAE'}
MODEL_TYPES = ('plain AE', 'sparse AE', 'Gaussian-NLL AE', 'conv VAE', 'LSTM-VAE', 'Transformer VAE', 'TCN-VAE')


# ----------------------------------------------------------------------------- runs

def get_model_label(config: dict, stem: str) -> str:
    """Readable model name from a run's config, e.g. 'sparse · 3 s + spectral · λ 0.1 · 8x4'; the run stem otherwise."""
    result = stem
    if 'variant' in config:
        parts = [config['variant'], INPUT_LABELS.get(config.get('inputs', 'raw_1s'), config.get('inputs', ''))]
        if config.get('sparsity_weight', 0):
            parts.append(f"λ {config['sparsity_weight']:g}")
        if config.get('latent', '4x8') != '4x8':
            parts.append(config['latent'])
        result = ' · '.join(parts)
    elif config.get('model') == 'vae':
        result = f"{ARCH_LABELS[config.get('arch', 'conv')]} · β {config['beta']:g} · {config['decoder_variance']} variance"
        if config.get('inputs', 'raw_1s') != 'raw_1s':
            result += f" · {INPUT_LABELS.get(config['inputs'], config['inputs'])}"
    return result


def get_model_type(config: dict) -> str:
    """The model type a run belongs to in the full comparison (MODEL_TYPES)."""
    if config.get('model') == 'vae':
        result = {'conv': 'conv VAE', 'lstm': 'LSTM-VAE', 'transformer': 'Transformer VAE', 'tcn': 'TCN-VAE'}[config.get('arch', 'conv')]
    else:
        result = {'vanilla': 'plain AE', 'sparse': 'sparse AE', 'gaussian': 'Gaussian-NLL AE'}[config['variant']]
    return result


def get_family(config: dict) -> str:
    """'vae' for the VAE grid, else the autoencoder variant ('vanilla', 'sparse', 'gaussian')."""
    return 'vae' if config.get('model') == 'vae' else config.get('variant', 'unknown')


def discover_runs(root: Path) -> pd.DataFrame:
    """Every exported run under ``root``: its folder, model stem (folder minus ``_s<seed>``), seed, label and config."""
    rows = []
    for latents in sorted(root.rglob('latents.npz')):
        run = latents.parent
        if (run / 'config.json').exists() and (run / 'summary.json').exists() and not run.name.endswith('_smoke'):
            config = json.loads((run / 'config.json').read_text())
            stem, _, seed = run.name.rpartition('_s')
            rows.append({'run': run, 'stem': stem, 'seed': int(seed), 'model': get_model_label(config, stem),
                         'family': run.parent.name, 'inputs': config.get('inputs', 'raw_1s'), 'config': config})
    return pd.DataFrame(rows)


LABEL_COLS = ['rec', 'end', 'subject', 'task', 'controller_class', 'family', 'direction']   # what the notebooks read
LAZY_KEYS = ('z', 'logvar')                          # (N, D) arrays read from disk only when a score has to be computed


def get_labels(windows: pd.DataFrame, rows: npt.NDArray) -> pd.DataFrame:
    """The label columns the notebooks use, for some window rows (all columns would cost about 5x the memory)."""
    return windows.iloc[rows][[c for c in LABEL_COLS if c in windows.columns]].reset_index(drop = True)


def load_export(run: Path, windows: pd.DataFrame, file: str = 'latents.npz', is_train: npt.NDArray | None = None) -> dict:
    """A run's export with its window labels: recon_mse (N,), is_train (N,), labels (N rows), and for a VAE nll and kl
    (N,). The latent ``z`` and the VAE's ``logvar`` (N, D) stay on disk (``get_array``, ``materialize``): with every run
    of the comparison loaded at once they would not fit in memory."""
    data = np.load(run / file)
    small = {k: data[k].astype(np.float64) for k in ('recon_mse', 'nll', 'kl') if k in data.files}
    return {'file': run / file, 'keys': tuple(data.files), 'labels': get_labels(windows, data['rows']), **small,
            'is_train': (data['is_train'] if is_train is None else is_train).astype(bool)}


def get_array(export: dict, key: str) -> npt.NDArray:
    """One array of an export, read from disk if it is not held in memory (z and logvar as float32)."""
    result = export.get(key)
    if result is None:
        result = np.load(export['file'])[key].astype(np.float32)
    return result


def materialize(export: dict) -> dict:
    """The export with its latent (and posterior log variance, for a VAE) loaded, for computing a score."""
    return {**export, **{k: get_array(export, k) for k in LAZY_KEYS if k in export['keys'] and k not in export}}


def get_fit_split(n_train: int, seed: int = SPLIT_SEED) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Indices (within the training windows) of the fit and calibration parts."""
    order = np.random.default_rng(seed).permutation(n_train)
    cut = int(FIT_SHARE * n_train)
    return np.sort(order[:cut]), np.sort(order[cut:])


# ----------------------------------------------------------------------------- scores

def score_lof(fit: npt.NDArray, others: list[npt.NDArray], seed: int) -> list[npt.NDArray[np.float64]]:
    """LOF (k = 20) fitted on ``fit``, scored on each array in ``others``; larger = more unfamiliar."""
    lof = LatentLOF(n_neighbors = LOF_NEIGHBORS, fit_samples = None, seed = seed).fit(fit)
    return [lof.score(o) for o in others]


def score_lof_raw(export: dict, fit_idx: npt.NDArray, cal_idx: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """The ankle paper's score: LOF on the raw latent."""
    train, test = export['z'][export['is_train']], export['z'][~export['is_train']]
    cal, out = score_lof(train[fit_idx], [train[cal_idx], test], SPLIT_SEED)
    return cal, out


def score_lof_standardized(export: dict, fit_idx: npt.NDArray, cal_idx: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """LOF on the latent scaled per unit by the fit set's mean and SD (units constant there dropped), so a block of
    large co-firing units cannot dominate the distances."""
    train, test = export['z'][export['is_train']], export['z'][~export['is_train']]
    mean, sd = train[fit_idx].mean(axis = 0), train[fit_idx].std(axis = 0)
    keep = sd > 1e-6
    scale = lambda x: (x[:, keep] - mean[keep]) / sd[keep]                            # noqa: E731
    cal, out = score_lof(scale(train[fit_idx]), [scale(train[cal_idx]), scale(test)], SPLIT_SEED)
    return cal, out


def score_reconstruction(export: dict, fit_idx: npt.NDArray, cal_idx: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Per-window reconstruction MSE: the ankle paper's rejected score, kept as a reference."""
    train, test = export['recon_mse'][export['is_train']], export['recon_mse'][~export['is_train']]
    return train[cal_idx], test


SCORES: dict[str, Callable] = {'LOF, raw latent': score_lof_raw,
                               'LOF, standardized latent': score_lof_standardized,
                               'reconstruction error': score_reconstruction}


def save_cache(path: Path, **arrays: npt.NDArray) -> None:
    """np.savez through a temporary file and an atomic rename, so a reader never sees a half-written cache entry."""
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp')
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def get_scores(run: Path, export: dict, score: str, cache: Path) -> tuple[npt.NDArray, npt.NDArray]:
    """(calibration, test) scores of one run, cached by the export's content and the protocol constants."""
    stamp = hashlib.sha256((run / 'latents.npz').read_bytes()).hexdigest()[:16]
    key = hashlib.sha256(f'{stamp}|{score}|{FIT_SHARE}|{SPLIT_SEED}|{LOF_NEIGHBORS}'.encode()).hexdigest()[:16]
    path = cache / f'{key}.npz'
    if path.exists():
        saved = np.load(path)
        result = (saved['cal'], saved['test'])
    else:
        fit_idx, cal_idx = get_fit_split(int(export['is_train'].sum()))
        result = SCORES[score](materialize(export), fit_idx, cal_idx)
        save_cache(path, cal = result[0], test = result[1])
    return result


# ----------------------------------------------------------------------------- evaluation

def get_causal_median(scores: npt.NDArray, rec: npt.NDArray, end: npt.NDArray, seconds: float = FILTER_S) -> npt.NDArray:
    """Median of each window's score and those of the same recording ending in the previous ``seconds`` (causal)."""
    df = pd.DataFrame({'rec': rec, 't': pd.to_timedelta(np.asarray(end) / RATE_HZ, unit = 's'), 's': scores,
                       'i': np.arange(len(scores))}).sort_values(['rec', 't'])
    rolled = df.set_index('t').groupby('rec')['s'].rolling(f'{int(seconds * 1000)}ms').median()   # (N,) in df order
    out = np.empty(len(scores))
    out[df['i'].to_numpy()] = rolled.to_numpy()
    return out


def evaluate_at(threshold: float, test: npt.NDArray, labels: pd.DataFrame, familiar: tuple[str, ...]) -> tuple[dict, pd.Series]:
    """Detection metrics on a split's windows at a given threshold, and the share of each task's windows flagged.

    AUROC uses the per-window scores; flagging uses the 0.5 s causal median against the threshold.
    """
    tasks = labels['task'].astype(str).to_numpy()
    cls = labels['controller_class'].astype(str).to_numpy()
    is_fam = np.isin(tasks, familiar)
    flagged = get_filtered(test, labels) > threshold
    dangerous = cls == 'OOD-dangerous'
    safe = cls == 'OOD-safe'
    auroc = lambda m: roc_auc_score(np.r_[np.zeros(is_fam.sum()), np.ones(m.sum())], np.r_[test[is_fam], test[m]])  # noqa: E731
    metrics = {'AUROC, unfamiliar vs familiar': auroc(~is_fam), 'AUROC, OOD-safe vs familiar': auroc(safe),
               'AUROC, OOD-dangerous vs familiar': auroc(dangerous),
               'flagged, unfamiliar (%)': 100 * flagged[~is_fam].mean(), 'flagged, OOD-dangerous (%)': 100 * flagged[dangerous].mean(),
               'false alarms, familiar (%)': 100 * flagged[is_fam].mean()}
    per_task = pd.Series(flagged).groupby(tasks).mean() * 100
    return metrics, per_task


def evaluate(cal: npt.NDArray, test: npt.NDArray, labels: pd.DataFrame, familiar: tuple[str, ...]) -> tuple[dict, pd.Series]:
    """``evaluate_at`` with the ankle paper's threshold: the 99.5th percentile of the calibration scores."""
    return evaluate_at(float(np.percentile(cal, THRESHOLD_PERCENTILE)), test, labels, familiar)


# ----------------------------------------------------------------------------- held-out operating point (section 8)

VAL_SPLIT = 'val'
PCA_KS = (4, 6, 8, 12, 16, 24)        # principal components of the standardized latent LOF runs in; None = every unit
TARGET_FA_PCT = 1.0                   # share of familiar validation windows the tuned threshold may flag


def load_split_export(run: Path, windows: pd.DataFrame, split: str = VAL_SPLIT) -> dict | None:
    """A split's export written by ``ml/nature/awareness/encode_split.py`` (same keys as ``load_export``), or None."""
    path = run / f'latents_{split}.npz'
    result = None
    if path.exists():
        rows = np.load(path)['rows']
        result = load_export(run, windows, path.name, is_train = np.zeros(len(rows), bool))
    return result


def get_representation(fit: npt.NDArray, k: int | None) -> Callable[[npt.NDArray], npt.NDArray]:
    """Standardize each unit by the fit set (constant units dropped), then project to ``k`` principal components.

    ``k = None`` keeps every unit; so does a ``k`` at or above the number of units kept, since LOF uses Euclidean
    distances and a full rotation leaves them unchanged.
    """
    mean, sd = fit.mean(axis = 0), fit.std(axis = 0)
    keep = sd > 1e-6
    scale = lambda x: (x[:, keep] - mean[keep]) / sd[keep]                             # noqa: E731
    transform = scale
    if k is not None and k < int(keep.sum()):
        pca = PCA(n_components = k, random_state = SPLIT_SEED).fit(scale(fit))
        transform = lambda x: pca.transform(scale(x))                                  # noqa: E731
    return transform


def get_split_scores(run: Path, export: dict, val: dict, representation: int | None | str, cache: Path
                     ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """(calibration, validation, test) LOF scores of one run in one representation, cached by the exports' content.

    ``representation`` is ``'raw'`` (the latent as exported), ``None`` (every unit standardized) or a number of
    principal components. LOF is fitted on the fit part of the exported training windows, as in ``get_scores``.
    """
    stamp = hashlib.sha256((run / 'latents.npz').read_bytes() + (run / f'latents_{VAL_SPLIT}.npz').read_bytes()).hexdigest()[:16]
    key = hashlib.sha256(f'{stamp}|split-lof|{representation}|{FIT_SHARE}|{SPLIT_SEED}|{LOF_NEIGHBORS}'.encode()).hexdigest()[:16]
    path = cache / f'{key}.npz'
    if path.exists():
        saved = np.load(path)
        result = (saved['cal'], saved['val'], saved['test'])
    else:
        fit_idx, cal_idx = get_fit_split(int(export['is_train'].sum()))
        export, val = materialize(export), materialize(val)
        train, test = export['z'][export['is_train']], export['z'][~export['is_train']]
        transform = (lambda x: x) if representation == 'raw' else get_representation(train[fit_idx], representation)
        scored = score_lof(transform(train[fit_idx]), [transform(train[cal_idx]), transform(val['z']), transform(test)], SPLIT_SEED)
        result = tuple(scored)
        save_cache(path, cal = result[0], val = result[1], test = result[2])
    return result


def get_filtered(scores: npt.NDArray, labels: pd.DataFrame) -> npt.NDArray:
    """A split's scores after the causal median by recording: what the threshold is applied to."""
    return get_causal_median(scores, labels['rec'].to_numpy(), labels['end'].to_numpy())


def tune_threshold(filtered: npt.NDArray, labels: pd.DataFrame, familiar: tuple[str, ...],
                   target_fa_pct: float = TARGET_FA_PCT) -> float:
    """The threshold that flags ``target_fa_pct`` % of a held-out split's familiar windows (filtered scores)."""
    is_fam = np.isin(labels['task'].astype(str).to_numpy(), familiar)
    return float(np.percentile(filtered[is_fam], 100 - target_fa_pct))


# ----------------------------------------------------------------------------- native metrics (section 7)

def get_active_dims(logvar: npt.NDArray) -> npt.NDArray[np.bool_]:
    """VAE dimensions whose median posterior variance on the given windows is below 0.9 (notebook 02's rule); every
    dimension when fewer than two pass. (N, D) -> (D,)."""
    active = np.median(np.exp(logvar), axis = 0) < 0.9
    return active if active.sum() >= 2 else np.ones(logvar.shape[1], bool)


def score_mahalanobis(fit: npt.NDArray, others: list[npt.NDArray]) -> list[npt.NDArray[np.float64]]:
    """Squared Mahalanobis distance to the fit set's mean, on units that vary in the fit set (a small ridge keeps the
    covariance invertible when units are collinear). (N, D) -> (N,)."""
    fit = fit.astype(np.float64)
    keep = fit.std(axis = 0) > 1e-6
    mean = fit[:, keep].mean(axis = 0)
    cov = np.cov(fit[:, keep], rowvar = False) + 1e-6 * np.eye(int(keep.sum()))
    factor = cho_factor(cov)
    out = []
    for other in others:
        diff = other[:, keep].astype(np.float64) - mean                          # (N, K)
        out.append(np.einsum('nk,nk->n', diff, cho_solve(factor, diff.T).T))
    return out


def score_out_of_range(fit: npt.NDArray, others: list[npt.NDArray]) -> list[npt.NDArray[np.float64]]:
    """How far units fire beyond their range on the fit set (0.5th to 99.5th percentile), summed over units, in units of
    the median unit SD: the label-free form of the sparse model's alarm units. (N, D) -> (N,)."""
    lo, hi = np.percentile(fit, [0.5, 99.5], axis = 0)                           # (D,), (D,)
    sd = fit.std(axis = 0)
    scale = float(np.median(sd[sd > 1e-6])) if (sd > 1e-6).any() else 1.0
    return [(np.clip(o - hi, 0, None) + np.clip(lo - o, 0, None)).sum(axis = 1) / scale for o in others]


def get_lof_part(fit: dict, parts: list[dict], dims: npt.NDArray[np.bool_] | None, standardize: bool) -> list[npt.NDArray]:
    """LOF (k = 20) on the latent (or some of its dimensions), optionally standardized by the fit set."""
    pick = (lambda d: d['z'] if dims is None else d['z'][:, dims])                # noqa: E731
    transform = get_representation(pick(fit), None) if standardize else (lambda x: x)  # noqa: E731
    return score_lof(transform(pick(fit)), [transform(pick(p)) for p in parts], SPLIT_SEED)


def get_posterior_variance(fit: dict, parts: list[dict]) -> list[npt.NDArray]:
    """Mean posterior variance over the dimensions active on the fit set: the VAE's own uncertainty about z."""
    active = get_active_dims(fit['logvar'])
    return [np.exp(p['logvar'][:, active]).mean(axis = 1) for p in parts]


# name -> (families it applies to, function(fit part, [parts]) -> [scores]); larger is not assumed to mean unfamiliar:
# the direction of every score is chosen on the validation participants
NATIVE: dict[str, tuple[tuple[str, ...] | None, Callable]] = {
    'LOF, raw latent': (None, lambda f, ps: get_lof_part(f, ps, None, False)),
    'LOF, standardized latent': (None, lambda f, ps: get_lof_part(f, ps, None, True)),
    'Mahalanobis, latent': (None, lambda f, ps: score_mahalanobis(f['z'], [p['z'] for p in ps])),
    'out-of-range units': (None, lambda f, ps: score_out_of_range(f['z'], [p['z'] for p in ps])),
    'reconstruction error': (None, lambda f, ps: [p['recon_mse'] for p in ps]),
    'LOF, active dimensions': (('vae',), lambda f, ps: get_lof_part(f, ps, get_active_dims(f['logvar']), True)),
    'KL': (('vae',), lambda f, ps: [p['kl'] for p in ps]),
    'NLL (decoder)': (('vae',), lambda f, ps: [p['nll'] for p in ps]),
    'negative ELBO': (('vae',), lambda f, ps: [p['nll'] + p['kl'] for p in ps]),
    'posterior variance': (('vae',), get_posterior_variance),
}


def get_native_names(family: str) -> list[str]:
    """The native scores that apply to a model family."""
    return [name for name, (families, _) in NATIVE.items() if families is None or family in families]


def get_native_scores(run: Path, export: dict, val: dict, name: str, cache: Path) -> tuple[npt.NDArray, npt.NDArray]:
    """(validation, test) scores of one native metric, fitted on the fit part of the exported training windows."""
    stamp = hashlib.sha256((run / 'latents.npz').read_bytes() + (run / f'latents_{VAL_SPLIT}.npz').read_bytes()).hexdigest()[:16]
    key = hashlib.sha256(f'{stamp}|native|{name}|{FIT_SHARE}|{SPLIT_SEED}|{LOF_NEIGHBORS}'.encode()).hexdigest()[:16]
    path = cache / f'{key}.npz'
    if path.exists():
        saved = np.load(path)
        result = (saved['val'], saved['test'])
    else:
        fit_idx, _ = get_fit_split(int(export['is_train'].sum()))
        export, val = materialize(export), materialize(val)
        arrays = [k for k in ('z', 'recon_mse', 'logvar', 'nll', 'kl') if k in export]
        train = {k: export[k][export['is_train']][fit_idx] for k in arrays}
        test = {k: export[k][~export['is_train']] for k in arrays}
        result = tuple(NATIVE[name][1](train, [{k: val[k] for k in arrays}, test]))
        if not all(np.isfinite(r).all() for r in result):
            raise FloatingPointError(f'{run.name}: non-finite {name} scores')
        save_cache(path, val = result[0], test = result[1])
    return result


def evaluate_oriented(val: npt.NDArray, test: npt.NDArray, labels_v: pd.DataFrame, labels_t: pd.DataFrame,
                      familiar: tuple[str, ...], target_fa_pct: float = TARGET_FA_PCT) -> tuple[dict, pd.Series]:
    """Orient a score on the validation participants (negate it if it ranks unfamiliar windows lower there), set the
    threshold for ``target_fa_pct`` % false alarms on them, and score the test participants once."""
    fam_v = np.isin(labels_v['task'].astype(str).to_numpy(), familiar)
    sign = 1.0 if roc_auc_score(~fam_v, val) >= 0.5 else -1.0
    threshold = tune_threshold(get_filtered(sign * val, labels_v), labels_v, familiar, target_fa_pct)
    m_val, _ = evaluate_at(threshold, sign * val, labels_v, familiar)
    m_test, per_task = evaluate_at(threshold, sign * test, labels_t, familiar)
    metrics = {'direction': 'higher = unfamiliar' if sign > 0 else 'lower = unfamiliar', 'sign': sign, 'threshold': threshold,
               **{f'val {k}': v for k, v in m_val.items()}, **{f'test {k}': v for k, v in m_test.items()}}
    return metrics, per_task


def get_combined_auroc(val_feats: npt.NDArray, labels_v: pd.DataFrame, test_feats: npt.NDArray, labels_t: pd.DataFrame,
                       familiar: tuple[str, ...], seed: int = SPLIT_SEED) -> dict[str, float]:
    """A downstream classifier's view: logistic regression on several scores (each mapped to a normal by its quantiles
    on the validation participants), trained to tell unfamiliar from familiar windows on the validation participants and
    scored on the test ones. (N, F) features. Supervised with task labels, unlike every single score above."""
    y_v = ~np.isin(labels_v['task'].astype(str).to_numpy(), familiar)
    model = make_pipeline(QuantileTransformer(output_distribution = 'normal', n_quantiles = 1000, random_state = seed,
                                              subsample = 200_000),
                          LogisticRegression(class_weight = 'balanced', max_iter = 2000))
    model.fit(val_feats, y_v)
    p = model.predict_proba(test_feats)[:, 1]
    tasks = labels_t['task'].astype(str).to_numpy()
    fam = np.isin(tasks, familiar)
    dangerous = labels_t['controller_class'].astype(str).to_numpy() == 'OOD-dangerous'
    auroc = lambda m: roc_auc_score(np.r_[np.zeros(fam.sum()), np.ones(m.sum())], np.r_[p[fam], p[m]])   # noqa: E731
    return {'AUROC, unfamiliar vs familiar': auroc(~fam), 'AUROC, OOD-dangerous vs familiar': auroc(dangerous)}


# ----------------------------------------------------------------------------- training-class ablation (notebook 07)

DANGEROUS, SAFE, ID = 'OOD-dangerous', 'OOD-safe', 'ID-safe'
# training class -> where its runs live under the runs folders: the ID-safe runs of notebooks 01 to 05, and the ablation's
# retrained twins (ml/nature/awareness/train_class_tasks.py; same folder names)
TRAIN_CLASS_DIRS = {'ID-safe': 'runs', 'ID-safe+OOD-safe': 'runs_train_class/id_ood_safe', 'OOD-safe': 'runs_train_class/ood_safe'}
TRAIN_CLASS_LABELS = {'ID-safe': 'ID', 'ID-safe+OOD-safe': 'ID + OOD-safe', 'OOD-safe': 'OOD-safe only'}


def get_auroc(positive: npt.NDArray, negative_sorted: npt.NDArray) -> float:
    """P(a positive window scores above a negative one), ties counted half: ``roc_auc_score`` for two groups, by a binary
    search of each positive among the negatives (sorted ascending, so several positive groups can share one sort)."""
    below = np.searchsorted(negative_sorted, positive, 'left')
    tied = np.searchsorted(negative_sorted, positive, 'right') - below
    return float((below.sum() + 0.5 * tied.sum()) / (len(positive) * len(negative_sorted)))


def get_dangerous_metrics(scores: npt.NDArray, labels: pd.DataFrame, flagged: npt.NDArray[np.bool_]) -> dict[str, float]:
    """OOD-dangerous against OOD-safe windows (and against every safe window) from oriented per-window scores (larger =
    more dangerous), and the shares of each class ``flagged`` at the threshold. Per dangerous task: AUROC against OOD-safe."""
    cls = labels['controller_class'].astype(str).to_numpy()
    tasks = labels['task'].astype(str).to_numpy()
    dang, safe, ident = cls == DANGEROUS, cls == SAFE, cls == ID
    safe_sorted, all_safe_sorted = np.sort(scores[safe]), np.sort(scores[safe | ident])
    per_task = {f'AUROC, {t} vs OOD-safe': get_auroc(scores[dang & (tasks == t)], safe_sorted) for t in sorted(set(tasks[dang]))}
    return {'AUROC, dangerous vs OOD-safe': get_auroc(scores[dang], safe_sorted),
            'AUROC, dangerous vs all safe': get_auroc(scores[dang], all_safe_sorted),
            'flagged, dangerous (%)': 100 * flagged[dang].mean(), 'false alarms, OOD-safe (%)': 100 * flagged[safe].mean(),
            'false alarms, ID-safe (%)': 100 * flagged[ident].mean(), **per_task}


def evaluate_dangerous(val: npt.NDArray, test: npt.NDArray, labels_v: pd.DataFrame, labels_t: pd.DataFrame,
                       target_fa_pct: float = TARGET_FA_PCT) -> tuple[dict, pd.Series]:
    """The training-class ablation's protocol: can a score tell OOD-dangerous movement from OOD-safe movement, with ID
    movement recognized upstream by a heuristic? Orient the score on the validation participants (negated if it ranks
    their dangerous windows below their OOD-safe ones), set the threshold to flag ``target_fa_pct`` % of their OOD-safe
    windows after the causal median, and score the test participants once. Returns the metrics and the share of each
    test task's windows flagged."""
    cls_v = labels_v['controller_class'].astype(str).to_numpy()
    pair = np.isin(cls_v, (DANGEROUS, SAFE))
    sign = 1.0 if roc_auc_score(cls_v[pair] == DANGEROUS, val[pair]) >= 0.5 else -1.0
    filtered_v = sign * get_filtered(val, labels_v)                         # the median commutes with the sign
    filtered_t = sign * get_filtered(test, labels_t)
    threshold = float(np.percentile(filtered_v[cls_v == SAFE], 100 - target_fa_pct))
    flagged_t = filtered_t > threshold
    metrics = {'direction': 'higher = dangerous' if sign > 0 else 'lower = dangerous', 'sign': sign, 'threshold': threshold,
               **{f'val {k}': v for k, v in get_dangerous_metrics(sign * val, labels_v, filtered_v > threshold).items()},
               **{f'test {k}': v for k, v in get_dangerous_metrics(sign * test, labels_t, flagged_t).items()}}
    per_task = pd.Series(flagged_t).groupby(labels_t['task'].astype(str).to_numpy()).mean() * 100
    return metrics, per_task


# ----------------------------------------------------------------------------- onset latency (section 8)

STRIDE = 10                          # samples between window ends (ae_data.STRIDE)
MIN_BOUT_S = 2.0                     # bouts shorter than this are not scored


def get_onset_latency(flagged: npt.NDArray[np.bool_], labels: pd.DataFrame, min_bout_s: float = MIN_BOUT_S) -> pd.DataFrame:
    """Per in-task bout (consecutive windows of one recording, one stride apart): its task, length, and the time from its
    first window to its first flagged window (NaN if never flagged). ``flagged`` is aligned with ``labels``."""
    frame = pd.DataFrame({'rec': labels['rec'].to_numpy(), 'end': labels['end'].to_numpy(), 'flag': flagged,
                          'task': labels['task'].astype(str).to_numpy(),
                          'controller_class': labels['controller_class'].astype(str).to_numpy()}).sort_values(['rec', 'end'])
    new = (frame['rec'].diff() != 0) | (frame['end'].diff() != STRIDE)                 # a new bout at a gap or new recording
    frame['bout'] = new.cumsum()
    g = frame.groupby('bout')
    first_end = g['end'].transform('min')
    frame['since'] = (frame['end'] - first_end) / RATE_HZ                                 # s since the bout's first window
    hits = frame[frame['flag']].groupby('bout')['since'].min()
    bouts = g.agg(rec = ('rec', 'first'), task = ('task', 'first'), controller_class = ('controller_class', 'first'),
                  length_s = ('since', 'max'))
    bouts['latency_s'] = hits.reindex(bouts.index)
    return bouts[bouts['length_s'] >= min_bout_s].reset_index(drop = True)


# ----------------------------------------------------------------------------- latent similarity (section 9)

def get_linear_cka(x: npt.NDArray, y: npt.NDArray) -> float:
    """Linear centered kernel alignment between two representations of the same N windows, (N, Dx) x (N, Dy) -> [0, 1].
    1 when one latent is a rotation and rescaling of the other; invariant to orthogonal maps and isotropic scale."""
    x = x.astype(np.float64) - x.mean(axis = 0)
    y = y.astype(np.float64) - y.mean(axis = 0)
    cross = np.linalg.norm(x.T @ y, 'fro') ** 2
    return float(cross / (np.linalg.norm(x.T @ x, 'fro') * np.linalg.norm(y.T @ y, 'fro')))

