"""Novelty scores for every trained latent model, and how well they flag unfamiliar movement (05_LOF_comparison).

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
"""
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'ml' / 'common').is_dir())
if str(REPO / 'ml' / 'common') not in sys.path:
    sys.path.insert(0, str(REPO / 'ml' / 'common'))
from models.autoencoder import LOF_NEIGHBORS, THRESHOLD_PERCENTILE, LatentLOF  # noqa: E402  (the ankle paper's LOF)

FIT_SHARE = 0.75
SPLIT_SEED = 0
FILTER_S = 0.5
RATE_HZ = 200
INPUT_LABELS = {'raw_1s': '1 s raw', 'raw_3s_spectral': '3 s + spectral'}


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
    return result


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


def load_export(run: Path, windows: pd.DataFrame) -> dict:
    """A run's export with its window labels: z (N, D), recon_mse (N,), is_train (N,), labels (N rows)."""
    data = np.load(run / 'latents.npz')
    labels = windows.iloc[data['rows']].reset_index(drop = True)
    return {'z': data['z'].astype(np.float32), 'recon_mse': data['recon_mse'].astype(np.float64),
            'is_train': data['is_train'].astype(bool), 'labels': labels}


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
        result = SCORES[score](export, fit_idx, cal_idx)
        cache.mkdir(parents = True, exist_ok = True)
        np.savez(path, cal = result[0], test = result[1])
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
    auroc = lambda m: roc_auc_score(np.r_[np.zeros(is_fam.sum()), np.ones(m.sum())], np.r_[test[is_fam], test[m]])  # noqa: E731
    metrics = {'AUROC, unfamiliar vs familiar': auroc(~is_fam), 'AUROC, OOD-dangerous vs familiar': auroc(dangerous),
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
        data = np.load(path)
        result = {'z': data['z'].astype(np.float32), 'recon_mse': data['recon_mse'].astype(np.float64),
                  'is_train': np.zeros(len(data['rows']), bool), 'labels': windows.iloc[data['rows']].reset_index(drop = True)}
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
        train, test = export['z'][export['is_train']], export['z'][~export['is_train']]
        transform = (lambda x: x) if representation == 'raw' else get_representation(train[fit_idx], representation)
        scored = score_lof(transform(train[fit_idx]), [transform(train[cal_idx]), transform(val['z']), transform(test)], SPLIT_SEED)
        result = tuple(scored)
        cache.mkdir(parents = True, exist_ok = True)
        np.savez(path, cal = result[0], val = result[1], test = result[2])
    return result


def get_filtered(scores: npt.NDArray, labels: pd.DataFrame) -> npt.NDArray:
    """A split's scores after the causal median by recording: what the threshold is applied to."""
    return get_causal_median(scores, labels['rec'].to_numpy(), labels['end'].to_numpy())


def tune_threshold(filtered: npt.NDArray, labels: pd.DataFrame, familiar: tuple[str, ...],
                   target_fa_pct: float = TARGET_FA_PCT) -> float:
    """The threshold that flags ``target_fa_pct`` % of a held-out split's familiar windows (filtered scores)."""
    is_fam = np.isin(labels['task'].astype(str).to_numpy(), familiar)
    return float(np.percentile(filtered[is_fam], 100 - target_fa_pct))
