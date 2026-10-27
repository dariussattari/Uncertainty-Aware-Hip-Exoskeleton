"""The awareness layer on the device: novelty scores, the regime classifier and the uncertainty score u.

Extracted from ``notebooks/nature/model_exploration/onboard/13_classifier_training_participants.ipynb`` (E39) and its
helpers ``_novelty_scores.py`` and ``_regime_classifier.py``; ``export_awareness.py`` fits it and checks that it
reproduces the notebook's test scores. Per window (3 s at 200 Hz, one every 10 samples):

    LSTM-VAE (ID, 3 s raw, beta 1)     z, logvar (32,)  -> LOF, Mahalanobis, posterior variance   (3,)
    TCN-VAE (ID + OOD-safe, beta 128)  z (32,)                                                      (32,)
    rhythm features (features.py)                                                                   (7,)
    -> 42 features -> boosted trees -> class probabilities -> 0.5 s causal median -> u -> regime

    regime     ID-safe -> spline,   OOD-safe -> joint-moment estimator,   OOD-dangerous -> exo off
    u          0 * p_ID + 0.5 * p_OOD-safe + 1 * p_dangerous; below cutoff 1 the spline, from cutoff 2 exo off

Version-free: the fitted LOF and boosted trees are stored as plain arrays (``.npz``) and evaluated here with numpy and
scipy only, exactly as scikit-learn computes them (``LocalOutlierFactor.score_samples``,
``HistGradientBoostingClassifier.predict_proba``), so the device needs no scikit-learn and no matching versions.
"""
from collections import deque
from dataclasses import asdict, dataclass, field, fields
import json
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_solve
from scipy.spatial import cKDTree

ID, SAFE, DANGEROUS = 'ID-safe', 'OOD-safe', 'OOD-dangerous'
REGIMES = (ID, SAFE, DANGEROUS)                         # class order everywhere: spline, estimator, exo off
UNCERTAINTY_POSITION = np.array([0.0, 0.5, 1.0])         # each regime's place on the u axis, REGIMES order
NOVELTY = ('LOF, raw latent', 'Mahalanobis, latent', 'posterior variance')
RHYTHM = ('dominant_hz_l', 'dominant_hz_r', 'spectral_entropy_l', 'spectral_entropy_r', 'stride_regularity_l',
          'stride_regularity_r', 'step_regularity')
LATENT_DIM = 32
N_FEATURES = LATENT_DIM + len(NOVELTY) + len(RHYTHM)   # 42
LOF_NEIGHBORS = 20
RATE_HZ = 200
FILTER_S = 0.5                                           # causal median over the windows ending in the last 0.5 s


@dataclass
class NoveltyFit:
    """The LSTM-VAE novelty scores' fits, on the fit part (75%) of the run's exported ID-safe training windows."""
    lof_fit: npt.NDArray[np.float64]                     # (n, 32) LOF reference latents (deduplicated)
    lof_kdist: npt.NDArray[np.float64]                   # (n,) each reference point's distance to its k-th neighbour
    lof_lrd: npt.NDArray[np.float64]                     # (n,) each reference point's local reachability density
    lof_cap: float                                       # largest finite LOF on the reference set; caps a non-finite score
    maha_keep: npt.NDArray[np.bool_]                     # (32,) latent units that vary on the fit set
    maha_mean: npt.NDArray[np.float64]                   # (K,)
    maha_chol: npt.NDArray[np.float64]                   # (K, K) upper Cholesky factor of the covariance + 1e-6 ridge
    active: npt.NDArray[np.bool_]                        # (32,) units with median posterior variance < 0.9
    tree: cKDTree = field(init = False, repr = False)     # neighbour index over lof_fit, built on load

    def __post_init__(self) -> None:
        self.tree = cKDTree(self.lof_fit)


@dataclass(frozen = True)
class Trees:
    """A fitted multiclass HistGradientBoostingClassifier as flat node arrays (all trees concatenated)."""
    feature: npt.NDArray[np.int64]                       # (nodes,) split feature
    threshold: npt.NDArray[np.float64]                   # (nodes,) go left when x <= threshold
    missing_left: npt.NDArray[np.bool_]                  # (nodes,) where a NaN goes
    left: npt.NDArray[np.int64]                          # (nodes,) global child indices
    right: npt.NDArray[np.int64]
    is_leaf: npt.NDArray[np.bool_]
    value: npt.NDArray[np.float64]                       # (nodes,) leaf values (raw score)
    roots: npt.NDArray[np.int64]                         # (iterations, classes) root node of each tree
    baseline: npt.NDArray[np.float64]                    # (classes,) raw score before any tree
    classes: tuple[str, ...]                             # the classifier's class order


@dataclass(frozen = True)
class AwarenessBundle:
    """Everything the awareness layer needs for one seed (encoder seed = classifier seed)."""
    novelty: NoveltyFit
    trees: Trees
    cutoffs: tuple[float, float]                         # on u, chosen on the validation participants
    meta: dict                                           # seed, runs, git SHA, data version, library versions


def get_active_dims(logvar: npt.NDArray) -> npt.NDArray[np.bool_]:
    """Units whose median posterior variance is below 0.9; every unit when fewer than two pass. (N, D) -> (D,)."""
    active = np.median(np.exp(logvar), axis = 0) < 0.9
    return active if active.sum() >= 2 else np.ones(logvar.shape[1], bool)


def fit_novelty(z: npt.NDArray, logvar: npt.NDArray, lof_fit: npt.NDArray, lof_kdist: npt.NDArray,
                lof_lrd: npt.NDArray, lof_cap: float) -> NoveltyFit:
    """The Mahalanobis and posterior-variance fits on the fit windows' latent (N, 32), with a fitted LOF's arrays."""
    if z.shape != logvar.shape or z.shape[1] != LATENT_DIM:
        raise ValueError(f'expected z and logvar of shape (N, {LATENT_DIM}), got {z.shape} and {logvar.shape}')
    fit = z.astype(np.float64)
    keep = fit.std(axis = 0) > 1e-6
    cov = np.cov(fit[:, keep], rowvar = False) + 1e-6 * np.eye(int(keep.sum()))
    return NoveltyFit(lof_fit = lof_fit, lof_kdist = lof_kdist, lof_lrd = lof_lrd, lof_cap = lof_cap, maha_keep = keep,
                      maha_mean = fit[:, keep].mean(axis = 0), maha_chol = np.linalg.cholesky(cov).T,
                      active = get_active_dims(logvar))


def score_lof(fit: NoveltyFit, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Local outlier factor of new latents against the reference set, as sklearn's novelty LOF. (N, 32) -> (N,)."""
    distances, neighbours = fit.tree.query(z, k = LOF_NEIGHBORS)              # (N, k), (N, k)
    reach = np.maximum(distances, fit.lof_kdist[neighbours])                   # (N, k) reachability distances
    lrd = 1.0 / (reach.mean(axis = 1) + 1e-10)                                 # (N,)
    return np.mean(fit.lof_lrd[neighbours] / lrd[:, None], axis = 1)


def score_novelty(fit: NoveltyFit, z: npt.NDArray, logvar: npt.NDArray) -> npt.NDArray[np.float64]:
    """LOF, squared Mahalanobis distance and mean posterior variance of LSTM-VAE latents. (N, 32) x 2 -> (N, 3)."""
    if z.shape != logvar.shape or z.ndim != 2 or z.shape[1] != LATENT_DIM:
        raise ValueError(f'expected z and logvar of shape (N, {LATENT_DIM}), got {z.shape} and {logvar.shape}')
    z = z.astype(np.float64)
    lof = score_lof(fit, z)                                                    # (N,) larger = more unfamiliar
    lof = np.where(np.isfinite(lof), lof, fit.lof_cap)
    diff = z[:, fit.maha_keep] - fit.maha_mean                                 # (N, K)
    maha = np.einsum('nk,nk->n', diff, cho_solve((fit.maha_chol, False), diff.T).T)   # (N,)
    variance = np.exp(logvar[:, fit.active]).mean(axis = 1)                    # (N,) in the encoder's float32, as fitted
    return np.stack([lof, maha, variance], axis = 1)


def get_features(tcn_z: npt.NDArray, novelty: npt.NDArray, rhythm: npt.NDArray) -> npt.NDArray[np.float64]:
    """The classifier's input, in its training order. (N, 32), (N, 3), (N, 7) -> (N, 42)."""
    if tcn_z.shape[1:] != (LATENT_DIM,) or novelty.shape[1:] != (len(NOVELTY),) or rhythm.shape[1:] != (len(RHYTHM),):
        raise ValueError(f'expected (N, 32), (N, 3), (N, 7); got {tcn_z.shape}, {novelty.shape}, {rhythm.shape}')
    x = np.concatenate([tcn_z, novelty, rhythm], axis = 1).astype(np.float64)
    if not np.isfinite(x).all():
        raise FloatingPointError('non-finite classifier features (rhythm features are NaN without a usable rhythm?)')
    return x


def get_proba(trees: Trees, x: npt.NDArray) -> npt.NDArray[np.float64]:
    """Class probabilities in REGIMES order, before smoothing: every tree walked at once, raw scores summed per
    iteration in sklearn's order, then the softmax. (N, 42) -> (N, 3)."""
    if x.ndim != 2 or x.shape[1] != N_FEATURES:
        raise ValueError(f'expected (N, {N_FEATURES}) features, got {x.shape}')
    n, (n_iter, n_class) = len(x), trees.roots.shape
    node = np.broadcast_to(trees.roots.ravel(), (n, n_iter * n_class)).copy()   # (N, T) current node per tree
    rows = np.arange(n)[:, None]
    for _ in range(64):                                                        # deeper than any tree (max 21 here)
        leaf = trees.is_leaf[node]
        if leaf.all():
            break
        value = x[rows, trees.feature[node]]                                   # (N, T)
        go_left = np.where(np.isnan(value), trees.missing_left[node], value <= trees.threshold[node])
        node = np.where(leaf, node, np.where(go_left, trees.left[node], trees.right[node]))
    leaves = trees.value[node].reshape(n, n_iter, n_class)                     # (N, iterations, classes)
    raw = np.broadcast_to(trees.baseline, (n, n_class)).copy()
    for i in range(n_iter):
        raw += leaves[:, i]
    exp = np.exp(raw - raw.max(axis = 1, keepdims = True))
    proba = exp / exp.sum(axis = 1, keepdims = True)                           # (N, classes), classifier order
    return proba[:, [trees.classes.index(r) for r in REGIMES]]


class CausalMedian:
    """Per-class median of the probabilities of the windows that ended in the last 0.5 s (this one included), as
    ``_novelty_scores.get_causal_median`` does over a whole recording. Owns the stream's history; one per recording,
    reset when the stream restarts."""

    def __init__(self, seconds: float = FILTER_S, rate_hz: int = RATE_HZ):
        self.span = int(round(seconds * rate_hz))                               # samples
        self.history: deque[tuple[int, npt.NDArray]] = deque()

    def push(self, end: int, proba: npt.NDArray) -> npt.NDArray[np.float64]:
        """Add one window (its last sample index, its (3,) probabilities); return the smoothed (3,) probabilities."""
        if self.history and end <= self.history[-1][0]:
            raise ValueError(f'window ends must increase: {end} after {self.history[-1][0]}')
        self.history.append((end, np.asarray(proba, dtype = np.float64)))
        for _ in range(len(self.history)):                                      # drop windows older than the span
            if self.history[0][0] <= end - self.span:
                self.history.popleft()
        return np.median(np.stack([p for _, p in self.history]), axis = 0)


def get_uncertainty(proba: npt.NDArray) -> npt.NDArray[np.float64]:
    """Expected position on ID-safe (0) -> OOD-safe (0.5) -> OOD-dangerous (1). (N, 3) or (3,) -> (N,) or ()."""
    return np.asarray(proba, dtype = np.float64) @ UNCERTAINTY_POSITION


def get_regime(u: npt.NDArray, cutoffs: tuple[float, float]) -> npt.NDArray:
    """Below the first cutoff the spline, up to the second the estimator, from it exo off."""
    return np.array(REGIMES)[np.searchsorted(np.asarray(cutoffs), u, side = 'right')]


def save_bundle(bundle: AwarenessBundle, path: Path) -> None:
    """Plain arrays in one .npz (no pickles), written through a temporary file and an atomic rename."""
    novelty = {f'novelty.{f.name}': np.asarray(getattr(bundle.novelty, f.name)) for f in fields(NoveltyFit) if f.init}
    trees = {f'trees.{k}': np.asarray(v) for k, v in asdict(bundle.trees).items()}
    header = json.dumps({'cutoffs': list(bundle.cutoffs), 'meta': bundle.meta})
    tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
    np.savez(tmp, header = np.array(header), **novelty, **trees)
    os.replace(tmp, path)


def load_bundle(path: Path) -> AwarenessBundle:
    """A bundle written by ``save_bundle``."""
    with np.load(path, allow_pickle = False) as data:
        arrays = {k: data[k] for k in data.files}
    header = json.loads(str(arrays.pop('header')))
    novelty = {k.split('.', 1)[1]: v for k, v in arrays.items() if k.startswith('novelty.')}
    novelty['lof_cap'] = float(novelty['lof_cap'])
    trees = {k.split('.', 1)[1]: v for k, v in arrays.items() if k.startswith('trees.')}
    trees['classes'] = tuple(str(c) for c in trees['classes'])
    return AwarenessBundle(novelty = NoveltyFit(**novelty), trees = Trees(**trees), cutoffs = tuple(header['cutoffs']),
                           meta = header['meta'])
