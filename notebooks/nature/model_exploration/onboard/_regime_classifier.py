"""The regime classifier (06_regime_classifier.ipynb): which controller regime each window should get.

    ID-safe        -> spline (phase-based torque profile)
    OOD-safe       -> Young's joint moment estimator
    OOD-dangerous  -> exo off, the user acts

Features per window come from the exports the earlier notebooks already scored (no retraining): an encoder's latent
(posterior means for a VAE), novelty scores of the ID-trained detector (``_novelty_scores.NATIVE``, cached), and the seven
rhythm features of the window table. Notebook 06 fitted on the validation participants (then the only ones exported in
full) and scored once on the test participants; notebook 13 also encodes the training participants
(``ml/nature/awareness/encode_split.py --split train``) and fits on them. Probabilities are smoothed
with the 0.5 s causal median by recording (as every alarm in notebooks 05 and 07) before the regime is chosen.
"""
import hashlib
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import _novelty_scores as ns

REGIMES = (ns.ID, ns.SAFE, ns.DANGEROUS)                    # class order everywhere: spline, estimator, exo off
RHYTHM = ['dominant_hz_l', 'dominant_hz_r', 'spectral_entropy_l', 'spectral_entropy_r', 'stride_regularity_l',
          'stride_regularity_r', 'step_regularity']


def get_split_rows(table: pd.DataFrame, split: str) -> pd.DataFrame:
    """The in-task windows of one participant split that every run exported (the 3 s table's windows), with labels."""
    keep = (table['split'] == split) & table['in_task']
    return table.loc[keep, ['rec', 'end', 'subject', 'task', 'controller_class', 'family', *RHYTHM]].reset_index(drop = True)


def get_aligned(values: npt.NDArray, labels: pd.DataFrame, keys: pd.DataFrame) -> npt.NDArray:
    """Rows of ``values`` (aligned with ``labels``) for the windows named by ``keys`` (rec, end), in that order."""
    idx = labels[['rec', 'end']].reset_index().merge(keys[['rec', 'end']], on = ['rec', 'end'], how = 'right')['index']
    if idx.isna().any():
        raise KeyError(f'{int(idx.isna().sum())} windows are missing from the export')
    return values[idx.to_numpy(dtype = np.int64)]


def get_latents(run: Path, table: pd.DataFrame, keys: dict[str, pd.DataFrame]) -> dict[str, npt.NDArray[np.float32]]:
    """A run's latent for the validation and test windows in ``keys``: {'val': (N_val, D), 'test': (N_test, D)}."""
    test = np.load(run / 'latents.npz')
    val = np.load(run / 'latents_val.npz')
    lab_t = ns.get_labels(table, test['rows'][~test['is_train']])
    lab_v = ns.get_labels(table, val['rows'])
    return {'val': get_aligned(val['z'], lab_v, keys['val']), 'test': get_aligned(test['z'][~test['is_train']], lab_t, keys['test'])}


def get_novelty(run: Path, table: pd.DataFrame, keys: dict[str, pd.DataFrame], names: list[str], cache: Path
                ) -> dict[str, npt.NDArray[np.float64]]:
    """Native novelty scores of a run (``_novelty_scores.get_native_scores``, cached) for the windows in ``keys``:
    {'val': (N_val, S), 'test': (N_test, S)}, one column per score in ``names``."""
    export, val = ns.load_export(run, table), ns.load_split_export(run, table)
    lab_t = export['labels'][~export['is_train']].reset_index(drop = True)
    cols = {'val': [], 'test': []}
    for name in names:
        v, t = ns.get_native_scores(run, export, val, name, cache)
        cols['val'].append(get_aligned(v, val['labels'], keys['val']))
        cols['test'].append(get_aligned(t, lab_t, keys['test']))
    return {k: np.stack(c, axis = 1) for k, c in cols.items()}


# ----------------------------------------------------------------------------- training participants (notebook 13)

def get_split_latents(run: Path, table: pd.DataFrame, keys: pd.DataFrame, split: str) -> npt.NDArray[np.float32]:
    """A run's latent for the windows in ``keys`` from its ``latents_<split>.npz`` (``encode_split.py --split``). (N, D)."""
    export = np.load(run / f'latents_{split}.npz')
    return get_aligned(export['z'], ns.get_labels(table, export['rows']), keys)


def get_split_novelty(run: Path, table: pd.DataFrame, keys: pd.DataFrame, names: list[str], split: str, cache: Path
                      ) -> npt.NDArray[np.float64]:
    """Native novelty scores for the windows in ``keys`` of another split's export, fitted exactly as ``get_novelty``'s (on
    the fit part of the run's exported training windows), cached. (N, S), one column per score in ``names``."""
    export, other = ns.load_export(run, table), ns.load_split_export(run, table, split)
    if other is None:
        raise FileNotFoundError(f'{run}/latents_{split}.npz is missing; run ml/nature/awareness/encode_split.py --split {split}')
    stamp = hashlib.sha256((run / 'latents.npz').read_bytes() + (run / f'latents_{split}.npz').read_bytes()).hexdigest()[:16]
    fit_idx, _ = ns.get_fit_split(int(export['is_train'].sum()))
    cols = []
    for name in names:
        key = hashlib.sha256(f'{stamp}|native-{split}|{name}|{ns.FIT_SHARE}|{ns.SPLIT_SEED}|{ns.LOF_NEIGHBORS}'.encode()).hexdigest()[:16]
        path = cache / f'{key}.npz'
        if path.exists():
            scores = np.load(path)['scores']
        else:
            full, part = ns.materialize(export), ns.materialize(other)
            arrays = [k for k in ('z', 'recon_mse', 'logvar', 'nll', 'kl') if k in full]
            fit = {k: full[k][full['is_train']][fit_idx] for k in arrays}
            scores = ns.NATIVE[name][1](fit, [{k: part[k] for k in arrays}])[0]
            if not np.isfinite(scores).all():
                raise FloatingPointError(f'{run.name}: non-finite {name} scores on the {split} split')
            ns.save_cache(path, scores = scores)
        cols.append(get_aligned(scores, other['labels'], keys))
    return np.stack(cols, axis = 1)


def get_score_fit_windows(run: Path, table: pd.DataFrame) -> pd.DataFrame:
    """(rec, end) of the training windows a run's novelty scores are fitted on (the fit part of its exported training
    sample). Their scores are in-sample (LOF finds each one among its own neighbours), so a classifier fitted on the
    training participants leaves them out."""
    export = np.load(run / 'latents.npz')
    fit_idx, _ = ns.get_fit_split(int(export['is_train'].sum()))
    return ns.get_labels(table, export['rows'][export['is_train']][fit_idx])[['rec', 'end']]


def get_smoothed(proba: npt.NDArray, labels: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Each class probability after the 0.5 s causal median by recording. (N, K) -> (N, K)."""
    return np.stack([ns.get_causal_median(proba[:, k], labels['rec'].to_numpy(), labels['end'].to_numpy())
                     for k in range(proba.shape[1])], axis = 1)


def get_regime_metrics(true: npt.NDArray, pred: npt.NDArray) -> dict[str, float]:
    """Macro F1, each regime's recall, and the two errors that matter most: dangerous windows sent to the spline, and safe
    (ID or OOD-safe) windows switched off."""
    dang, safe = true == ns.DANGEROUS, np.isin(true, (ns.ID, ns.SAFE))
    out = {'macro F1': f1_score(true, pred, labels = list(REGIMES), average = 'macro')}
    out.update({f'recall, {r}': float((pred[true == r] == r).mean()) for r in REGIMES})
    out['dangerous sent to the spline (%)'] = 100 * float((pred[dang] == ns.ID).mean())
    out['safe switched off (%)'] = 100 * float((pred[safe] == ns.DANGEROUS).mean())
    return out


def get_confusion(true: npt.NDArray, pred: npt.NDArray) -> pd.DataFrame:
    """Share of each true regime's windows given each regime (rows sum to 100)."""
    m = confusion_matrix(true, pred, labels = list(REGIMES)).astype(float)
    return pd.DataFrame(100 * m / m.sum(axis = 1, keepdims = True), index = pd.Index(REGIMES, name = 'true'),
                        columns = pd.Index(REGIMES, name = 'given'))


# ----------------------------------------------------------------------------- target ablation

# What the classifier is trained to predict; every target is mapped back to a regime before it is scored, so all are
# compared on the same decision.
#   regime            the three regimes directly
#   family            the five task families (notebook 01), then the regime most of the family's fitting windows have
#   task              the 26 tasks, then each task's regime (a task left out of fitting cannot be predicted as itself)
#   dangerous or not  OOD-dangerous against OOD-safe, fitted on OOD windows only; ID windows are given the spline, as if
#                     the upstream ID heuristic were perfect
TARGETS = ('regime', 'family', 'task', 'dangerous or not')


def get_target(rows: pd.DataFrame, target: str) -> npt.NDArray:
    """The label each window is fitted to for one target."""
    column = {'regime': 'controller_class', 'family': 'family', 'task': 'task', 'dangerous or not': 'controller_class'}[target]
    return rows[column].astype(str).to_numpy()


def get_regime_map(rows: pd.DataFrame, target: str) -> dict[str, str]:
    """Label -> regime for one target, from the fitting windows: the identity for regimes, each task's regime, and for a
    family the regime most of its windows have."""
    labels, regimes = get_target(rows, target), rows['controller_class'].astype(str).to_numpy()
    frame = pd.DataFrame({'label': labels, 'regime': regimes})
    return frame.groupby('label')['regime'].agg(lambda r: r.value_counts().idxmax()).to_dict()


def get_fit_rows(rows: pd.DataFrame, target: str) -> npt.NDArray[np.bool_]:
    """Which fitting windows a target uses: all of them, or the OOD windows for 'dangerous or not'."""
    return (rows['controller_class'].astype(str) != ns.ID).to_numpy() if target == 'dangerous or not' else np.ones(len(rows), bool)


# ----------------------------------------------------------------------------- aggregated uncertainty (notebook 09)

UNCERTAINTY_POSITION = np.array([0.0, 0.5, 1.0])            # each regime's place on the axis, REGIMES order


def get_uncertainty(proba: npt.NDArray) -> npt.NDArray[np.float64]:
    """One score per window: its expected position on ID-safe (0) -> OOD-safe (0.5) -> OOD-dangerous (1) under the regime
    probabilities. (N, 3) -> (N,)."""
    return np.asarray(proba, dtype = np.float64) @ UNCERTAINTY_POSITION


def get_banded(u: npt.NDArray, cutoffs: tuple[float, float]) -> npt.NDArray:
    """The regime from the score: below the first cutoff the spline, up to the second the estimator, from it exo off."""
    return np.array(REGIMES)[np.searchsorted(np.asarray(cutoffs), u, side = 'right')]


def get_cutoffs(u: npt.NDArray, true: npt.NDArray, grid: npt.NDArray = np.linspace(0, 1, 201)) -> tuple[float, float]:
    """The two cutoffs on ``grid`` with the best macro F1 of ``get_banded``, every pair searched at once from the counts of
    each regime below each grid value."""
    below = np.stack([(u[true == r][:, None] < grid).sum(axis = 0) for r in REGIMES]).astype(np.float64)    # (3, G)
    total = np.array([(true == r).sum() for r in REGIMES], dtype = np.float64)[:, None, None]               # (3, 1, 1)
    low, high = below[:, :, None], below[:, None, :]                                                       # t1 rows, t2 columns
    given = np.stack(np.broadcast_arrays(low, high - low, total - high), axis = 1)                       # (3 true, 3 given, G, G)
    tp = np.einsum('kkij->kij', given)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):                                              # t1 > t2 cells, masked below
        f1 = 2 * tp / (given.sum(axis = 0) + total)                                                        # (3, G, G)
    macro = np.where(np.triu(np.ones((len(grid), len(grid)), bool)), f1.mean(axis = 0), -np.inf)          # t1 <= t2 only
    i, j = np.unravel_index(np.argmax(macro), macro.shape)
    return float(grid[i]), float(grid[j])


def get_assistance(u: npt.NDArray, cutoffs: tuple[float, float], floor: float) -> npt.NDArray[np.float64]:
    """Share of the controller's torque given: 1 on the spline; across the estimator band it falls linearly from 1 to
    ``floor`` as the score rises; 0 from the second cutoff (exo off)."""
    # ponytail: linear ramp in the score; fit the ramp to the estimator's error against the score once that is measured
    t1, t2 = cutoffs
    ramp = 1 - (1 - floor) * (u - t1) / max(t2 - t1, 1e-12)
    return np.where(u < t1, 1.0, np.where(u < t2, ramp, 0.0))


def get_ramped_assistance(u: npt.NDArray, score: npt.NDArray, cutoffs: tuple[float, float], anchors: tuple[float, float],
                          floor: float) -> npt.NDArray[np.float64]:
    """Share of the controller's torque when the estimator band is ramped on a separate score: 1 on the spline (``u`` below
    the first cutoff); in the band 1 up to the score's first anchor, falling linearly to ``floor`` at its second and held
    there; 0 from the second cutoff of ``u`` (exo off). ``score`` is oriented so higher means less familiar."""
    # ponytail: linear ramp between two quantile anchors; fit it to the estimator's error against the score once measured
    lo, hi = anchors
    ramp = 1 - (1 - floor) * np.clip((score - lo) / max(hi - lo, 1e-12), 0, 1)
    return np.where(u < cutoffs[0], 1.0, np.where(u < cutoffs[1], ramp, 0.0))


# ----------------------------------------------------------------------------- what the classifier relies on (notebook 12)

def get_predicted_regimes(model, x: npt.NDArray, labels: pd.DataFrame) -> npt.NDArray:
    """The regime a fitted classifier gives each window: probabilities in REGIMES order, the 0.5 s causal median by
    recording, the most probable regime (notebook 06's decision). (N, F) -> (N,)."""
    proba = pd.DataFrame(model.predict_proba(x), columns = model.classes_).reindex(columns = list(REGIMES), fill_value = 0).to_numpy()
    return np.array(REGIMES)[get_smoothed(proba, labels).argmax(axis = 1)]


def get_permutation_importance(model, x: npt.NDArray, true: npt.NDArray, labels: pd.DataFrame, groups: dict[str, list[int]],
                               repeats: int, rng: np.random.Generator) -> pd.DataFrame:
    """How much each group of feature columns matters to a fitted classifier: the drop in each regime metric when the
    group's columns are shuffled across windows, all with one shared permutation so the group's own correlations are kept.

    One row per (group, repeat) with the drop (baseline minus shuffled) in macro F1 and each regime's recall, and the rise
    in dangerous windows sent to the spline and safe windows switched off. Features in other groups that carry the same
    information can stand in for a shuffled group, so the drop measures what only that group provides.
    """
    base = get_regime_metrics(true, get_predicted_regimes(model, x, labels))
    keys = ['macro F1', *[f'recall, {r}' for r in REGIMES]]
    rises = ['dangerous sent to the spline (%)', 'safe switched off (%)']
    rows = []
    for name, cols in groups.items():
        for repeat in range(repeats):
            shuffled = x.copy()
            shuffled[:, cols] = x[rng.permutation(len(x))][:, cols]
            m = get_regime_metrics(true, get_predicted_regimes(model, shuffled, labels))
            rows.append({'group': name, 'repeat': repeat, **{f'drop, {k}': base[k] - m[k] for k in keys},
                         **{f'rise, {k}': m[k] - base[k] for k in rises}})
    return pd.DataFrame(rows)
