"""The ankle paper's gait-phase ensemble as the baseline (10_gait_phase_ensemble_baseline): what its uncertainty score
Psi does on the Nature data, on the paper's own task (unfamiliar against familiar) and on this project's (ID-safe,
OOD-safe, OOD-dangerous).

Runs come from ``ml/nature/awareness/train_ensemble.py`` (``scores.npz``, ``scores_val.npz``, ``summary.json``). The
protocol follows notebooks 05 and 09 so the numbers sit beside theirs: Psi is filtered with the 0.5 s causal median by
recording before any threshold; the paper's threshold is the 99.5th percentile of the training windows' Psi
(``summary.json``), the project's the filtered Psi that flags ``ns.TARGET_FA_PCT`` % of the validation participants'
familiar windows; the two regime cutoffs are chosen on the validation participants (``rc.get_cutoffs``, on Psi mapped to
its validation quantile so the grid covers it) and fixed for the test participants, which are scored once.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score

import _novelty_scores as ns
import _regime_classifier as rc

REGIME_C = dict(zip(rc.REGIMES, ['#287da8', '#dd791a', '#a32d2d']))     # notebook 01's controller-class colours
MODEL_TYPE_PREFIXES = (('TCN-VAE', 'TCN-VAE'), ('LSTM-VAE', 'LSTM-VAE'), ('Transformer VAE', 'Transformer VAE'), ('VAE', 'conv VAE'),
                       ('vanilla', 'plain AE'), ('sparse', 'sparse AE'), ('gaussian', 'Gaussian-NLL AE'))


# ----------------------------------------------------------------------------- runs

def discover_runs(root: Path, include_smoke: bool = False) -> pd.DataFrame:
    """Every finished ensemble run under ``root``: folder, protocol, seed and the paper's threshold."""
    rows = []
    for summary in sorted(root.glob('*/summary.json')):
        run = summary.parent
        if include_smoke or not run.name.endswith('_smoke'):
            s = json.loads(summary.read_text())
            rows.append({'run': run, 'protocol': s['protocol'], 'seed': s['seed'], 'threshold': s['threshold'], 'val_mse': s['val_mse'],
                         'epochs_run': s['epochs_run'], 'best_epoch': s['best_epoch']})
    return pd.DataFrame(rows)


def load_scores(run: Path, windows: pd.DataFrame, file: str = 'scores.npz') -> dict:
    """One export with its window labels: psi (N,), members (N, M, 2), target (N, 2), is_train (N,), labels (N rows)."""
    data = np.load(run / file)
    return {'psi': data['psi'].astype(np.float64), 'members': data['members'], 'target': data['target'],
            'is_train': data['is_train'].astype(bool), 'labels': ns.get_labels(windows, data['rows'])}


def get_test_part(export: dict) -> dict:
    """The export without its training sample."""
    keep = ~export['is_train']
    return {k: (v[keep].reset_index(drop = True) if isinstance(v, pd.DataFrame) else v[keep]) for k, v in export.items()}


# ----------------------------------------------------------------------------- phase estimation

def get_phase_rmse(members: npt.NDArray, target: npt.NDArray) -> float:
    """RMSE of the ensemble mean against sin(2 pi phase) over the labelled legs. (N, M, 2), (N, 2) -> ()."""
    err = (members.mean(axis = 1) - target) ** 2                                   # (N, 2), NaN on unlabelled legs
    return float(np.sqrt(np.nanmean(err))) if np.isfinite(err).any() else float('nan')


def get_phase_rmse_per_task(export: dict) -> pd.Series:
    """Phase RMSE per task over an export's labelled legs (tasks without a labelled leg are left out)."""
    tasks = export['labels']['task'].astype(str).to_numpy()
    out = {}
    for task in np.unique(tasks):
        m = tasks == task
        if np.isfinite(export['target'][m]).any():
            out[task] = get_phase_rmse(export['members'][m], export['target'][m])
    return pd.Series(out, name = 'phase RMSE')


# ----------------------------------------------------------------------------- scoring

def get_quantile_map(fit: npt.NDArray):
    """Scores -> their quantile among ``fit`` (monotone, in [0, 1]), so cutoffs on a unit grid cover any score's range."""
    ordered = np.sort(fit)
    return lambda x: np.searchsorted(ordered, x, side = 'right') / len(ordered)


def get_three_way(u: npt.NDArray, labels: pd.DataFrame) -> dict[str, float]:
    """AUROC of a score for ID-safe against OOD-safe and OOD-safe against OOD-dangerous (higher = further from ID)."""
    cls = labels['controller_class'].astype(str).to_numpy()
    is_id, is_safe, is_dang = (cls == r for r in rc.REGIMES)
    return {'AUROC, ID-safe vs OOD-safe': roc_auc_score(is_safe[is_id | is_safe], u[is_id | is_safe]),
            'AUROC, OOD-safe vs dangerous': roc_auc_score(is_dang[is_safe | is_dang], u[is_safe | is_dang])}


def evaluate_run(val: dict, test: dict, threshold: float, familiar: tuple[str, ...]) -> dict[str, float]:
    """Every number of one run on the test participants.

    The paper's rule (filtered Psi above the 99.5th percentile of training Psi), the project's operating point
    (threshold at ``ns.TARGET_FA_PCT`` % false alarms on the validation participants), and the three-way split: AUROCs
    and the two cutoffs chosen on validation (macro F1 of ``rc.get_banded``), applied to the test participants.
    """
    f_val, f_test = ns.get_filtered(val['psi'], val['labels']), ns.get_filtered(test['psi'], test['labels'])
    out = {}
    paper, _ = ns.evaluate_at(threshold, test['psi'], test['labels'], familiar)
    out.update({f'paper rule: {k}': v for k, v in paper.items()})
    tuned = ns.tune_threshold(f_val, val['labels'], familiar)
    project, _ = ns.evaluate_at(tuned, test['psi'], test['labels'], familiar)
    out.update({f'{ns.TARGET_FA_PCT:g}% FA rule: {k}': v for k, v in project.items() if 'flagged' in k or 'false alarms' in k})
    out['threshold, paper'], out['threshold, tuned'] = threshold, tuned
    out.update(get_three_way(test['psi'], test['labels']))
    to_q = get_quantile_map(f_val)
    cutoffs = rc.get_cutoffs(to_q(f_val), val['labels']['controller_class'].astype(str).to_numpy())
    pred = rc.get_banded(to_q(f_test), cutoffs)
    out.update(rc.get_regime_metrics(test['labels']['controller_class'].astype(str).to_numpy(), pred))
    out['cutoff, spline | estimator (val quantile)'], out['cutoff, estimator | off (val quantile)'] = cutoffs
    return out


def get_reference_detectors(scores_csv: Path) -> pd.DataFrame:
    """Notebook 05's native scores, each model type's best (model, score) by validation AUROC for unfamiliar against
    familiar (as notebook 09 picks them), with the test AUROCs' mean and SD over seeds."""
    native = pd.read_csv(scores_csv)
    native['type'] = [next(t for prefix, t in MODEL_TYPE_PREFIXES if m.startswith(prefix)) for m in native['model']]
    key = ['type', 'model', 'score']
    mean = native.groupby(key).mean(numeric_only = True)
    best = mean.loc[mean.groupby('type')['val AUROC, unfamiliar vs familiar'].idxmax()]
    cols = ['test AUROC, unfamiliar vs familiar', 'test AUROC, OOD-dangerous vs familiar']
    sd = native.groupby(key)[cols].std().loc[best.index]
    out = best[cols].copy()
    for c in cols:
        out[f'{c} sd'] = sd[c]
    return out.reset_index().sort_values(cols[0], ascending = False)


# ----------------------------------------------------------------------------- per task

def get_per_task(u: npt.NDArray, labels: pd.DataFrame) -> pd.DataFrame:
    """Median and interquartile range of a score per task, with the task's regime, sorted by median."""
    frame = pd.DataFrame({'task': labels['task'].astype(str).to_numpy(), 'regime': labels['controller_class'].astype(str).to_numpy(), 'u': u})
    per = frame.groupby(['task', 'regime'])['u'].describe()[['25%', '50%', '75%']].reset_index()
    return per.sort_values('50%').reset_index(drop = True)


def plot_per_task(ax: plt.Axes, per: pd.DataFrame, cutoffs: tuple[float, ...], ylabel: str, log: bool = False, title: str = '') -> None:
    """Notebook 09's per-task panel: one dot (median) and bar (IQR) per task, coloured by the task's regime."""
    for k, t in enumerate(per.itertuples()):
        ax.plot([k, k], [t._3, t._5], color = REGIME_C[t.regime], linewidth = 2)
        ax.plot(k, t._4, 'o', color = REGIME_C[t.regime], markersize = 7)
    for c in cutoffs:
        ax.axhline(c, color = '#888', linestyle = '--', linewidth = 1)
    ax.set_xticks(range(len(per)), per['task'], rotation = 60, ha = 'right', fontsize = 8)
    for k, t in enumerate(per.itertuples()):
        ax.get_xticklabels()[k].set_color(REGIME_C[t.regime])
    if log:
        ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize = 10, loc = 'left')
    ax.spines[['top', 'right']].set_visible(False)
