"""Scoring and metrics shared by every model — the paper's evaluation procedure.

Section IV of Tourk et al. specifies five steps: score each window, filter the score
sequence, threshold at the 99.5th percentile of in-distribution training scores, compare
against the known label, report metrics. None of that is model-specific, so it lives here
and every per-model evaluator imports it. What differs between models is only how Psi is
computed, which is what ``evaluation/<model>.py`` provides.

Two ambiguities in the paper are resolved here once, rather than per model:

* **Median or mean?** Table IV says SMA; the evaluation procedure says causal median.
  We follow the evaluation text and expose ``kind="mean"`` to check the other reading.
* **88 of what?** At stride 10, 88 consecutive scores span 5.0 s, not the 0.5 s stated.
  Duration is held at 0.5 s, which is 10 scores at 200 Hz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

__all__ = ["EVAL_DEVIATIONS", "SMOOTH_SECONDS", "FILTER_KIND", "causal_filter",
           "smooth_by_trial", "classification_metrics", "brier",
           "expected_calibration_error", "fit_steepness", "sigmoid_calibrate",
           "filter_length"]

SMOOTH_SECONDS = 0.5
FILTER_KIND = "median"

EVAL_DEVIATIONS = (
    ("filter kind", "Table IV says SMA; evaluation text says causal median. The paper "
                    "contradicts itself. We follow the evaluation text."),
    ("filter length", "Held at 0.5 s of wall time (10 strided scores at 200 Hz) rather than "
                      "a literal 88 scores, which at stride 10 would span 5.0 s."),
    ("class balance", "Test set is ~27% OOD vs the paper's 80.1%, because pseudo-OOD comes "
                      "from held-out ambulation modes. J-statistic and AUROC are comparable "
                      "across that difference; precision and F1 are not."),
)


def causal_filter(x: np.ndarray, k: int, kind: str = FILTER_KIND) -> np.ndarray:
    """Causal rolling median (or mean) over the last ``k`` values, inclusive.

    Causal by construction: position *i* sees only ``x[i-k+1 : i+1]``. Early positions use
    however many values exist, so the output is the same length as the input.
    """
    if k <= 1:
        return np.asarray(x, dtype=np.float64)
    s = pd.Series(np.asarray(x, dtype=np.float64)).rolling(k, min_periods=1)
    return (s.median() if kind == "median" else s.mean()).to_numpy()

def smooth_by_trial(
    scores: np.ndarray,
    meta: pd.DataFrame,
    k: int,
    kind: str = FILTER_KIND,
) -> np.ndarray:
    """Apply :func:`causal_filter` within each ``(subject, trial)``, ordered by time.

    Windows from different trials are unrelated recordings; smoothing across the boundary
    would leak one trial's scores into another's.
    """
    out = np.empty_like(scores, dtype=np.float64)
    order_col = "end_idx" if "end_idx" in meta.columns else None
    for _, g in meta.groupby(["subject", "trial"], observed=True, sort=False):
        rows = g.sort_values(order_col).index.to_numpy() if order_col else g.index.to_numpy()
        out[rows] = causal_filter(scores[rows], k, kind)
    return out

def classification_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """Accuracy / precision / recall / specificity / F1 / J / AUROC.

    ``y_true`` is 1 for out-of-distribution. Positive = "flagged as OOD", so recall is the
    fraction of OOD windows caught and specificity the fraction of ID windows left alone.
    """
    pred = score > threshold
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": 100 * (tp + tn) / max(len(y_true), 1),
        "precision": 100 * precision,
        "recall": 100 * recall,
        "specificity": 100 * specificity,
        "f1": 100 * f1,
        "j_statistic": 100 * (recall + specificity - 1),
        "auroc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }

def sigmoid_calibrate(score: np.ndarray, threshold: float, steepness: float) -> np.ndarray:
    """p = 1 / (1 + exp(-s * (score - threshold))) — the paper's calibration map.

    Centred on the threshold, so a window exactly at the decision boundary gets p = 0.5.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(steepness * (score - threshold), -60, 60)))

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))

def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Mean |accuracy - confidence| across equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        total += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(total)

def fit_steepness(
    score: np.ndarray,
    y_true: np.ndarray,
    subjects: np.ndarray,
    threshold: float,
    grid: np.ndarray | None = None,
) -> float:
    """Choose the sigmoid steepness by leave-one-subject-out, minimising Brier score.

    The paper: *"The steepness parameter was optimized using leave-one-subject-out
    cross-validation to minimize Brier score, ensuring no subject-level data leakage in
    calibration parameter selection."* Each candidate is scored on held-out subjects only.
    """
    if grid is None:
        # Psi is small and unbounded above; span several orders of magnitude around 1/threshold.
        base = 1.0 / max(threshold, 1e-9)
        grid = base * np.logspace(-2, 2, 41)

    uniq = np.unique(subjects)
    if len(uniq) < 2:
        return float(grid[np.argmin([brier(sigmoid_calibrate(score, threshold, s), y_true) for s in grid])])

    losses = []
    for s in grid:
        fold = [brier(sigmoid_calibrate(score[subjects == u], threshold, s), y_true[subjects == u])
                for u in uniq]
        losses.append(np.mean(fold))
    return float(grid[int(np.argmin(losses))])


def filter_length(config: dict, smooth_seconds: float = SMOOTH_SECONDS,
                  override: int | None = None) -> int:
    """How many consecutive scores make up ``smooth_seconds`` at this dataset's stride."""
    if override is not None:
        return override
    rate = config.get("sample_rate_hz", 200)
    stride = config.get("stride_samples", 10)
    return max(1, round(smooth_seconds * rate / stride))
