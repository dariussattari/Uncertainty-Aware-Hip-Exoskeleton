"""Offline ID/OOD evaluation, following the ankle paper's procedure.

Section IV of Tourk et al. (arXiv:2508.21221) specifies exactly five steps:

    1) Pass a window (~1 second) of scaled data into the pre-trained model and get the
       uncertainty score for this window.
    2) Filter the output with an 88 point (~0.5s) causal median filter.
    3) Label the window as in or out-of-distribution based on whether the filtered
       uncertainty score is above or below the threshold.
    4) Mark the window as correct if the label of the window is the same as the ground
       truth label of the window.
    5) Calculate the overall accuracy, precision, recall, F1 score, J-statistic, ECE,
       and Brier score.

Metrics are led by the **J-statistic** (``recall + specificity - 1``), which the paper adopts
because it is robust to class imbalance. That matters more here than there: their offline test
set was 80.1% OOD, ours is ~27%, so precision and F1 are not directly comparable to their
numbers while J and AUROC are.

Two places the paper forces a judgement call, both declared in :data:`EVAL_DEVIATIONS`:

**Median or mean?** Table IV says "SMA (88 samples, 0.5 second window)"; the evaluation
procedure above says "88 point (~0.5s) causal median filter". The paper contradicts itself.
We follow the evaluation text, since it describes the scoring pipeline specifically, and
expose ``kind="mean"`` to check the other reading.

**88 of what?** Offline, windows are strided by 10 samples, so 88 *consecutive scores* would
span 88 x 10 / 175 = 5.0 s, not the 0.5 s the paper states. The two readings disagree by 10x.
We hold the **duration** at 0.5 s, which at 200 Hz with stride 10 is 10 scores. Set
``filter_scores=88`` to take the literal reading instead.

The filter is applied **within a trial**, in time order, never across trial boundaries —
smoothing across a cut would blend unrelated recordings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluation.common import (FILTER_KIND, SMOOTH_SECONDS, classification_metrics,
                               expected_calibration_error, brier, filter_length,
                               fit_steepness, sigmoid_calibrate, smooth_by_trial)

from dataset import EnsembleGaitPhase
from models.ensemble import THRESHOLD_PERCENTILE, create_ensemble
from training.common import pick_device

__all__ = ["score_windows", "evaluate_model", "load_checkpoint", "EvalResult"]

SMOOTH_SECONDS = 0.5          # paper: "~0.5s"
FILTER_KIND = "median"        # paper's evaluation text (Table IV says SMA -- see module docstring)



@torch.no_grad()
def score_windows(model, loader, device: torch.device, labelled: bool) -> np.ndarray:
    """Psi for every window a loader yields.

    ``labelled=True`` for the ID loaders, which yield ``(x, y, mask)``; ``False`` for the OOD
    loader, which yields ``x`` alone.
    """
    model.eval()
    out = []
    for batch in loader:
        x = batch[0] if labelled else batch
        out.append(model.uncertainty(x.to(device)).cpu().numpy())
    return np.concatenate(out)


@dataclass
class EvalResult:
    metrics: dict
    per_mode: pd.DataFrame
    threshold: float
    steepness: float
    scores: np.ndarray
    labels: np.ndarray

    def __repr__(self) -> str:
        m = self.metrics
        return (f"EvalResult(J={m['j_statistic']:.1f}, AUROC={m['auroc']:.3f}, "
                f"recall={m['recall']:.1f}%, specificity={m['specificity']:.1f}%)")


def evaluate_model(
    model,
    data: EnsembleGaitPhase,
    threshold: float,
    split: str = "test",
    device: torch.device | None = None,
    batch_size: int | None = None,
    smooth_seconds: float = SMOOTH_SECONDS,
    filter_kind: str = FILTER_KIND,
    filter_scores: int | None = None,
    calibrate_on: str = "val",
    verbose: bool = True,
) -> EvalResult:
    """Run the paper's five-step procedure on one split.

    The sigmoid steepness for ECE/Brier is fitted on ``calibrate_on`` (the validation split by
    default) so nothing about the test set informs its own calibration.
    """
    device = device or pick_device()
    model.to(device)
    bs = batch_size or data.batch_size

    # how many consecutive scores make up smooth_seconds, given the window stride
    cfg = data.config
    rate, stride = cfg.get("sample_rate_hz", 200), cfg.get("stride_samples", 10)
    k = filter_scores if filter_scores is not None else max(1, round(smooth_seconds * rate / stride))

    def collect(sp: str):
        id_s = score_windows(model, data.loader(sp, batch_size=bs), device, labelled=True)
        ood_s = score_windows(model, data.ood_loader(sp, batch_size=bs), device, labelled=False)
        id_m, ood_m = data.splits[sp].meta, data.ood[sp].meta
        # smooth within trials, separately per set, before concatenating
        id_f = smooth_by_trial(id_s, id_m, k, filter_kind)
        ood_f = smooth_by_trial(ood_s, ood_m, k, filter_kind)
        score = np.concatenate([id_f, ood_f])
        label = np.concatenate([np.zeros(len(id_f), int), np.ones(len(ood_f), int)])
        subj = np.concatenate([id_m["subject"].to_numpy(), ood_m["subject"].to_numpy()])
        mode = np.concatenate([id_m["mode"].to_numpy(), ood_m["mode"].to_numpy()])
        return score, label, subj, mode

    if verbose:
        print(f"Evaluating on {split!r} | device={device} | "
              f"causal {filter_kind} filter over {k} scores "
              f"({k * stride / rate:.2f}s) | threshold={threshold:.6f}")

    score, label, subj, mode = collect(split)

    # calibration steepness from a different split, to avoid tuning on the test set
    try:
        c_score, c_label, c_subj, _ = collect(calibrate_on) if calibrate_on != split else (
            score, label, subj, mode)
        steepness = fit_steepness(c_score, c_label, c_subj, threshold)
    except KeyError:
        steepness = fit_steepness(score, label, subj, threshold)

    p = sigmoid_calibrate(score, threshold, steepness)
    metrics = classification_metrics(label, score, threshold)
    metrics |= {"ece": expected_calibration_error(p, label), "brier": brier(p, label),
                "steepness": steepness, "n_id": int((label == 0).sum()),
                "n_ood": int((label == 1).sum()),
                "pct_ood": 100 * float((label == 1).mean())}

    rows = []
    for mo in pd.unique(mode):
        sel = mode == mo
        is_ood = bool(label[sel][0])
        rows.append({"mode": mo, "kind": "OOD" if is_ood else "ID", "windows": int(sel.sum()),
                     "flagged_%": 100 * float((score[sel] > threshold).mean()),
                     "median_psi": float(np.median(score[sel]))})
    per_mode = pd.DataFrame(rows).sort_values(["kind", "mode"]).reset_index(drop=True)

    if verbose:
        print(f"\n  {metrics['n_id']:,} ID + {metrics['n_ood']:,} OOD "
              f"({metrics['pct_ood']:.1f}% OOD; the paper's test set was 80.1%)\n")
        for key in ("j_statistic", "auroc", "accuracy", "recall", "specificity",
                    "precision", "f1", "ece", "brier"):
            v = metrics[key]
            unit = "" if key in ("auroc", "ece", "brier") else "%"
            print(f"    {key:14s} {v:8.3f}{unit}")
        print()
        print(per_mode.to_string(index=False))

    return EvalResult(metrics, per_mode, threshold, steepness, score, label)


def load_checkpoint(path: Path | str, data: EnsembleGaitPhase, device=None):
    """Rebuild the model from a ``train.py`` checkpoint. Returns ``(model, threshold)``."""
    device = device or pick_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_ensemble(data.n_channels, data.n_targets)
    model.load_state_dict(ckpt["model"])
    return model.to(device), float(ckpt.get("threshold", float("nan")))


if __name__ == "__main__":  # python ml/vanilla/eval.py [checkpoint]
    import sys

    data = EnsembleGaitPhase()
    ckpt_path = Path(sys.argv[1] if len(sys.argv) > 1 else "ml/vanilla/runs/smoke/final.pt")
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}. Train one first (python main.py train).")

    model, threshold = load_checkpoint(ckpt_path, data)
    if not np.isfinite(threshold):
        from train import fit_threshold
        threshold, _ = fit_threshold(model, data)
    evaluate_model(model, data, threshold, split="test")
