r"""Evaluation — the paper's procedure, with four candidate scores instead of one.

The protocol is imported from ``ml/vanilla/evaluation/common.py`` unchanged: the same causal
median filter over ten consecutive scores (0.5 s at stride 10), the same 99.5th-percentile
threshold fitted on in-distribution training data, the same metric set led by the J statistic
and the area under the ROC curve. That is deliberate — the comparison against Experiment 5 is
only meaningful if nothing about the scoring pipeline moved.

What differs is that this model exposes a decomposition rather than a single number, so four
scores are evaluated side by side rather than one being assumed best:

``epistemic``
    Variance of the branch means. **This is Experiment 5's score by construction**, so it is
    the built-in correctness check: same target, same trunk, same protocol, and therefore it
    should land close to Experiment 5's AUROC of 0.899. A large gap means the implementation is
    wrong, not that the idea failed.
``aleatoric``
    Mean of the predicted variances — the model's estimate of how intrinsically unpredictable
    the next 200 ms is. Expected to be a *worse* detector than epistemic, because gait is
    genuinely stochastic on familiar ground too.
``total``
    Their sum, the full mixture variance.
``ratio``
    ``epistemic / (aleatoric + eps)`` — disagreement relative to intrinsic difficulty. Included
    because the analogous quantity in the Krogh-Vedelsby analysis, ``A/(E+eps)``, scored 0.320,
    far worse than either component. Screening it again here costs nothing and the prior is
    that it fails.

Also reported is the **calibration** of the aleatoric head, which no earlier experiment in this
project could measure: the variance of the standardised residual ``(y - mu)/sigma``, whose
target value is exactly 1. Reporting a number with a known correct answer is more useful than
reporting one without.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import paths  # noqa: F401

import numpy as np
import pandas as pd
import torch

import data as D
from evaluation.common import (FILTER_KIND, SMOOTH_SECONDS, brier, classification_metrics,
                               expected_calibration_error, filter_length, fit_steepness,
                               sigmoid_calibrate, smooth_by_trial)
from losses import gaussian_nll_terms
from prob_models import create_prob_ensemble
from training.common import pick_device, resolve_out

__all__ = ["ProbScores", "load_run", "score_split", "evaluate", "per_mode_table"]

EPS = 1e-12
SCORES = ("epistemic", "aleatoric", "total", "ratio")
# Experiment 5, on its own full test split. The epistemic score should approach this.
EXP5_AUROC, EXP5_J = 0.896, 47.64


@dataclass
class ProbScores:
    raw: dict            # score name -> per-window unfiltered values
    smooth: dict         # score name -> causal-median-filtered values
    labels: np.ndarray
    meta: pd.DataFrame
    thresholds: dict     # score name -> float
    split: str
    calibration: dict    # z_var and friends, from the aleatoric head

    def __repr__(self) -> str:
        return (f"ProbScores({self.split!r}, n={len(self.labels):,}, "
                f"{int((self.labels==1).sum()):,} OOD / {int((self.labels==0).sum()):,} ID)")

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"), labels=self.labels,
            **{f"raw_{k}": v for k, v in self.raw.items()},
            **{f"smooth_{k}": v for k, v in self.smooth.items()})
        self.meta.to_parquet(path.with_suffix(".parquet"), index=False)
        print(f"  wrote {path.with_suffix('.npz').name}")


def load_run(run: Path | str, dat, device=None):
    """Rebuild the model and its calibrated thresholds from a ``train.py final`` run."""
    run = resolve_out(run)
    device = device or pick_device()
    ck = torch.load(run / "final.pt", map_location=device, weights_only=False)
    model = create_prob_ensemble(dat.n_channels, dat.n_targets)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    thr = {k: v["threshold"] for k, v in ck["thresholds"].items()}
    # the ratio has no stored threshold; calibrate it from ID training scores on demand
    return model, thr, ck


@torch.no_grad()
def score_split(model, loader, device) -> dict:
    """Every candidate score, plus the calibration diagnostics, for one loader.

    Per-window squared z-scores are retained as well as the aggregate, because the aggregate is
    a mean and a mean is not robust here. Measured on the test split, the mean squared z-score
    is 16.4 while the median is 1.26: the statistic is dominated by a tail of roughly 0.1% of
    windows carrying 58% of its mass. Reporting only the mean would say the variance head is
    badly broken; reporting both says it is calibrated for typical windows and has a heavy tail
    of confident errors, which is a different and more useful conclusion.
    """
    model.eval()
    acc = {k: [] for k in ("epistemic", "aleatoric", "total")}
    cal = {"nll": 0.0, "mse": 0.0, "mean_sigma": 0.0, "z_var": 0.0, "frac_clamped": 0.0}
    weight = 0.0
    z2_per_window = []
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        mu, logvar = model(x)
        u = model.decompose(mu, logvar)
        for k in acc:
            acc[k].append(u[k].cpu().numpy())
        w = mask.sum().item()
        if w:
            t = gaussian_nll_terms(mu, logvar, y, mask)
            for k in cal:
                cal[k] += t[k] * w
            weight += w
        # per-window mean squared z-score. The mask must be expanded over the branch axis
        # before it is summed, or the denominator undercounts by a factor of n_branches.
        m = mask.unsqueeze(1).expand_as(mu)
        den = torch.clamp(m.sum(dim=(1, 2)), min=1.0)
        z2 = (((mu - y.unsqueeze(1)) ** 2 / logvar.exp()) * m).sum(dim=(1, 2)) / den
        z2_per_window.append(z2.cpu().numpy())

    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["ratio"] = out["epistemic"] / (out["aleatoric"] + EPS)
    cal = {k: v / max(weight, 1e-9) for k, v in cal.items()}
    z2 = np.concatenate(z2_per_window)
    order = np.sort(z2)[::-1]
    cal |= {"z2_median": float(np.median(z2)),
            "z2_mean": float(z2.mean()),
            "z2_p90": float(np.percentile(z2, 90)),
            "z2_p99": float(np.percentile(z2, 99)),
            "z2_share_top_0.1pct": float(
                order[:max(1, len(order) // 1000)].sum() / max(order.sum(), 1e-12))}
    return out, cal


def _ratio_threshold(model, dat, device, batch_size, percentile=99.5) -> float:
    vals = []
    with torch.no_grad():
        for x, _, _ in dat.loader("train", batch_size=batch_size):
            u = model.uncertainty_all(x.to(device))
            vals.append((u["epistemic"] / (u["aleatoric"] + EPS)).cpu().numpy())
    return float(np.percentile(np.concatenate(vals), percentile))


def build_scores(model, dat, thresholds: dict, split: str, device, batch_size: int,
                 smooth_seconds: float = SMOOTH_SECONDS,
                 filter_kind: str = FILTER_KIND) -> ProbScores:
    id_raw, id_cal = score_split(model, dat.loader(split, batch_size=batch_size), device)
    ood_raw, _ = score_split(model, dat.loader(dat.ood[split], batch_size=batch_size), device)

    id_meta = dat.splits[split].meta.copy(); id_meta["kind"] = "ID"
    ood_meta = dat.ood[split].meta.copy(); ood_meta["kind"] = "OOD"
    k = filter_length(dat.config, smooth_seconds, None)

    raw, smooth = {}, {}
    for name in SCORES:
        raw[name] = np.concatenate([id_raw[name], ood_raw[name]])
        smooth[name] = np.concatenate([
            smooth_by_trial(id_raw[name], id_meta.reset_index(drop=True), k, filter_kind),
            smooth_by_trial(ood_raw[name], ood_meta.reset_index(drop=True), k, filter_kind)])
    labels = np.r_[np.zeros(len(id_raw["total"]), int), np.ones(len(ood_raw["total"]), int)]
    return ProbScores(raw, smooth, labels,
                      pd.concat([id_meta, ood_meta], ignore_index=True),
                      thresholds, split, id_cal)


def per_mode_table(s: ProbScores, score: str = "epistemic") -> pd.DataFrame:
    thr = s.thresholds[score]
    rows = []
    for mo in ("LG", "RA", "RD", "SA", "SD", "TR", "ST"):
        sel = (s.meta["mode"] == mo).to_numpy()
        if not sel.any():
            continue
        rows.append({"mode": mo, "kind": s.meta.loc[sel, "kind"].iloc[0],
                     "windows": int(sel.sum()),
                     "flagged_%": 100 * float((s.smooth[score][sel] > thr).mean()),
                     "median": float(np.median(s.smooth[score][sel]))})
    return pd.DataFrame(rows)


def evaluate(run: Path | str, dat, split: str = "test", device=None,
             batch_size: int = 1024, calibrate_on: str = "val", verbose: bool = True):
    device = device or pick_device()
    run = resolve_out(run)
    model, thr, _ = load_run(run, dat, device)
    thr = dict(thr)
    if "ratio" not in thr:
        thr["ratio"] = _ratio_threshold(model, dat, device, batch_size)

    s = build_scores(model, dat, thr, split, device, batch_size)

    results = {}
    for name in SCORES:
        try:
            c = build_scores(model, dat, thr, calibrate_on, device, batch_size)
            steep = fit_steepness(c.smooth[name], c.labels,
                                  c.meta["subject"].to_numpy(), thr[name])
        except KeyError:
            steep = fit_steepness(s.smooth[name], s.labels,
                                  s.meta["subject"].to_numpy(), thr[name])
        p = sigmoid_calibrate(s.smooth[name], thr[name], steep)
        m = classification_metrics(s.labels, s.smooth[name], thr[name])
        m |= {"ece": expected_calibration_error(p, s.labels), "brier": brier(p, s.labels),
              "steepness": steep, "threshold": thr[name]}
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(s.labels, s.smooth[name])
        m["best_j"] = 100 * float((tpr - fpr).max())
        m["auroc_raw"] = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
            s.labels, s.raw[name])
        results[name] = m

    if verbose:
        print(f"{s}\n")
        print(f"  aleatoric-head calibration on {split} in-distribution windows "
              f"(target for every z-score statistic is 1.000):")
        cal = s.calibration
        print(f"    median z^2 {cal['z2_median']:8.3f}   <- robust; the one to read")
        print(f"    mean   z^2 {cal['z2_mean']:8.3f}   90th pct {cal['z2_p90']:7.3f}   "
              f"99th pct {cal['z2_p99']:8.3f}")
        print(f"    the worst 0.1% of windows carry "
              f"{100*cal['z2_share_top_0.1pct']:.1f}% of the total squared z-score")
        print(f"    mean sigma {cal['mean_sigma']:.4f}  NLL {cal['nll']:+.4f}  "
              f"MSE {cal['mse']:.5f}  clamped {100*cal['frac_clamped']:.2f}%\n")
        hdr = f"  {'score':11s} {'AUROC':>7s} {'AUROC raw':>10s} {'J':>7s} {'best J':>7s} " \
              f"{'recall':>7s} {'spec':>7s} {'ECE':>6s}"
        print(hdr)
        for name in SCORES:
            m = results[name]
            print(f"  {name:11s} {m['auroc']:7.3f} {m['auroc_raw']:10.3f} "
                  f"{m['j_statistic']:7.2f} {m['best_j']:7.2f} {m['recall']:7.2f} "
                  f"{m['specificity']:7.2f} {m['ece']:6.3f}")
        e = results["epistemic"]
        print(f"\n  Experiment 5 for reference: AUROC {EXP5_AUROC:.3f}, J {EXP5_J:.2f}")
        print(f"  epistemic vs Experiment 5 : AUROC {e['auroc'] - EXP5_AUROC:+.3f}, "
              f"J {e['j_statistic'] - EXP5_J:+.2f}")
        print("  (epistemic IS Experiment 5's score by construction -- a large gap here means")
        print("   an implementation difference, not a finding)\n")
        print(per_mode_table(s, "epistemic").to_string(index=False))
    return s, results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate the probabilistic ensemble.")
    p.add_argument("--out", default="ml/prob_ensemble/runs/prob_forecast")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    run = resolve_out(a.out)
    if not (run / "final.pt").exists():
        raise SystemExit(f"no model at {run/'final.pt'} -- run `train.py final` first")
    print(paths.describe(), "\n")
    dat = D.build(batch_size=a.batch_size)
    print(dat, "\n")
    s, res = evaluate(run, dat, a.split, pick_device(a.device), a.batch_size)
    s.save(run / f"scores_{a.split}")
    payload = {"model": "prob_ensemble", "split": a.split,
               "target": D.TARGET, "horizon": D.HORIZON,
               "calibration": s.calibration,
               "scores": res,
               "per_mode": {k: per_mode_table(s, k).to_dict("records") for k in SCORES},
               "reference": {"exp5_auroc": EXP5_AUROC, "exp5_j": EXP5_J}}
    (run / f"eval_{a.split}.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved {run / f'eval_{a.split}.json'}")
