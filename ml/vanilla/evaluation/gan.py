"""Evaluation for the TCN GAN — the paper's procedure, with the discriminator as the scorer.

Same five steps as every other model (Section IV of the paper): score each window, filter the
score sequence within each trial, threshold at the 99.5th percentile of in-distribution
training scores, compare against the known label, report metrics. Only the score differs:

    Psi(x) = 1 - D(x)

There is no latent to inspect and no alternative score to compare against, so this module is
markedly shorter than ``evaluation/autoencoder.py``. What it adds instead is a **degeneracy
check**, because this model has a specific way of failing silently: if the discriminator has
learned to separate real from generated too well, ``D(x)`` saturates near 1 for every real
window and ``Psi`` carries no information. The metrics would then be uninformative rather than
wrong, which is harder to notice. :func:`degeneracy_report` measures it directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import GanData
from evaluation.common import (FILTER_KIND, SMOOTH_SECONDS, brier, classification_metrics,
                               expected_calibration_error, filter_length, fit_steepness,
                               sigmoid_calibrate, smooth_by_trial)
from models.gan import create_gan
from training.common import pick_device, resolve_out
from training.gan import score_split

__all__ = ["GanScores", "load_gan_run", "score_bundle", "evaluate_gan",
           "per_mode_table", "degeneracy_report"]


@dataclass
class GanScores:
    """Per-window discriminator scores for one split, with provenance."""

    psi: np.ndarray               # 1 - D(x), raw
    psi_smooth: np.ndarray        # after the causal filter
    labels: np.ndarray            # 1 = OOD
    meta: pd.DataFrame
    threshold: float
    split: str

    def __repr__(self) -> str:
        return (f"GanScores({self.split!r}, n={len(self.psi):,}, "
                f"{int((self.labels==1).sum()):,} OOD / {int((self.labels==0).sum()):,} ID)")

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), psi=self.psi,
                            psi_smooth=self.psi_smooth, labels=self.labels,
                            threshold=self.threshold)
        self.meta.to_parquet(path.with_suffix(".parquet"), index=False)
        print(f"  wrote {path.with_suffix('.npz').name} and {path.with_suffix('.parquet').name}")

    @staticmethod
    def load(path: Path, split: str = "test") -> "GanScores":
        path = Path(path)
        d = np.load(path.with_suffix(".npz"))
        return GanScores(d["psi"], d["psi_smooth"], d["labels"],
                         pd.read_parquet(path.with_suffix(".parquet")),
                         float(d["threshold"]), split)


def load_gan_run(run: Path | str, data: GanData, device=None):
    """Rebuild the model and threshold from a ``training/gan.py`` run directory."""
    run = resolve_out(run)
    device = device or pick_device()
    ck = torch.load(run / "final.pt", map_location=device, weights_only=False)
    model = create_gan(data.n_channels, data.window)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, float(ck["threshold"])


def score_bundle(model, data: GanData, threshold: float, split: str = "test",
                 device=None, batch_size: int | None = None,
                 smooth_seconds: float = SMOOTH_SECONDS,
                 filter_kind: str = FILTER_KIND,
                 filter_scores: int | None = None) -> GanScores:
    """Score the ID and OOD halves of a split and filter within each trial."""
    device = device or pick_device()
    bs = batch_size or data.batch_size

    psi_id = score_split(model, data.loader(split, batch_size=bs), device)
    psi_ood = score_split(model, data.ood_loader(split, batch_size=bs), device)

    id_meta = data.splits[split].meta.copy(); id_meta["kind"] = "ID"
    ood_meta = data.ood[split].meta.copy(); ood_meta["kind"] = "OOD"

    k = filter_length(data.config, smooth_seconds, filter_scores)
    sm_id = smooth_by_trial(psi_id, id_meta.reset_index(drop=True), k, filter_kind)
    sm_ood = smooth_by_trial(psi_ood, ood_meta.reset_index(drop=True), k, filter_kind)

    return GanScores(np.concatenate([psi_id, psi_ood]),
                     np.concatenate([sm_id, sm_ood]),
                     np.concatenate([np.zeros(len(psi_id), int), np.ones(len(psi_ood), int)]),
                     pd.concat([id_meta, ood_meta], ignore_index=True),
                     threshold, split)


def per_mode_table(s: GanScores) -> pd.DataFrame:
    rows = []
    for mo in ("LG", "RA", "RD", "SA", "SD", "TR", "ST"):
        sel = (s.meta["mode"] == mo).to_numpy()
        if not sel.any():
            continue
        rows.append({"mode": mo, "kind": s.meta.loc[sel, "kind"].iloc[0],
                     "windows": int(sel.sum()),
                     "flagged_%": 100 * float((s.psi_smooth[sel] > s.threshold).mean()),
                     "median_psi": float(np.median(s.psi_smooth[sel]))})
    return pd.DataFrame(rows)


def degeneracy_report(s: GanScores, verbose: bool = True) -> dict:
    """Has the discriminator saturated, making Psi uninformative?

    The failure is specific to this architecture: D is trained to separate real from
    *generated*, and every OOD window is real hip data. A discriminator that wins outputs ~1
    for all real input, so Psi collapses to ~0 and the metrics become uninformative rather than
    visibly wrong. Three signals, none of which needs a label to compute.
    """
    psi = s.psi_smooth
    q1, q3 = np.percentile(psi, [25, 75])
    frac_low = float((psi < 1e-3).mean())
    d = {"psi_median": float(np.median(psi)), "psi_iqr": float(q3 - q1),
         "psi_min": float(psi.min()), "psi_max": float(psi.max()),
         "fraction_near_zero": frac_low,
         "distinct_values": int(len(np.unique(np.round(psi, 6))))}
    d["degenerate"] = bool(d["psi_iqr"] < 1e-3 or frac_low > 0.95)
    if verbose:
        print(f"  discriminator health: Psi median {d['psi_median']:.5f}, "
              f"IQR {d['psi_iqr']:.5f}, range [{d['psi_min']:.4f}, {d['psi_max']:.4f}]")
        print(f"    {100*frac_low:.1f}% of windows score below 1e-3; "
              f"{d['distinct_values']:,} distinct values")
        if d["degenerate"]:
            print("    DEGENERATE: the discriminator has saturated. Psi carries almost no "
                  "information, so the metrics below are uninformative rather than poor.")
    return d


def evaluate_gan(run: Path | str, data: GanData, split: str = "test", device=None,
                 batch_size: int | None = None, calibrate_on: str = "val",
                 verbose: bool = True):
    """Full evaluation. Returns ``(scores, metrics)``, mirroring the other evaluators."""
    device = device or pick_device()
    run = resolve_out(run)
    model, threshold = load_gan_run(run, data, device)
    s = score_bundle(model, data, threshold, split, device, batch_size)

    try:
        c = score_bundle(model, data, threshold, calibrate_on, device, batch_size)
        steep = fit_steepness(c.psi_smooth, c.labels, c.meta["subject"].to_numpy(), threshold)
    except KeyError:
        steep = fit_steepness(s.psi_smooth, s.labels, s.meta["subject"].to_numpy(), threshold)

    p = sigmoid_calibrate(s.psi_smooth, threshold, steep)
    m = classification_metrics(s.labels, s.psi_smooth, threshold)
    n_id = int((s.labels == 0).sum())
    m |= {"ece": expected_calibration_error(p, s.labels), "brier": brier(p, s.labels),
          "steepness": steep, "n_id": n_id, "n_ood": int((s.labels == 1).sum()),
          "pct_ood": 100 * float((s.labels == 1).mean())}

    if verbose:
        print(f"{s}\n")
        health = degeneracy_report(s)
        m["degeneracy"] = health
        print()
        ref = {"accuracy": "89.4", "precision": "74.9", "recall": "68.7",
               "f1": "71.5", "j_statistic": "63.0"}
        print(f"  {'metric':16s} {'GAN':>10s}   paper (GAN)")
        for key in ("j_statistic", "auroc", "accuracy", "recall", "specificity",
                    "precision", "f1"):
            print(f"  {key:16s} {m[key]:10.3f}   {ref.get(key, '—'):>6s}")
        print(f"  {'ece':16s} {m['ece']:10.3f}   0.06")
        print(f"  {'brier':16s} {m['brier']:10.3f}   0.09")
        print(f"\n  {n_id:,} ID + {m['n_ood']:,} OOD ({m['pct_ood']:.1f}% OOD; "
              f"the paper's test set was 80.1%)\n")
        print(per_mode_table(s).to_string(index=False))
    else:
        m["degeneracy"] = degeneracy_report(s, verbose=False)
    return s, m


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Evaluate the GAN.")
    p.add_argument("--out", default="ml/vanilla/runs/gan")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    run = resolve_out(a.out)
    if not (run / "final.pt").exists():
        raise SystemExit(f"no trained model at {run/'final.pt'} — run training/gan.py first.")

    data = GanData(batch_size=a.batch_size)
    print(data, "\n")
    s, metrics = evaluate_gan(run, data, a.split, pick_device(a.device), a.batch_size)
    s.save(run / f"scores_{a.split}")
    (run / f"eval_{a.split}.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nsaved {run / f'eval_{a.split}.json'}")
