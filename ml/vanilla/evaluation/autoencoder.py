"""Evaluation for the convolutional autoencoder — the paper's procedure, plus latent analysis.

Scoring follows the same five steps as the ensemble (Section IV of the paper): score each
window, apply a causal filter to the score sequence, threshold at the 99.5th percentile of
in-distribution training scores, compare against the known label, report metrics.

Two things are evaluated side by side:

* **Psi = LOF on the latent space** — the paper's chosen score.
* **reconstruction error** — the score the paper *rejected*, kept so that rejection is
  reproducible rather than assumed. The paper's reasoning was that stationary data such as
  standing reconstructs too easily; on hip data that prediction is directly testable.

Beyond the paper's metrics this module extracts the **latent representation itself**, with every
window traceable back to its participant, trial, ambulation mode, speed and position in the
recording. That is what makes the latent plots in ``figures_ae.py`` interpretable — a cluster
can be asked "which trials are you?" rather than merely observed.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.autoencoder import LatentLOF, create_autoencoder
from dataset import AutoencoderData
from evaluation.common import (FILTER_KIND, SMOOTH_SECONDS, brier,
                               classification_metrics, expected_calibration_error,
                               filter_length, fit_steepness, sigmoid_calibrate,
                               smooth_by_trial)
from training.common import pick_device, resolve_out
from training.autoencoder import encode_split

__all__ = ["LatentBundle", "load_ae_run", "extract_latents", "score_bundle",
           "evaluate_autoencoder", "trials_near"]


@dataclass
class LatentBundle:
    """Latents and scores for one split, with full provenance per window.

    ``meta`` has one row per latent vector, carrying ``subject``, ``trial``, ``mode``,
    ``speed``, ``controller``, ``end_idx`` and ``kind`` (ID / OOD). Row *i* of ``Z``
    corresponds to row *i* of ``meta``, so any point in a latent plot can be traced back to
    the recording it came from.
    """

    Z: np.ndarray                 # (n, latent_size)
    psi: np.ndarray               # LOF score, larger = more anomalous
    psi_smooth: np.ndarray        # after the causal filter
    recon: np.ndarray             # reconstruction error (the rejected score)
    recon_smooth: np.ndarray
    labels: np.ndarray            # 1 = OOD
    meta: pd.DataFrame
    threshold: float
    split: str

    def __repr__(self) -> str:
        return (f"LatentBundle({self.split!r}, n={len(self.Z):,}, latent={self.Z.shape[1]}, "
                f"{int((self.labels==1).sum()):,} OOD / {int((self.labels==0).sum()):,} ID)")

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), Z=self.Z, psi=self.psi,
                            psi_smooth=self.psi_smooth, recon=self.recon,
                            recon_smooth=self.recon_smooth, labels=self.labels,
                            threshold=self.threshold)
        self.meta.to_parquet(path.with_suffix(".parquet"), index=False)
        print(f"  wrote {path.with_suffix('.npz').name} and {path.with_suffix('.parquet').name}")

    @staticmethod
    def load(path: Path, split: str = "test") -> "LatentBundle":
        path = Path(path)
        d = np.load(path.with_suffix(".npz"))
        return LatentBundle(d["Z"], d["psi"], d["psi_smooth"], d["recon"], d["recon_smooth"],
                            d["labels"], pd.read_parquet(path.with_suffix(".parquet")),
                            float(d["threshold"]), split)


def load_ae_run(run: Path | str, data: AutoencoderData, device=None):
    """Rebuild the model, LOF and threshold from a ``train_ae.py`` run directory."""
    run = resolve_out(run)
    device = device or pick_device()
    ck = torch.load(run / "final.pt", map_location=device, weights_only=False)
    model = create_autoencoder(data.n_channels, data.window)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    with open(run / "lof.pkl", "rb") as f:
        lof: LatentLOF = pickle.load(f)
    return model, lof, float(ck["threshold"])


def extract_latents(model, lof, data: AutoencoderData, split: str = "test",
                    device=None, batch_size: int | None = None) -> tuple:
    """Encode the ID and OOD halves of a split, keeping provenance aligned."""
    device = device or pick_device()
    bs = batch_size or data.batch_size

    Z_id, e_id = encode_split(model, data.loader(split, batch_size=bs), device)
    Z_ood, e_ood = encode_split(model, data.ood_loader(split, batch_size=bs), device)

    id_meta = data.splits[split].meta.copy(); id_meta["kind"] = "ID"
    ood_meta = data.ood[split].meta.copy(); ood_meta["kind"] = "OOD"
    meta = pd.concat([id_meta, ood_meta], ignore_index=True)

    Z = np.concatenate([Z_id, Z_ood])
    recon = np.concatenate([e_id, e_ood])
    labels = np.concatenate([np.zeros(len(Z_id), int), np.ones(len(Z_ood), int)])
    psi = lof.score(Z)
    if not (len(Z) == len(meta) == len(labels) == len(psi)):
        raise RuntimeError("latent/meta/label lengths disagree")
    return Z, psi, recon, labels, meta, (len(Z_id), len(Z_ood))


def score_bundle(Z, psi, recon, labels, meta, threshold, data, split,
                 smooth_seconds: float = SMOOTH_SECONDS, filter_kind: str = FILTER_KIND,
                 filter_scores: int | None = None) -> LatentBundle:
    """Apply the paper's causal filter within each trial, for both candidate scores."""
    cfg = data.config
    rate, stride = cfg.get("sample_rate_hz", 200), cfg.get("stride_samples", 10)
    k = filter_scores if filter_scores is not None else max(1, round(smooth_seconds * rate / stride))

    n_id = int((labels == 0).sum())
    def smooth(v):
        a = smooth_by_trial(v[:n_id], meta.iloc[:n_id].reset_index(drop=True), k, filter_kind)
        b = smooth_by_trial(v[n_id:], meta.iloc[n_id:].reset_index(drop=True), k, filter_kind)
        return np.concatenate([a, b])

    return LatentBundle(Z, psi, smooth(psi), recon, smooth(recon), labels, meta, threshold, split)


def evaluate_autoencoder(run: Path | str, data: AutoencoderData, split: str = "test",
                         device=None, batch_size: int | None = None,
                         calibrate_on: str = "val", verbose: bool = True):
    """Full evaluation. Returns ``(bundle, metrics)`` and mirrors ``eval.evaluate_model``."""
    device = device or pick_device()
    model, lof, threshold = load_ae_run(run, data, device)

    Z, psi, recon, labels, meta, (n_id, n_ood) = extract_latents(
        model, lof, data, split, device, batch_size)
    bundle = score_bundle(Z, psi, recon, labels, meta, threshold, data, split)

    # steepness for the calibration metrics, fitted on a different split
    try:
        Zc, pc, rc, lc, mc, _ = extract_latents(model, lof, data, calibrate_on, device, batch_size)
        bc = score_bundle(Zc, pc, rc, lc, mc, threshold, data, calibrate_on)
        steep = fit_steepness(bc.psi_smooth, bc.labels, bc.meta["subject"].to_numpy(), threshold)
    except KeyError:
        steep = fit_steepness(bundle.psi_smooth, bundle.labels,
                              bundle.meta["subject"].to_numpy(), threshold)

    p = sigmoid_calibrate(bundle.psi_smooth, threshold, steep)
    m = classification_metrics(bundle.labels, bundle.psi_smooth, threshold)
    m |= {"ece": expected_calibration_error(p, bundle.labels),
          "brier": brier(p, bundle.labels), "steepness": steep,
          "n_id": n_id, "n_ood": n_ood, "pct_ood": 100 * n_ood / (n_id + n_ood),
          "latent_dim": int(Z.shape[1])}

    # the paper's rejected score, on the same windows, thresholded the same way
    thr_r = float(np.percentile(bundle.recon_smooth[bundle.labels == 0], 99.5))
    m["recon"] = classification_metrics(bundle.labels, bundle.recon_smooth, thr_r)

    if verbose:
        print(f"{bundle}\n")
        print(f"  {'metric':16s} {'LOF(latent)':>12s} {'recon error':>12s}   paper (AE)")
        ref = {"accuracy": "69.8", "precision": "40.0", "recall": "94.7",
               "f1": "55.7", "j_statistic": "58.5"}
        for key in ("j_statistic", "auroc", "accuracy", "recall", "specificity", "precision", "f1"):
            rv = m["recon"][key]
            print(f"  {key:16s} {m[key]:12.3f} {rv:12.3f}   {ref.get(key,'—'):>6s}")
        print(f"  {'ece':16s} {m['ece']:12.3f} {'—':>12s}   0.26")
        print(f"  {'brier':16s} {m['brier']:12.3f} {'—':>12s}   0.18")
        print(f"\n  {m['n_id']:,} ID + {m['n_ood']:,} OOD ({m['pct_ood']:.1f}% OOD; "
              f"the paper's test set was 80.1%)")

        per = per_mode_table(bundle)
        print()
        print(per.to_string(index=False))
    return bundle, m


def per_mode_table(b: LatentBundle) -> pd.DataFrame:
    rows = []
    for mo in ("LG", "RA", "RD", "SA", "SD", "TR", "ST"):
        sel = (b.meta["mode"] == mo).to_numpy()
        if not sel.any():
            continue
        thr_r = float(np.percentile(b.recon_smooth[b.labels == 0], 99.5))
        rows.append({"mode": mo, "kind": b.meta.loc[sel, "kind"].iloc[0],
                     "windows": int(sel.sum()),
                     "flagged_LOF_%": 100 * float((b.psi_smooth[sel] > b.threshold).mean()),
                     "flagged_recon_%": 100 * float((b.recon_smooth[sel] > thr_r).mean()),
                     "median_psi": float(np.median(b.psi_smooth[sel])),
                     "median_recon": float(np.median(b.recon_smooth[sel]))})
    return pd.DataFrame(rows)


def trials_near(b: LatentBundle, pc: np.ndarray, centre, radius: float,
                top: int = 15) -> pd.DataFrame:
    """Which trials occupy a region of the latent projection?

    This is the traceability tool: give it a point in the 2-D projection and a radius, and it
    reports which participant/trial/mode the windows in that neighbourhood came from, so a
    cluster can be identified rather than guessed at.

        pcs = pca.transform(bundle.Z)          # or whatever projection the figure used
        trials_near(bundle, pcs, centre=(-4.0, 1.5), radius=1.0)
    """
    d = np.linalg.norm(pc[:, :2] - np.asarray(centre, float)[None, :], axis=1)
    sel = d <= radius
    if not sel.any():
        return pd.DataFrame(columns=["subject", "trial", "mode", "windows", "share_%"])
    sub = b.meta.loc[sel].copy()
    g = (sub.groupby(["subject", "trial", "mode", "kind"], observed=True)
         .size().rename("windows").reset_index())
    g["share_%"] = (100 * g["windows"] / sel.sum()).round(1)
    return g.sort_values("windows", ascending=False).head(top).reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Evaluate the autoencoder.")
    p.add_argument("--out", default="ml/vanilla/runs/autoencoder")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    run = resolve_out(a.out)
    if not (run / "final.pt").exists():
        raise SystemExit(f"no trained model at {run/'final.pt'} — run train_ae.py first.")

    data = AutoencoderData(batch_size=a.batch_size)
    print(data, "\n")
    bundle, metrics = evaluate_autoencoder(run, data, a.split,
                                           pick_device(a.device), a.batch_size)
    bundle.save(run / f"latents_{a.split}")
    (run / f"eval_{a.split}.json").write_text(json.dumps(
        {k: v for k, v in metrics.items() if k != "recon"} |
        {"recon_score": metrics["recon"]},
        indent=2, default=float))
    print(f"\nsaved {run / f'eval_{a.split}.json'}")
