r"""Krogh-Vedelsby decomposition — the axis gait phase could never provide.

    python ml/vanilla/evaluation/ambiguity.py --model forecast-angle
    python ml/vanilla/evaluation/ambiguity.py --model forecast-all --split test

Experiment 1's uncertainty score is branch *disagreement*, and that is all it can ever be:
gait phase is recovered from force-plate data offline, so the reference work has no target at
run time and therefore no access to the realised error. The forecast targets of Experiments 5
and 6 are different in kind — the encoder tells you the true hip angle 200 ms later — which
makes a second, independent quantity measurable online.

For an ensemble combined by uniform averaging under squared-error loss, the decomposition is an
**exact pointwise identity**, not an approximation:

    E = Ebar - A

    A     = (1/M) sum_i (f_i - fbar)^2      across-branch variance  (the existing Psi)
    Ebar  = (1/M) sum_i (f_i - y)^2         mean individual branch error
    E     =        (fbar - y)^2             error of the ensemble mean

:func:`decompose` verifies the identity numerically on every window. A residual above
floating-point noise means the masking or the averaging is wrong, so it doubles as a
correctness check on this module and on the target construction.

**What the two axes buy.** ``A`` is the primary detector (AUROC 0.893 on Experiment 5). ``E``
turns out to be a *near-equal* detector (0.867), which contradicts what this docstring
previously claimed and is worth recording because the error was instructive.

The screening notebook predicted ``E`` would score ~0.31 — anti-correlated, the same failure as
the autoencoder's reconstruction error at 0.312. That prediction came from a **linear-
extrapolation baseline's** error, and a baseline has no training distribution to be unfamiliar
with, so its error can only rank windows by signal complexity: standing is nearly constant, so
it came out *easiest* and the score inverted. A **trained** ensemble's error behaves differently
because it has a manifold to fall off. Measured per-task median ``E`` on Experiment 5:

    LG 0.018  RA 0.019  RD 0.020  |  ST 0.038  TR 0.039  SA 0.072  SD 0.075
    \_____ the three trained-on tasks _____/    \_____ the four held-out tasks _____/

The three in-distribution tasks are the three lowest, cleanly separated from the four held-out
ones — the opposite ordering to the baseline's, and standing has moved from easiest to fourth.
The lesson generalises: an untrained baseline is the wrong instrument for predicting whether a
*trained* model's error will track novelty.

``E`` is still not a drop-in replacement for ``A``: it needs the target, so it is only available
``horizon`` samples late (200 ms here), where ``A`` is available immediately.

Its value is the **joint** reading, which a scalar score cannot express:

    high A, high E      unfamiliar movement -- the branches disagree and are wrong
    low  A, high E       the branches confidently agreed and were WRONG. Nothing in the
                         training distribution explains this: it is the signature of a sensor
                         fault or an external impact rather than novel terrain.
    high A, low  E      the branches disagree but the mean is right -- benign, often a
                         transition the ensemble has partially learned
    low  A, low  E      familiar movement, tracking correctly

The low-A/high-E quadrant is the one worth having. It is unreachable from disagreement alone,
and it is the distinction a controller actually needs: novel terrain calls for reduced
assistance, a failing sensor calls for shutdown.

Also reported is ``A / (E + eps)`` — disagreement relative to task difficulty. The reasoning was
that three dead ends in this project collapsed to a measure of signal magnitude, so dividing by
realised difficulty should quotient that out. **Measured, it does the opposite**: AUROC 0.320 on
Experiment 5, worse than either component alone and worse than chance. It is retained in the
output precisely because it was screened rather than assumed, and because a plausible-sounding
normalisation failing this badly is worth having on the record.

Runs entirely from a saved checkpoint, so no training has to be repeated.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Runnable directly (`python evaluation/ambiguity.py`) as well as through main.py, so the
# package root has to be importable either way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

import registry
from dataset import SyntheticEnsembleData
from evaluation.common import (FILTER_KIND, SMOOTH_SECONDS, classification_metrics,
                               filter_length, smooth_by_trial)
from evaluation.ensemble import load_checkpoint
from training.common import pick_device, resolve_out

__all__ = ["Decomposition", "decompose", "quadrant_table", "run"]

C_ID, C_OOD = "#4C72B0", "#DD8452"
MODE_NAMES = {"LG": "level ground", "RA": "ramp ascent", "RD": "ramp descent",
              "SA": "stair ascent", "SD": "stair descent", "ST": "standing",
              "TR": "transitions"}
MODE_COLS = {"LG": "#4C72B0", "RA": "#6FA8DC", "RD": "#9FC5E8",
             "SA": "#C44E52", "SD": "#DD8452", "ST": "#55A868", "TR": "#8172B3"}
ORDER = ("LG", "RA", "RD", "SA", "SD", "TR", "ST")
EPS = 1e-9


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote {name}.png / .pdf")


@dataclass
class Decomposition:
    """Per-window A, Ebar and E for one split, filtered and raw."""

    A: np.ndarray            # across-branch variance -- the existing Psi
    Ebar: np.ndarray         # mean individual branch error
    E: np.ndarray            # ensemble error
    A_s: np.ndarray          # each, after the causal filter
    Ebar_s: np.ndarray
    E_s: np.ndarray
    ratio_s: np.ndarray      # A / (E + eps), filtered
    labels: np.ndarray
    meta: pd.DataFrame
    threshold: float
    split: str
    residual: float          # max |E - (Ebar - A)| -- must be floating-point noise
    per_dim: dict            # per-output-channel A and E, for attribution

    def __repr__(self) -> str:
        return (f"Decomposition({self.split!r}, n={len(self.A):,}, "
                f"identity residual {self.residual:.2e})")

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), A=self.A, Ebar=self.Ebar, E=self.E,
                            A_s=self.A_s, Ebar_s=self.Ebar_s, E_s=self.E_s,
                            ratio_s=self.ratio_s, labels=self.labels,
                            threshold=self.threshold)
        self.meta.to_parquet(path.with_suffix(".parquet"), index=False)
        print(f"  wrote {path.with_suffix('.npz').name}")


@torch.no_grad()
def _collect(model, data, sp, idx_meta, device, bs, labelled: bool):
    """A, Ebar, E per window, plus their per-output-dim components."""
    model.eval()
    A, Eb, E, pdA, pdE = [], [], [], [], []
    loader = data.loader(sp, batch_size=bs)
    for batch in loader:
        x, y, m = batch
        x, y, m = x.to(device), y.to(device), m.to(device)
        preds = model(x)                                  # (b, M, D)
        fbar = preds.mean(dim=1)                          # (b, D)
        yv = y.unsqueeze(1)                               # (b, 1, D)

        a_d = preds.var(dim=1, unbiased=False)            # (b, D)
        eb_d = ((preds - yv) ** 2).mean(dim=1)            # (b, D)
        e_d = (fbar - y) ** 2                             # (b, D)

        # mask out output dims with no target, then average over the surviving ones
        w = m.clamp(0, 1)
        denom = w.sum(dim=1).clamp(min=1.0)
        A.append((a_d * w).sum(1).div(denom).cpu().numpy())
        Eb.append((eb_d * w).sum(1).div(denom).cpu().numpy())
        E.append((e_d * w).sum(1).div(denom).cpu().numpy())
        pdA.append((a_d * w).cpu().numpy())
        pdE.append((e_d * w).cpu().numpy())
    return (np.concatenate(A), np.concatenate(Eb), np.concatenate(E),
            np.concatenate(pdA), np.concatenate(pdE))


def decompose(model, data: SyntheticEnsembleData, threshold: float, split: str = "test",
              device=None, batch_size: int | None = None,
              smooth_seconds: float = SMOOTH_SECONDS,
              filter_kind: str = FILTER_KIND) -> Decomposition:
    """A, Ebar and E for the ID and OOD halves of a split.

    The OOD half is scored through the **labelled** loader deliberately. Psi never needs a
    target, but E does, and the forecast target exists for held-out tasks as readily as for
    training ones — it is just a later sensor reading. That is precisely what makes this
    decomposition possible here and impossible for gait phase.
    """
    device = device or pick_device()
    bs = batch_size or data.batch_size

    a_i, eb_i, e_i, pdA_i, pdE_i = _collect(model, data, split, None, device, bs, True)
    a_o, eb_o, e_o, pdA_o, pdE_o = _collect(model, data, data.ood[split], None, device, bs, True)

    id_meta = data.splits[split].meta.copy(); id_meta["kind"] = "ID"
    ood_meta = data.ood[split].meta.copy(); ood_meta["kind"] = "OOD"
    meta = pd.concat([id_meta, ood_meta], ignore_index=True)

    A = np.concatenate([a_i, a_o]); Ebar = np.concatenate([eb_i, eb_o])
    E = np.concatenate([e_i, e_o])
    labels = np.r_[np.zeros(len(a_i), int), np.ones(len(a_o), int)]

    # the identity is exact for uniform averaging; anything above float noise is a bug here
    residual = float(np.nanmax(np.abs(E - (Ebar - A))))

    k = filter_length(data.config, smooth_seconds, None)
    sm = lambda v_i, v_o: np.concatenate([
        smooth_by_trial(v_i, id_meta.reset_index(drop=True), k, filter_kind),
        smooth_by_trial(v_o, ood_meta.reset_index(drop=True), k, filter_kind)])
    A_s, Ebar_s, E_s = sm(a_i, a_o), sm(eb_i, eb_o), sm(e_i, e_o)
    ratio_s = A_s / (E_s + EPS)

    per_dim = {"names": data.target_names,
               "A": np.concatenate([pdA_i, pdA_o]).mean(axis=0).tolist(),
               "E": np.concatenate([pdE_i, pdE_o]).mean(axis=0).tolist(),
               "A_auroc": [], "E_auroc": []}
    pdA, pdE = np.concatenate([pdA_i, pdA_o]), np.concatenate([pdE_i, pdE_o])
    for d in range(pdA.shape[1]):
        per_dim["A_auroc"].append(float(roc_auc_score(labels, pdA[:, d])))
        per_dim["E_auroc"].append(float(roc_auc_score(labels, pdE[:, d])))

    return Decomposition(A, Ebar, E, A_s, Ebar_s, E_s, ratio_s, labels, meta,
                         threshold, split, residual, per_dim)


def quadrant_table(d: Decomposition, e_thresh: float | None = None) -> pd.DataFrame:
    """Per-task occupancy of the four (A, E) quadrants.

    ``A`` is split at the model's own 99.5th-percentile threshold. ``E`` has no calibrated
    threshold of its own, so the 95th percentile of in-distribution error is used and stated —
    it is a descriptive cut for reading the table, not a decision rule.
    """
    if e_thresh is None:
        e_thresh = float(np.percentile(d.E_s[d.labels == 0], 95))
    hi_a, hi_e = d.A_s > d.threshold, d.E_s > e_thresh
    rows = []
    for mo in ORDER:
        sel = (d.meta["mode"] == mo).to_numpy()
        if not sel.any():
            continue
        n = int(sel.sum())
        rows.append({
            "mode": mo, "kind": d.meta.loc[sel, "kind"].iloc[0], "windows": n,
            "lowA_lowE_%": 100 * float((~hi_a[sel] & ~hi_e[sel]).mean()),
            "highA_lowE_%": 100 * float((hi_a[sel] & ~hi_e[sel]).mean()),
            "lowA_highE_%": 100 * float((~hi_a[sel] & hi_e[sel]).mean()),
            "highA_highE_%": 100 * float((hi_a[sel] & hi_e[sel]).mean()),
            "median_A": float(np.median(d.A_s[sel])),
            "median_E": float(np.median(d.E_s[sel])),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def fig_scores_by_task(d: Decomposition, out: Path):
    """A, Ebar and E per task, side by side — the performance-by-task view."""
    order = [m for m in ORDER if (d.meta["mode"] == m).any()]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), sharey=True)
    panels = [(d.A_s, "$A$ — branch disagreement\n(the detector)", d.threshold),
              (d.Ebar_s, r"$\bar{E}$ — mean branch error", None),
              (d.E_s, "$E$ — ensemble error\n(NOT a detector: AUROC ~0.31)", None)]
    for ax, (v, title, thr) in zip(axes, panels):
        for i, mo in enumerate(order):
            sel = (d.meta["mode"] == mo).to_numpy()
            s = np.clip(v[sel], 1e-12, None)
            parts = ax.violinplot([np.log10(s)], positions=[i], orientation="horizontal",
                                  widths=0.8, showextrema=False, showmedians=True)
            for b in parts["bodies"]:
                b.set_facecolor(C_OOD if d.labels[sel][0] else C_ID)
                b.set_alpha(0.68); b.set_edgecolor("none")
            parts["cmedians"].set_color("black")
        if thr is not None:
            ax.axvline(np.log10(thr), color="k", ls="--", lw=1.3,
                       label=f"threshold = {thr:.2e}")
            ax.legend(fontsize=8, loc="lower right")
        ax.set_xlabel(r"$\log_{10}$ value")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="x")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{MODE_NAMES[m]}\n({m})" for m in order], fontsize=9)
    fig.suptitle("Krogh-Vedelsby components by task — blue in-distribution, orange held-out",
                 y=1.01)
    fig.tight_layout(); _save(fig, out, "kv_by_task")


def fig_quadrants(d: Decomposition, out: Path, n_plot: int = 14_000, seed: int = 0):
    """The 2-D space itself, plus per-task quadrant occupancy.

    The left panel is the whole point of this module: a scalar score collapses this plane onto
    one axis and cannot distinguish the lower-right quadrant (confidently wrong — a fault) from
    the lower-left (familiar and correct).
    """
    rng = np.random.default_rng(seed)
    e_thresh = float(np.percentile(d.E_s[d.labels == 0], 95))
    sub = rng.choice(len(d.A_s), min(n_plot, len(d.A_s)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    for mo in ORDER:
        s = sub[(d.meta["mode"].to_numpy()[sub] == mo)]
        if not len(s):
            continue
        ax.scatter(np.clip(d.E_s[s], 1e-12, None), np.clip(d.A_s[s], 1e-12, None),
                   s=3.5, alpha=0.3, c=MODE_COLS[mo], edgecolors="none",
                   rasterized=True, label=f"{mo} ({MODE_NAMES[mo]})")
    # legend outside the axes: the four quadrant annotations occupy all four corners
    lg = ax.legend(fontsize=7, markerscale=5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.13), ncol=4, framealpha=0.9)
    for h in lg.legend_handles:
        h.set_alpha(1)
    ax.axhline(d.threshold, color="k", ls="--", lw=1.2)
    ax.axvline(e_thresh, color="k", ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$E$ — ensemble error (needs the target; available at run time)")
    ax.set_ylabel(r"$A$ — branch disagreement (needs no target)")
    ax.set_title("The two-dimensional anomaly space")
    for (ax_x, ax_y, ha, va, txt, col) in (
            (0.02, 0.97, "left", "top",
             "high $A$, low $E$\ndisagree but correct\n(benign)", "#555555"),
            (0.98, 0.97, "right", "top",
             "high $A$, high $E$\nunfamiliar movement", "#8B2E2E"),
            (0.02, 0.03, "left", "bottom",
             "low $A$, low $E$\nfamiliar, tracking", "#2E5E8B"),
            (0.98, 0.03, "right", "bottom",
             "low $A$, high $E$\nCONFIDENTLY WRONG\n(sensor fault)", "#B8860B")):
        ax.text(ax_x, ax_y, txt, transform=ax.transAxes, fontsize=8, color=col,
                ha=ha, va=va, weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.72))
    ax.grid(alpha=0.2)

    per = quadrant_table(d, e_thresh)
    ax = axes[1]
    cols = ["lowA_lowE_%", "highA_lowE_%", "lowA_highE_%", "highA_highE_%"]
    labs = ["low A, low E\n(familiar)", "high A, low E\n(benign)",
            "low A, high E\n(fault-like)", "high A, high E\n(unfamiliar)"]
    shades = ["#2E5E8B", "#888888", "#B8860B", "#8B2E2E"]
    bottom = np.zeros(len(per))
    x = np.arange(len(per))
    for c, lab, sh in zip(cols, labs, shades):
        ax.bar(x, per[c], 0.66, bottom=bottom, label=lab, color=sh)
        bottom += per[c].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{'out' if k == 'OOD' else 'in'}\nn={n:,}"
                        for m, k, n in zip(per["mode"], per["kind"], per["windows"])],
                       fontsize=8)
    ax.set_ylabel("% of the task's windows"); ax.set_ylim(0, 100)
    ax.set_title(f"Quadrant occupancy by task\n($E$ cut at the 95th pct of ID error "
                 f"= {e_thresh:.2e}, descriptive only)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); _save(fig, out, "kv_quadrants")
    return per


def fig_roc_compare(d: Decomposition, out: Path):
    """Every candidate score on one ROC, so the useless ones are visibly useless."""
    cands = [(d.A_s, "$A$ — disagreement", C_ID, "-"),
             (d.E_s, "$E$ — ensemble error", "#B8860B", "--"),
             (d.Ebar_s, r"$\bar{E}$ — mean branch error", "#999999", ":"),
             (d.ratio_s, r"$A/(E+\epsilon)$", "#8172B3", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    rows = []
    for v, lab, col, ls in cands:
        ok = np.isfinite(v)
        auc = roc_auc_score(d.labels[ok], v[ok])
        fpr, tpr, _ = roc_curve(d.labels[ok], v[ok])
        axes[0].plot(fpr, tpr, lw=1.9, color=col, ls=ls, label=f"{lab} (AUROC {auc:.3f})")
        j = (tpr - fpr).max()
        rows.append({"score": lab.replace("$", ""), "auroc": auc, "best_J": 100 * j})
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="chance")
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
    axes[0].set_title("Which component actually detects?")
    axes[0].legend(fontsize=8, loc="lower right"); axes[0].grid(alpha=0.25)

    t = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    axes[1].barh(np.arange(len(t)), t["auroc"],
                 color=["#2E5E8B" if a >= 0.5 else "#B8860B" for a in t["auroc"]])
    axes[1].axvline(0.5, color="k", ls="--", lw=1.2, label="chance")
    axes[1].axvline(0.859, color="#55A868", ls=":", lw=1.4,
                    label="Experiment 1 gait phase (0.859)")
    for i, (a, s) in enumerate(zip(t["auroc"], t["score"])):
        axes[1].text(a + 0.008, i, f"{a:.3f}", va="center", fontsize=9)
    axes[1].set_yticks(np.arange(len(t))); axes[1].set_yticklabels(t["score"], fontsize=9)
    axes[1].invert_yaxis(); axes[1].set_xlim(0, 1.0)
    axes[1].set_xlabel("AUROC")
    axes[1].set_title("An AUROC below 0.5 means the score is\nanti-correlated with novelty")
    axes[1].legend(fontsize=8, loc="lower right"); axes[1].grid(alpha=0.25, axis="x")
    fig.tight_layout(); _save(fig, out, "kv_roc_compare")
    return t


def fig_per_channel(d: Decomposition, out: Path):
    """Per-output-channel A and E — which sensor tripped the flag.

    This is the fault-attribution view, and it is only meaningful for the sixteen-output model.
    Note the selection trap: choosing channels by the AUROC shown here and then reporting that
    AUROC is circular. Channel selection has to be fitted on the validation split.
    """
    names = d.per_dim["names"]
    if len(names) < 2:
        return None
    n = len(names)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, max(3.4, 0.34 * n + 1.6)))
    y = np.arange(n)
    order = np.argsort(d.per_dim["A_auroc"])[::-1]

    axes[0].barh(y - 0.2, np.array(d.per_dim["A_auroc"])[order], 0.38,
                 color=C_ID, label="$A$ (disagreement)")
    axes[0].barh(y + 0.2, np.array(d.per_dim["E_auroc"])[order], 0.38,
                 color="#B8860B", label="$E$ (error)")
    axes[0].axvline(0.5, color="k", ls="--", lw=1.2, label="chance")
    axes[0].set_yticks(y); axes[0].set_yticklabels([names[i] for i in order], fontsize=7)
    axes[0].invert_yaxis(); axes[0].set_xlim(0.3, 0.8)
    axes[0].set_xlabel("AUROC of that channel's component used alone")
    axes[0].set_title("Per-channel detection — ranked by $A$\n"
                      "(selecting on these and reporting them would be circular)",
                      fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25, axis="x")

    axes[1].barh(y - 0.2, np.array(d.per_dim["A"])[order], 0.38, color=C_ID, label="mean $A$")
    axes[1].barh(y + 0.2, np.array(d.per_dim["E"])[order], 0.38, color="#B8860B",
                 label="mean $E$")
    axes[1].set_yticks(y); axes[1].set_yticklabels([names[i] for i in order], fontsize=7)
    axes[1].invert_yaxis(); axes[1].set_xscale("log")
    axes[1].set_xlabel("mean magnitude (standardized units squared)")
    axes[1].set_title("Per-channel magnitude — how much each\nchannel contributes to the mean",
                      fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25, axis="x")
    fig.tight_layout(); _save(fig, out, "kv_per_channel")
    return pd.DataFrame({"channel": names, "A_auroc": d.per_dim["A_auroc"],
                         "E_auroc": d.per_dim["E_auroc"],
                         "mean_A": d.per_dim["A"], "mean_E": d.per_dim["E"]})


def fig_identity(d: Decomposition, out: Path, n_plot: int = 9000, seed: int = 0):
    """The decomposition verified: E against Ebar - A should lie exactly on y = x."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(d.A), min(n_plot, len(d.A)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    lhs, rhs = d.E[sub], (d.Ebar - d.A)[sub]
    axes[0].scatter(rhs, lhs, s=4, alpha=0.3, color=C_ID, edgecolors="none", rasterized=True)
    lim = [min(rhs.min(), lhs.min()), max(rhs.max(), lhs.max())]
    axes[0].plot(lim, lim, "k--", lw=1.2, label="$y = x$ (exact identity)")
    axes[0].set_xlabel(r"$\bar{E} - A$"); axes[0].set_ylabel("$E$")
    axes[0].set_title(f"Krogh-Vedelsby identity\nmax residual = {d.residual:.2e}")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.25)

    axes[1].hist(np.abs(d.E - (d.Ebar - d.A)), bins=70, color=C_ID)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$|E - (\bar{E} - A)|$")
    axes[1].set_ylabel("windows")
    axes[1].set_title("Residual distribution — should be float32 noise only")
    axes[1].grid(alpha=0.25)
    fig.tight_layout(); _save(fig, out, "kv_identity")


# ---------------------------------------------------------------- driver

def summary(d: Decomposition, scores: pd.DataFrame, quad: pd.DataFrame,
            per_ch, model_name: str) -> str:
    L = [f"# Krogh-Vedelsby decomposition — {model_name} ({d.split} split)", "",
         "Available only because the target is a future sensor reading. Gait phase is "
         "recovered from force plates offline, so Experiment 1 can measure branch "
         "disagreement and nothing else.", "",
         f"Identity `E = Ebar - A` verified: max residual **{d.residual:.2e}** "
         f"over {len(d.A):,} windows.", "",
         "## Which component detects", "",
         "| score | AUROC | best achievable J |", "|---|---|---|"]
    for _, r in scores.iterrows():
        L.append(f"| {r['score']} | **{r['auroc']:.3f}** | {r['best_J']:.1f} |")
    L += ["", "Experiment 1's gait-phase ensemble reaches AUROC 0.859 for comparison. An AUROC "
          "below 0.5 means the score is *anti*-correlated with novelty — the failure the "
          "autoencoder's reconstruction error showed at 0.312 and the screening notebook "
          "predicted for prediction error at 0.31.", "",
          "## Quadrant occupancy by task", "",
          "| task | kind | windows | low A low E | high A low E | low A high E | high A high E |",
          "|---|---|---|---|---|---|---|"]
    for _, r in quad.iterrows():
        L.append(f"| {MODE_NAMES[r['mode']]} ({r['mode']}) | {r['kind']} | {r['windows']:,} | "
                 f"{r['lowA_lowE_%']:.1f}% | {r['highA_lowE_%']:.1f}% | "
                 f"{r['lowA_highE_%']:.1f}% | {r['highA_highE_%']:.1f}% |")
    L += ["", "The **low A, high E** column is the one a scalar score cannot reach: the "
          "branches agreed and were wrong. On in-distribution tasks it should be near zero; "
          "where it is not, the ensemble is confidently mistaken and that is a fault "
          "signature rather than unfamiliar terrain.", ""]
    if per_ch is not None:
        top = per_ch.sort_values("A_auroc", ascending=False).head(6)
        L += ["## Per-channel attribution", "",
              "| channel | AUROC of A | AUROC of E |", "|---|---|---|"]
        for _, r in top.iterrows():
            L.append(f"| `{r['channel']}` | {r['A_auroc']:.3f} | {r['E_auroc']:.3f} |")
        L += ["", "**Do not select channels on this table and then report the resulting "
              "AUROC** — that is circular. Selection has to be fitted on the validation "
              "split and evaluated on test.", ""]
    L += ["## Figures", "",
          "- `kv_by_task` — A, Ebar and E per task, the performance-by-task view",
          "- `kv_quadrants` — the 2-D space and per-task quadrant occupancy",
          "- `kv_roc_compare` — every candidate score on one ROC",
          "- `kv_identity` — the decomposition verified numerically"]
    if per_ch is not None:
        L.append("- `kv_per_channel` — which sensor tripped the flag")
    return "\n".join(L)


def run(model: str = "forecast-angle", out: str | None = None, split: str = "test",
        batch_size: int = 1024, device=None) -> int:
    spec = registry.get(model)
    if spec.name not in ("forecast-angle", "forecast-all"):
        raise SystemExit(
            f"{model!r} has no run-time target, so E cannot be computed.\n"
            "This applies to forecast-angle and forecast-all only: the gait-phase and "
            "correlation targets give no future ground truth to compare against.")
    parts = spec.load()
    run_dir = resolve_out(out or spec.default_run)
    if not (run_dir / "final.pt").exists():
        raise SystemExit(f"no checkpoint at {run_dir/'final.pt'} — train it first.")

    device = pick_device(device)
    data = parts["data"](batch_size=batch_size)
    print(data)
    model_obj, threshold = load_checkpoint(run_dir / "final.pt", data, device=device)
    if not np.isfinite(threshold):
        from training.ensemble import fit_threshold
        # pass the device explicitly: fit_threshold defaults to pick_device() and calls
        # model.to(), which would silently relocate the model away from the one chosen here
        threshold, _ = fit_threshold(model_obj, data, device=device)
    print(f"threshold {threshold:.6e} | device {device}\n")

    d = decompose(model_obj, data, threshold, split, device, batch_size)
    print(d)
    if d.residual > 1e-4:
        print(f"  WARNING: identity residual {d.residual:.2e} is too large to be float noise. "
              f"The masking or the averaging is wrong -- do not trust these numbers.")
    else:
        print(f"  identity holds (residual {d.residual:.2e} = float32 noise)")

    figs = run_dir / "figures"
    print(f"\nwriting figures to {figs}")
    fig_scores_by_task(d, figs)
    quad = fig_quadrants(d, figs)
    scores = fig_roc_compare(d, figs)
    fig_identity(d, figs)
    per_ch = fig_per_channel(d, figs)

    d.save(run_dir / f"kv_{split}")
    text = summary(d, scores, quad, per_ch, spec.description)
    (run_dir / f"kv_summary_{split}.md").write_text(text)
    payload = {"model": spec.name, "split": split, "threshold": threshold,
               "identity_residual": d.residual,
               "scores": scores.to_dict("records"),
               "quadrants": quad.to_dict("records"),
               "per_channel": None if per_ch is None else per_ch.to_dict("records")}
    (run_dir / f"kv_{split}.json").write_text(json.dumps(payload, indent=2, default=float))
    if per_ch is not None:
        per_ch.to_csv(run_dir / f"kv_per_channel_{split}.csv", index=False)
    print(f"\nwrote {run_dir/f'kv_summary_{split}.md'}\n")
    print(text)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Krogh-Vedelsby decomposition for the forecast models.")
    p.add_argument("--model", default="forecast-angle",
                   choices=("forecast-angle", "forecast-all"))
    p.add_argument("--out", default=None, help="run directory (default: the model's own)")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    return run(a.model, a.out, a.split, a.batch_size, a.device)


if __name__ == "__main__":
    raise SystemExit(main())
