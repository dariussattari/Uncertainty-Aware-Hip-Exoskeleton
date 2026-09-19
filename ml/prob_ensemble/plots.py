r"""Figures for the probabilistic ensemble.

    python plots.py                      # uses runs/prob_forecast
    python plots.py --out <run> --split test

Reproduces the reporting used for every other model in this project — per-task score
distributions, a ROC, per-task detection — so the results are directly comparable, and adds
the three views only this model can produce:

* **Calibration.** Every earlier score in this project is assessed by ranking alone, because
  none of them claims a scale. A predicted variance does, so it can be checked against the
  realised residual. ``z_var`` has a correct value of exactly 1, which makes it the only
  diagnostic here that can be wrong rather than merely disappointing.
* **The aleatoric/epistemic plane.** The analogue of the Krogh-Vedelsby quadrants, but along
  axes that are available without waiting for the target.
* **Predicted sigma by task**, which is the aleatoric story told directly: where does the model
  think the next 200 ms is intrinsically unpredictable?

Writes PNG + PDF to ``<run>/figures/`` and a Markdown summary to ``<run>/summary.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import paths  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from training.common import resolve_out

C_ID, C_OOD = "#4C72B0", "#DD8452"
MODE_NAMES = {"LG": "level ground", "RA": "ramp ascent", "RD": "ramp descent",
              "SA": "stair ascent", "SD": "stair descent", "ST": "standing",
              "TR": "transitions"}
MODE_COLS = {"LG": "#4C72B0", "RA": "#6FA8DC", "RD": "#9FC5E8",
             "SA": "#C44E52", "SD": "#DD8452", "ST": "#55A868", "TR": "#8172B3"}
ORDER = ("LG", "RA", "RD", "SA", "SD", "TR", "ST")
SCORES = ("epistemic", "aleatoric", "total", "ratio")
SCORE_LABEL = {"epistemic": r"$\sigma^2_{\mathrm{epi}}$  (disagreement)",
               "aleatoric": r"$\sigma^2_{\mathrm{alea}}$  (intrinsic)",
               "total": r"$\sigma^2_{\mathrm{total}}$",
               "ratio": r"$\sigma^2_{\mathrm{epi}} / \sigma^2_{\mathrm{alea}}$"}
SCORE_COL = {"epistemic": "#2E5E8B", "aleatoric": "#B8860B",
             "total": "#55A868", "ratio": "#8172B3"}
EXP5_AUROC, EXP5_J = 0.896, 47.64
EXP1_AUROC = 0.859


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote {name}.png / .pdf")


class Run:
    """Everything a figure needs, loaded once."""

    def __init__(self, run: Path, split: str = "test"):
        self.dir, self.split = run, split
        d = np.load(run / f"scores_{split}.npz")
        self.labels = d["labels"]
        self.raw = {k: d[f"raw_{k}"] for k in SCORES}
        self.smooth = {k: d[f"smooth_{k}"] for k in SCORES}
        self.meta = pd.read_parquet(run / f"scores_{split}.parquet")
        self.eval = json.loads((run / f"eval_{split}.json").read_text())
        self.report = json.loads((run / "report.json").read_text())
        self.loso = json.loads((run / "loso_summary.json").read_text())
        self.thr = {k: self.eval["scores"][k]["threshold"] for k in SCORES}

    def m(self, score: str) -> dict:
        return self.eval["scores"][score]


# ---------------------------------------------------------------- the paper's reporting

def fig_scores_by_task(r: Run, out: Path):
    """All four candidate scores per task, on one row for comparison."""
    order = [m for m in ORDER if (r.meta["mode"] == m).any()]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4), sharey=True)
    for ax, name in zip(axes, SCORES):
        v = r.smooth[name]
        for i, mo in enumerate(order):
            sel = (r.meta["mode"] == mo).to_numpy()
            s = np.clip(v[sel], 1e-12, None)
            parts = ax.violinplot([np.log10(s)], positions=[i], orientation="horizontal",
                                  widths=0.8, showextrema=False, showmedians=True)
            for b in parts["bodies"]:
                b.set_facecolor(C_OOD if r.labels[sel][0] else C_ID)
                b.set_alpha(0.68); b.set_edgecolor("none")
            parts["cmedians"].set_color("black")
        ax.axvline(np.log10(max(r.thr[name], 1e-12)), color="k", ls="--", lw=1.2)
        ax.set_xlabel(r"$\log_{10}$ value")
        ax.set_title(f"{SCORE_LABEL[name]}\nAUROC {r.m(name)['auroc']:.3f}", fontsize=10)
        ax.grid(alpha=0.25, axis="x")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{MODE_NAMES[m]}\n({m})" for m in order], fontsize=9)
    fig.suptitle("Candidate scores by task — blue in-distribution, orange held-out; "
                 "dashed line is each score's own 99.5th-percentile threshold", y=1.02)
    fig.tight_layout(); _save(fig, out, "scores_by_task")


def fig_roc(r: Run, out: Path):
    """Every score on one ROC, against the two models this one has to beat."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    for name in SCORES:
        fpr, tpr, _ = roc_curve(r.labels, r.smooth[name])
        axes[0].plot(fpr, tpr, lw=1.9, color=SCORE_COL[name],
                     label=f"{SCORE_LABEL[name]} ({r.m(name)['auroc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="chance")
    e = r.m("epistemic")
    axes[0].plot(1 - e["specificity"] / 100, e["recall"] / 100, "o", ms=9,
                 color=SCORE_COL["epistemic"], zorder=5,
                 label=f"99.5th-pct operating point (J = {e['j_statistic']:.1f})")
    axes[0].set_xlabel("false positive rate  (ID wrongly flagged)")
    axes[0].set_ylabel("true positive rate  (OOD caught)")
    axes[0].set_title("Which component detects?")
    axes[0].legend(fontsize=8, loc="lower right"); axes[0].grid(alpha=0.25)
    axes[0].set_xlim(-0.02, 1.02); axes[0].set_ylim(-0.02, 1.02)

    names = list(SCORES)
    vals = [r.m(n)["auroc"] for n in names]
    ax = axes[1]
    ax.barh(np.arange(len(names)), vals, color=[SCORE_COL[n] for n in names])
    ax.axvline(0.5, color="k", ls="--", lw=1.1, label="chance")
    ax.axvline(EXP5_AUROC, color="#C44E52", ls="-", lw=1.6,
               label=f"Experiment 5, same target (0.896)")
    ax.axvline(EXP1_AUROC, color="#999999", ls=":", lw=1.6,
               label=f"Experiment 1, gait phase (0.859)")
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([SCORE_LABEL[n] for n in names], fontsize=10)
    ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_xlabel("AUROC")
    ax.set_title("The epistemic score is Experiment 5's score by construction,\n"
                 "so the red line is a verification target, not a competitor", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25, axis="x")
    fig.tight_layout(); _save(fig, out, "roc_compare")


def fig_detection_by_task(r: Run, out: Path, score: str = "epistemic"):
    per = pd.DataFrame(r.eval["per_mode"][score])
    per = per.set_index("mode").reindex([m for m in ORDER if m in set(per["mode"])]).reset_index()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    cols = [C_OOD if k == "OOD" else C_ID for k in per["kind"]]
    ax.bar(np.arange(len(per)), per["flagged_%"], 0.62, color=cols)
    for i, v in enumerate(per["flagged_%"]):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(per)))
    ax.set_xticklabels([f"{MODE_NAMES[m]}\n({m})\nn={n:,}"
                        for m, n in zip(per["mode"], per["windows"])], fontsize=8)
    ax.set_ylabel("% of windows flagged"); ax.set_ylim(0, 115)
    ax.set_title(f"Detection by task, {SCORE_LABEL[score]} at its own threshold — "
                 "blue is in-distribution, where lower is better")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); _save(fig, out, "detection_by_task")


# ---------------------------------------------------------------- what only this model can show

def fig_calibration(r: Run, out: Path):
    """The diagnostic with a known answer, across folds and across splits.

    z_var is the variance of the standardised residual. A calibrated model gives exactly 1.
    This is the only number in the project that can be *wrong* rather than merely low, and the
    interesting result is that it is near 1 on training participants and roughly 2 on held-out
    ones.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))

    # per-fold held-out calibration
    z = r.loso.get("val_z_var", {})
    subs = [s for s in r.loso["best_epochs"] if z.get(s) is not None]
    vals = [z[s] for s in subs]
    ax = axes[0]
    ax.bar(np.arange(len(subs)), vals, color=["#C44E52" if v > 1.5 else "#B8860B" for v in vals])
    ax.axhline(1.0, color="k", ls="--", lw=1.4, label="calibrated (target)")
    ax.axhline(float(np.mean(vals)), color="#2E5E8B", ls=":", lw=1.6,
               label=f"mean {np.mean(vals):.2f}")
    ax.set_xticks(np.arange(len(subs))); ax.set_xticklabels(subs, rotation=60, fontsize=8)
    ax.set_ylabel(r"$\mathrm{Var}[z]$ on the held-out participant")
    ax.set_title("Calibration per LOSO fold\n(>1 = over-confident)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")

    # training curve of z_var: the warm-up transition is visible
    h = pd.DataFrame(r.report["stage2_history"])
    ax = axes[1]
    warm = h["phase"] == "warmup"
    ax.plot(h["epoch"], h["z_var"], "o-", ms=4, color="#2E5E8B", label=r"$\mathrm{Var}[z]$")
    ax.axhline(1.0, color="k", ls="--", lw=1.4, label="calibrated")
    if warm.any():
        ax.axvspan(-0.5, h.loc[warm, "epoch"].max() + 0.5, color="#999999", alpha=0.15,
                   label="warm-up (squared error only)")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$\mathrm{Var}[z]$")
    ax.set_title("Calibration during stage 2\nthe variance head only trains after warm-up",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # mean sigma against the residual it is supposed to describe
    ax = axes[2]
    ax.plot(h["epoch"], h["mean_sigma"], "o-", ms=4, color="#B8860B",
            label=r"predicted $\bar{\sigma}$")
    ax.plot(h["epoch"], np.sqrt(h["train_mse"]), "s-", ms=4, color="#C44E52",
            label=r"realised RMSE of the mean")
    if warm.any():
        ax.axvspan(-0.5, h.loc[warm, "epoch"].max() + 0.5, color="#999999", alpha=0.15)
    ax.set_xlabel("epoch"); ax.set_ylabel("standardised units"); ax.set_yscale("log")
    ax.set_title("Predicted vs realised spread\nthey should converge, and do", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); _save(fig, out, "calibration")


def fig_plane(r: Run, out: Path, n_plot: int = 14_000, seed: int = 0):
    """The aleatoric/epistemic plane, and per-task occupancy of its quadrants.

    The analogue of the Krogh-Vedelsby quadrants of Section 27.3, but on axes that need no
    target and so are both available at run time.
    """
    rng = np.random.default_rng(seed)
    a, e = r.smooth["aleatoric"], r.smooth["epistemic"]
    ta, te = r.thr["aleatoric"], r.thr["epistemic"]
    sub = rng.choice(len(a), min(n_plot, len(a)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2),
                             gridspec_kw={"width_ratios": [1, 1.12]})
    ax = axes[0]
    for mo in ORDER:
        s = sub[(r.meta["mode"].to_numpy()[sub] == mo)]
        if not len(s):
            continue
        ax.scatter(np.clip(a[s], 1e-12, None), np.clip(e[s], 1e-12, None), s=3.5, alpha=0.3,
                   c=MODE_COLS[mo], edgecolors="none", rasterized=True,
                   label=f"{mo} ({MODE_NAMES[mo]})")
    lg = ax.legend(fontsize=7, markerscale=5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.13), ncol=4, framealpha=0.9)
    for h_ in lg.legend_handles:
        h_.set_alpha(1)
    ax.axhline(te, color="k", ls="--", lw=1.2)
    ax.axvline(ta, color="k", ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma^2_{\mathrm{alea}}$ — intrinsic unpredictability")
    ax.set_ylabel(r"$\sigma^2_{\mathrm{epi}}$ — disagreement")
    ax.set_title("The two-dimensional space, both axes available at run time")
    for (x, y, ha, va, txt, col) in (
            (0.02, 0.97, "left", "top", "unfamiliar,\nbut predictable", "#8B2E2E"),
            (0.98, 0.97, "right", "top", "unfamiliar AND\nintrinsically variable", "#2E5E8B"),
            (0.02, 0.03, "left", "bottom", "familiar,\ntracking", "#2E5E8B"),
            (0.98, 0.03, "right", "bottom", "familiar but\nintrinsically variable", "#B8860B")):
        ax.text(x, y, txt, transform=ax.transAxes, fontsize=8, color=col, ha=ha, va=va,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.72))
    ax.grid(alpha=0.2)

    hi_a, hi_e = a > ta, e > te
    rows = []
    for mo in ORDER:
        sel = (r.meta["mode"] == mo).to_numpy()
        if not sel.any():
            continue
        rows.append({"mode": mo, "kind": r.meta.loc[sel, "kind"].iloc[0],
                     "n": int(sel.sum()),
                     "lo_lo": 100 * float((~hi_a[sel] & ~hi_e[sel]).mean()),
                     "lo_hi": 100 * float((~hi_a[sel] & hi_e[sel]).mean()),
                     "hi_lo": 100 * float((hi_a[sel] & ~hi_e[sel]).mean()),
                     "hi_hi": 100 * float((hi_a[sel] & hi_e[sel]).mean())})
    q = pd.DataFrame(rows)
    ax = axes[1]
    bottom = np.zeros(len(q)); x = np.arange(len(q))
    for col, lab, sh in (("lo_lo", "low both (familiar)", "#2E5E8B"),
                         ("hi_lo", "high alea only (variable)", "#B8860B"),
                         ("lo_hi", "high epi only (unfamiliar)", "#8B2E2E"),
                         ("hi_hi", "high both", "#555555")):
        ax.bar(x, q[col], 0.66, bottom=bottom, label=lab, color=sh)
        bottom += q[col].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{'out' if k == 'OOD' else 'in'}\nn={n:,}"
                        for m, k, n in zip(q["mode"], q["kind"], q["n"])], fontsize=8)
    ax.set_ylabel("% of the task's windows"); ax.set_ylim(0, 100)
    ax.set_title("Quadrant occupancy by task\n(each axis split at its own 99.5th percentile)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); _save(fig, out, "alea_epi_plane")
    return q


def fig_sigma_by_task(r: Run, out: Path):
    """Where does the model think the next 200 ms is intrinsically unpredictable?"""
    order = [m for m in ORDER if (r.meta["mode"] == m).any()]
    sig = np.sqrt(np.clip(r.smooth["aleatoric"], 0, None))
    epi = np.sqrt(np.clip(r.smooth["epistemic"], 0, None))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.38
    xs = np.arange(len(order))
    ax.bar(xs - w / 2, [np.median(sig[(r.meta["mode"] == m).to_numpy()]) for m in order], w,
           color="#B8860B", label=r"predicted $\sigma_{\mathrm{alea}}$")
    ax.bar(xs + w / 2, [np.median(epi[(r.meta["mode"] == m).to_numpy()]) for m in order], w,
           color="#2E5E8B", label=r"$\sigma_{\mathrm{epi}}$ (disagreement)")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{MODE_NAMES[m]}\n({m})" for m in order], fontsize=8)
    ax.set_ylabel("median, standardised units (log scale)")
    ax.set_title("Intrinsic uncertainty against disagreement, per task — "
                 "the two do not order tasks the same way")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); _save(fig, out, "sigma_by_task")


def fig_loso(r: Run, out: Path):
    folds = r.loso
    subs = list(folds["best_epochs"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    ax = axes[0]
    ep = [folds["best_epochs"][s] for s in subs]
    ax.bar(np.arange(len(subs)), ep, color=C_ID)
    ax.axhline(float(np.mean(ep)), color="k", ls="--", lw=1.4,
               label=f"mean {np.mean(ep):.1f} -> {folds['n_epochs']} used")
    ax.set_xticks(np.arange(len(subs))); ax.set_xticklabels(subs, rotation=60, fontsize=8)
    ax.set_ylabel("best epoch"); ax.set_title("Stage 1: epoch chosen per fold", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    nll = [folds["val_nll"][s] for s in subs]
    ax.bar(np.arange(len(subs)), nll, color=["#C44E52" if v > 0 else C_ID for v in nll])
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(np.arange(len(subs))); ax.set_xticklabels(subs, rotation=60, fontsize=8)
    ax.set_ylabel("held-out NLL")
    ax.set_title("Stage 1: held-out likelihood\n(negative is better than a unit Gaussian)",
                 fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    h = pd.DataFrame(r.report["stage2_history"])
    ax.plot(h["epoch"], h["train_mse"], "o-", ms=4, color=C_ID, label="train MSE of the mean")
    warm = h["phase"] == "warmup"
    if warm.any():
        ax.axvspan(-0.5, h.loc[warm, "epoch"].max() + 0.5, color="#999999", alpha=0.15,
                   label="warm-up")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked MSE"); ax.set_yscale("log")
    ax.set_title(f"Stage 2: {len(h)} epochs on all 12 participants", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); _save(fig, out, "loso_and_training")


# ---------------------------------------------------------------- summary

def summary(r: Run, quad: pd.DataFrame) -> str:
    e = r.m("epistemic")
    cal = r.eval["calibration"]
    L = [f"# Probabilistic ensemble — results ({r.split} split)", "",
         "Experiment 5's target and trunk, trained under Gaussian negative log-likelihood so "
         "the uncertainty splits into an aleatoric and an epistemic part.", "",
         "## Which component detects", "",
         "| score | AUROC | AUROC (unfiltered) | J | best J | recall | specificity |",
         "|---|---|---|---|---|---|---|"]
    for n in SCORES:
        m = r.m(n)
        L.append(f"| {n} | **{m['auroc']:.3f}** | {m['auroc_raw']:.3f} | "
                 f"{m['j_statistic']:.2f} | {m['best_j']:.2f} | {m['recall']:.2f} | "
                 f"{m['specificity']:.2f} |")
    L += ["",
          f"Experiment 5, the same target scored by branch variance alone, reached AUROC "
          f"{EXP5_AUROC:.3f} and J {EXP5_J:.2f}. The epistemic component here is that same "
          f"quantity by construction, so the difference of "
          f"{e['auroc'] - EXP5_AUROC:+.3f} AUROC is a verification rather than a comparison.",
          "",
          "## Calibration", "",
          f"- Held-out `Var[z]` averaged over the twelve LOSO folds: "
          f"**{np.mean([v for v in r.loso['val_z_var'].values() if v is not None]):.3f}** "
          f"(target 1.000).",
          f"- On {r.split} in-distribution windows: median squared z-score "
          f"**{cal['z2_median']:.3f}**, mean {cal['z2_mean']:.3f}, 90th percentile "
          f"{cal['z2_p90']:.3f}, 99th {cal['z2_p99']:.3f}.",
          f"- The mean is not the number to read: the worst 0.1% of windows carry "
          f"**{100*cal['z2_share_top_0.1pct']:.1f}%** of the total. The median says the "
          f"variance head is calibrated for typical windows; the mean says it has a heavy "
          f"tail of confident errors. Both are true.",
          f"- Mean predicted sigma {cal['mean_sigma']:.4f}; realised RMSE "
          f"{np.sqrt(cal['mse']):.4f}.",
          f"- Log-variance on a clamp bound for {100*cal['frac_clamped']:.2f}% of entries.",
          "",
          "## Quadrant occupancy", "",
          "| task | kind | n | low both | high alea only | high epi only | high both |",
          "|---|---|---|---|---|---|---|"]
    for _, q in quad.iterrows():
        L.append(f"| {MODE_NAMES[q['mode']]} ({q['mode']}) | {q['kind']} | {q['n']:,} | "
                 f"{q['lo_lo']:.1f}% | {q['hi_lo']:.1f}% | {q['lo_hi']:.1f}% | "
                 f"{q['hi_hi']:.1f}% |")
    L += ["", "## Training", "",
          f"- Stage 1: {r.loso['n_folds']}/{r.loso['expected_folds']} folds, best epochs "
          f"{list(r.loso['best_epochs'].values())}, mean "
          f"{r.loso['mean_best_epoch']:.1f} (sd {r.loso['sd_best_epoch']:.1f}) -> "
          f"{r.loso['n_epochs']} used.",
          f"- Stage 2: {len(r.report['stage2_history'])} epochs on all twelve participants, "
          f"{r.report['config']['warmup_epochs']} of them squared-error warm-up.",
          f"- beta = {r.report['config']['beta']}, learning rate "
          f"{r.report['config']['lr']}, batch {r.report['config']['batch_size']}.",
          f"- Held-out validation NLL {r.report['val']['nll']:+.5f}, MSE "
          f"{r.report['val']['mse']:.5f}, Var[z] {r.report['val']['z_var']:.3f}.",
          "", "## Figures", "",
          "- `scores_by_task` — all four candidates per task",
          "- `roc_compare` — every score on one ROC, against Experiments 1 and 5",
          "- `detection_by_task` — per-task detection at the calibrated threshold",
          "- `calibration` — Var[z] per fold and through training (target exactly 1)",
          "- `alea_epi_plane` — the two-dimensional space and quadrant occupancy",
          "- `sigma_by_task` — intrinsic uncertainty against disagreement",
          "- `loso_and_training` — fold spread and the stage-2 curve"]
    return "\n".join(L)


def run_all(out: str = "ml/prob_ensemble/runs/prob_forecast", split: str = "test") -> int:
    run_dir = resolve_out(out)
    need = [f"scores_{split}.npz", f"eval_{split}.json", "report.json", "loso_summary.json"]
    missing = [n for n in need if not (run_dir / n).exists()]
    if missing:
        raise SystemExit(f"missing {missing} in {run_dir} -- run train.py and evaluate.py first")

    r = Run(run_dir, split)
    figs = run_dir / "figures"
    print(f"writing figures to {figs}")
    fig_scores_by_task(r, figs)
    fig_roc(r, figs)
    fig_detection_by_task(r, figs)
    fig_calibration(r, figs)
    quad = fig_plane(r, figs)
    fig_sigma_by_task(r, figs)
    fig_loso(r, figs)

    text = summary(r, quad)
    (run_dir / "summary.md").write_text(text)
    quad.to_csv(run_dir / f"quadrants_{split}.csv", index=False)
    print(f"\nwrote {run_dir/'summary.md'}\n")
    print(text)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figures for the probabilistic ensemble.")
    p.add_argument("--out", default="ml/prob_ensemble/runs/prob_forecast")
    p.add_argument("--split", default="test", choices=("val", "test"))
    a = p.parse_args()
    raise SystemExit(run_all(a.out, a.split))
