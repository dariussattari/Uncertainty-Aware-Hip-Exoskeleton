"""Summary statistics and figures for the vanilla gait-phase ensemble.

Mirrors the reporting in Tourk et al. (arXiv:2508.21221): the Table I metric row, the
per-task uncertainty distributions of their Fig. 4, and an ROC curve. Adds two diagnostics
the paper does not show but which a thesis chapter wants — the LOSO fold spread that produced
the epoch count, and the Stage 2 training curve.

    python ml/vanilla/figures.py                       # uses runs/paper
    python ml/vanilla/figures.py --out <run> --split test

Writes PNG + PDF into ``<run>/figures/`` and a Markdown summary table to
``<run>/summary.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from dataset import EnsembleGaitPhase
from eval import evaluate_model, load_checkpoint
from train import pick_device, resolve_out

C_ID, C_OOD = "#4C72B0", "#DD8452"
MODE_NAMES = {
    "LG": "level ground", "RA": "ramp ascent", "RD": "ramp descent",
    "SA": "stair ascent", "SD": "stair descent", "ST": "standing",
    "TR": "transitions",
}
# The paper's gait-phase ensemble, for reference. Its OOD set was genuinely non-cyclic
# (sitting, jumping, lying down) and 80.1% of the test windows, so these are context, not
# a like-for-like target.
PAPER_REFERENCE = {"accuracy": 96.1, "precision": 93.0, "recall": 96.2,
                   "f1": 90.3, "j_statistic": 92.3, "ece": 0.03, "brier": 0.03}


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote {name}.png / .pdf")


def collect(run: Path, split: str, data: EnsembleGaitPhase):
    """Re-score the split so figures carry per-window mode labels, not just aggregates."""
    model, threshold = load_checkpoint(run / "final.pt", data)
    res = evaluate_model(model, data, threshold, split=split,
                         device=pick_device(), verbose=False)
    # evaluate_model concatenates ID then OOD; rebuild the mode/subject vectors to match.
    id_meta, ood_meta = data.splits[split].meta, data.ood[split].meta
    modes = np.concatenate([id_meta["mode"].to_numpy(), ood_meta["mode"].to_numpy()])
    subjects = np.concatenate([id_meta["subject"].to_numpy(), ood_meta["subject"].to_numpy()])
    assert len(modes) == len(res.scores)
    return res, modes, subjects


def fig_psi_by_task(res, modes, out_dir: Path):
    """Per-task uncertainty distributions with the decision threshold — the paper's Fig. 4."""
    order = [m for m in ("LG", "RA", "RD", "SA", "SD", "TR", "ST") if (modes == m).any()]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, mo in enumerate(order):
        sel = modes == mo
        s = np.clip(res.scores[sel], 1e-7, None)
        is_ood = bool(res.labels[sel][0])
        color = C_OOD if is_ood else C_ID

        parts = ax.violinplot([np.log10(s)], positions=[i], orientation='horizontal',
                              widths=0.8, showextrema=False, showmedians=True)
        for b in parts["bodies"]:
            b.set_facecolor(color); b.set_alpha(0.65); b.set_edgecolor("none")
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.2)

        flagged = 100 * (res.scores[sel] > res.threshold).mean()
        ax.text(np.log10(res.threshold) + 0.12, i + 0.34, f"{flagged:.0f}% flagged",
                fontsize=8, color="black", va="center")

    ax.axvline(np.log10(res.threshold), color="k", ls="--", lw=1.3,
               label=f"threshold (99.5th pct of ID train) = {res.threshold:.2e}")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{MODE_NAMES.get(m, m)}\n({m})" for m in order], fontsize=9)
    ax.set_xlabel(r"$\log_{10}\ \Psi$   (ensemble variance, causal-median filtered)")
    ax.set_title("Uncertainty by task — blue in-distribution, orange pseudo-OOD")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    _save(fig, out_dir, "psi_by_task")
    return fig


def fig_roc(res, out_dir: Path):
    fpr, tpr, _ = roc_curve(res.labels, res.scores)
    op_fpr = 1 - res.metrics["specificity"] / 100
    op_tpr = res.metrics["recall"] / 100

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, lw=2, color=C_ID, label=f"AUROC = {res.metrics['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="chance")
    ax.plot(op_fpr, op_tpr, "o", ms=9, color=C_OOD, zorder=5,
            label=f"99.5th-pct threshold\n(J = {res.metrics['j_statistic']:.1f})")
    ax.set_xlabel("false positive rate  (ID wrongly flagged)")
    ax.set_ylabel("true positive rate  (OOD caught)")
    ax.set_title("ID vs pseudo-OOD detection")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save(fig, out_dir, "roc")
    return fig


def fig_detection_by_task(res, modes, out_dir: Path):
    rows = []
    for mo in ("LG", "RA", "RD", "SA", "SD", "TR", "ST"):
        sel = modes == mo
        if not sel.any():
            continue
        rows.append({"mode": mo, "ood": bool(res.labels[sel][0]),
                     "flagged": 100 * (res.scores[sel] > res.threshold).mean(),
                     "n": int(sel.sum())})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [C_OOD if o else C_ID for o in df["ood"]]
    bars = ax.bar(range(len(df)), df["flagged"], color=colors, alpha=0.85)
    for b, (_, r) in zip(bars, df.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{r['flagged']:.1f}%\nn={r['n']:,}", ha="center", fontsize=8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{MODE_NAMES.get(m, m)}\n({m})" for m in df["mode"]], fontsize=9)
    ax.set_ylabel("% of windows flagged as OOD")
    ax.set_ylim(0, 115)
    ax.axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_title("Detection rate by task — for blue bars, lower is better; for orange, higher")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    _save(fig, out_dir, "detection_by_task")
    return fig


def fig_loso(run: Path, out_dir: Path):
    folds = json.loads((run / "loso_folds.json").read_text())
    df = pd.DataFrame(folds)
    mean_ep = df["best_epoch"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(df["subject"], df["best_epoch"], color=C_ID, alpha=0.85)
    axes[0].axhline(mean_ep, color="k", ls="--", lw=1.2,
                    label=f"mean = {mean_ep:.1f} -> {round(mean_ep)} used for Stage 2")
    axes[0].axhline(13, color=C_OOD, ls=":", lw=1.4, label="paper's ankle result = 13")
    axes[0].set_ylabel("best epoch"); axes[0].set_title("Stage 1: LOSO early-stopping epoch")
    axes[0].tick_params(axis="x", rotation=60, labelsize=8)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(df["subject"], df["val_loss"], color=C_ID, alpha=0.85)
    axes[1].set_ylabel("held-out masked MSE"); axes[1].set_title("Stage 1: per-subject validation loss")
    axes[1].tick_params(axis="x", rotation=60, labelsize=8)
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    _save(fig, out_dir, "loso")
    return fig, df


def fig_training(run: Path, out_dir: Path):
    report = json.loads((run / "report.json").read_text())
    hist = pd.DataFrame(report["stage2_history"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["epoch"], hist["train_loss"], "o-", color=C_ID, ms=4, label="train (all 12 subjects)")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked MSE")
    ax.set_title(f"Stage 2: final model, {len(hist)} epochs from LOSO")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, out_dir, "training_curve")
    return fig


def summary(res, modes, run: Path, folds: pd.DataFrame, split: str) -> str:
    m = res.metrics
    report = json.loads((run / "report.json").read_text())

    lines = [
        f"# Vanilla gait-phase ensemble — results ({split} split)", "",
        "## Headline", "",
        "| metric | ours | paper (ankle) | comparable? |",
        "|---|---|---|---|",
        f"| J-statistic | **{m['j_statistic']:.1f}** | {PAPER_REFERENCE['j_statistic']:.1f} | yes |",
        f"| AUROC | **{m['auroc']:.3f}** | 0.993 | yes |",
        f"| accuracy | {m['accuracy']:.1f}% | {PAPER_REFERENCE['accuracy']:.1f}% | no |",
        f"| recall (OOD caught) | {m['recall']:.1f}% | {PAPER_REFERENCE['recall']:.1f}% | no |",
        f"| specificity (ID kept) | {m['specificity']:.1f}% | — | no |",
        f"| precision | {m['precision']:.1f}% | {PAPER_REFERENCE['precision']:.1f}% | no |",
        f"| F1 | {m['f1']:.1f}% | {PAPER_REFERENCE['f1']:.1f}% | no |",
        f"| ECE | {m['ece']:.3f} | {PAPER_REFERENCE['ece']:.2f} | partly |",
        f"| Brier | {m['brier']:.3f} | {PAPER_REFERENCE['brier']:.2f} | partly |",
        "",
        f"Test set is **{m['pct_ood']:.1f}% OOD** against the paper's **80.1%**. Accuracy, "
        "precision and F1 all move with class balance, so only J-statistic and AUROC compare "
        "directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to "
        "in-distribution than their sitting/jumping/lying-down, which makes this a strictly "
        "harder detection problem.", "",
        "## Per-task", "",
        "| task | kind | windows | flagged as OOD | median Psi |",
        "|---|---|---|---|---|",
    ]
    for mo in ("LG", "RA", "RD", "SA", "SD", "TR", "ST"):
        sel = modes == mo
        if not sel.any():
            continue
        kind = "OOD" if res.labels[sel][0] else "ID"
        lines.append(f"| {MODE_NAMES.get(mo, mo)} ({mo}) | {kind} | {sel.sum():,} | "
                     f"{100 * (res.scores[sel] > res.threshold).mean():.1f}% | "
                     f"{np.median(res.scores[sel]):.2e} |")

    lines += [
        "", "## Training", "",
        f"- Stage 1: LOSO over {len(folds)} subjects, best epoch "
        f"{folds['best_epoch'].min()}–{folds['best_epoch'].max()} "
        f"(mean {folds['best_epoch'].mean():.1f}) — the paper found 13 on ankle data.",
        f"- Stage 2: retrained on all {len(folds)} subjects for "
        f"{report['stage1']['n_epochs']} epochs.",
        f"- Threshold: {res.threshold:.3e} (99.5th percentile of training Psi).",
        f"- Held-out validation masked MSE: {report['val_loss']:.5f}.",
        "", "## Figures", "",
        "- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)",
        "- `roc` — ROC curve and the operating point the threshold selects",
        "- `detection_by_task` — flagged fraction per task",
        "- `loso` — Stage 1 fold spread",
        "- `training_curve` — Stage 2 loss",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Summary stats and figures.")
    p.add_argument("--out", default="ml/vanilla/runs/paper")
    p.add_argument("--split", default="test", choices=("val", "test"))
    args = p.parse_args()

    run = resolve_out(args.out)
    if not (run / "final.pt").exists():
        raise SystemExit(f"no trained model at {run / 'final.pt'}")

    data = EnsembleGaitPhase()
    print(f"scoring {args.split}...")
    res, modes, subjects = collect(run, args.split, data)

    fig_dir = run / "figures"
    print(f"\nwriting figures to {fig_dir}")
    fig_psi_by_task(res, modes, fig_dir)
    fig_roc(res, fig_dir)
    fig_detection_by_task(res, modes, fig_dir)
    _, folds = fig_loso(run, fig_dir)
    fig_training(run, fig_dir)

    text = summary(res, modes, run, folds, args.split)
    (run / "summary.md").write_text(text)
    print(f"\nwrote {run / 'summary.md'}\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
