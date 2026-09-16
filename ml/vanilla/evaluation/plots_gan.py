"""Figures for the TCN GAN — the paper's reporting, plus the adversarial-health diagnostics.

    python ml/vanilla/evaluation/plots_gan.py                      # uses runs/gan
    python ml/vanilla/evaluation/plots_gan.py --out <run> --split test --reuse

Reproduces the ankle paper's reporting (Table I row, Fig. 4 per-task distributions, ROC) and
adds the two things this architecture specifically needs looked at:

* **Adversarial health over training.** ``D(real)`` and ``D(fake)`` against epoch, with the
  band where the score still discriminates marked. This is the plot that says whether the run
  produced a detector or a won argument — a GAN whose losses look textbook-perfect can have a
  dead ``Psi``, and no other figure would reveal it.
* **Generated windows against real ones.** The generator is not the deliverable, but it is the
  discriminator's only training signal, so its output quality bounds what the discriminator
  could have learned. If the generated windows are noise, ``D`` never had to learn anything
  about gait to win.

Unlike the autoencoder there is no latent to project and no second candidate score, so the
latent-space and score-comparison figures have no analogue here.

Writes PNG + PDF to ``<run>/figures/`` and a Markdown summary to ``<run>/summary.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from dataset import GanData
from evaluation.gan import GanScores, degeneracy_report, evaluate_gan, load_gan_run, per_mode_table
from training.common import pick_device, resolve_out

C_ID, C_OOD = "#4C72B0", "#DD8452"
MODE_NAMES = {"LG": "level ground", "RA": "ramp ascent", "RD": "ramp descent",
              "SA": "stair ascent", "SD": "stair descent", "ST": "standing",
              "TR": "transitions"}
MODE_COLS = {"LG": "#4C72B0", "RA": "#6FA8DC", "RD": "#9FC5E8",
             "SA": "#C44E52", "SD": "#DD8452", "ST": "#55A868", "TR": "#8172B3"}
ORDER = ("LG", "RA", "RD", "SA", "SD", "TR", "ST")
# Paper's GAN row, Table I. Its OOD set was genuinely non-cyclic and 80.1% of windows, so
# these are context rather than a like-for-like target.
PAPER_GAN = {"accuracy": 89.4, "precision": 74.9, "recall": 68.7,
             "f1": 71.5, "j_statistic": 63.0, "ece": 0.06, "brier": 0.09}


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote {name}.png / .pdf")


# ---------------------------------------------------------------- paper figures

def fig_psi_by_task(s: GanScores, out: Path):
    """Per-task uncertainty distributions with the threshold — the paper's Fig. 4.

    Plotted on a linear axis, unlike the autoencoder's: ``Psi = 1 - D(x)`` is bounded in
    [0, 1], so the absolute position of each distribution is directly meaningful and a log
    axis would hide how compressed the scores are when the discriminator saturates.
    """
    order = [m for m in ORDER if (s.meta["mode"] == m).any()]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, mo in enumerate(order):
        sel = (s.meta["mode"] == mo).to_numpy()
        parts = ax.violinplot([s.psi_smooth[sel]], positions=[i], orientation="horizontal",
                              widths=0.8, showextrema=False, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(C_OOD if s.labels[sel][0] else C_ID)
            body.set_alpha(0.65); body.set_edgecolor("none")
        parts["cmedians"].set_color("black")
        ax.text(s.threshold + 0.01, i + 0.33,
                f"{100*(s.psi_smooth[sel] > s.threshold).mean():.0f}% flagged", fontsize=8)
    ax.axvline(s.threshold, color="k", ls="--", lw=1.3,
               label=f"threshold (99.5th pct of ID train) = {s.threshold:.4f}")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{MODE_NAMES[m]}\n({m})" for m in order], fontsize=9)
    ax.set_xlabel(r"$\Psi = 1 - D(x)$   (causal-median filtered)")
    ax.set_title("GAN uncertainty by task — blue in-distribution, orange held-out")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25, axis="x")
    fig.tight_layout(); _save(fig, out, "gan_psi_by_task")


def fig_roc(s: GanScores, m: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5.6))
    fpr, tpr, _ = roc_curve(s.labels, s.psi_smooth)
    ax.plot(fpr, tpr, lw=2, color=C_ID, label=f"$\\Psi = 1 - D(x)$ (AUROC {m['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="chance")
    ax.plot(1 - m["specificity"]/100, m["recall"]/100, "o", ms=9, color=C_OOD, zorder=5,
            label=f"99.5th-pct threshold\n(J = {m['j_statistic']:.1f})")
    ax.set_xlabel("false positive rate  (ID wrongly flagged)")
    ax.set_ylabel("true positive rate  (OOD caught)")
    ax.set_title("GAN discriminator: ID vs held-out modes")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout(); _save(fig, out, "gan_roc")


def fig_detection_by_task(s: GanScores, out: Path):
    per = per_mode_table(s)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(per))
    cols = [C_OOD if k == "OOD" else C_ID for k in per["kind"]]
    ax.bar(x, per["flagged_%"], 0.62, color=cols)
    for i, r in per.iterrows():
        ax.text(i, r["flagged_%"] + 1.5, f"{r['flagged_%']:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODE_NAMES[mo]}\n({mo})\nn={n:,}"
                        for mo, n in zip(per["mode"], per["windows"])], fontsize=8)
    ax.set_ylabel("% of windows flagged as OOD"); ax.set_ylim(0, 115)
    ax.set_title("Detection rate by task — blue bars are in-distribution, where lower is better")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); _save(fig, out, "gan_detection_by_task")


# ---------------------------------------------------------------- adversarial health

def fig_training(run: Path, out: Path):
    """Losses, the D(real)/D(fake) balance, and the spread of Psi — three panels.

    The middle panel is the one to read first. Adversarial losses are famously uninformative
    about whether anything useful was learned: G and D are optimising against each other, so
    both curves can look stable while the equilibrium sits somewhere useless. What matters for
    a *detector* is whether D stayed uncertain. The shaded band marks D(real) in 0.4-0.7; a
    curve that leaves it upward means the discriminator has learned to recognise real hip data
    as such, at which point Psi is near zero for stair ascent as much as for level ground.
    """
    rep = json.loads((run / "report.json").read_text())
    h = pd.DataFrame(rep["history"])
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.4))

    axes[0].plot(h["epoch"], h["loss_g"], lw=1.4, color=C_ID, label="generator")
    axes[0].plot(h["epoch"], h["loss_d"], lw=1.4, color=C_OOD, label="discriminator")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("binary cross-entropy")
    axes[0].set_title(f"Adversarial losses, {rep['epochs_run']} epochs "
                      f"({rep['config']['g_steps_per_d']} G steps per D step)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.25)

    axes[1].axhspan(0.4, 0.7, color="#55A868", alpha=0.13,
                    label="D(real) band where $\\Psi$ still discriminates")
    axes[1].plot(h["epoch"], h["d_real"], lw=1.4, color="#333333", label="D(real)")
    axes[1].plot(h["epoch"], h["d_fake"], lw=1.4, color="#C44E52", label="D(fake)")
    axes[1].axhline(0.5, color="k", ls=":", lw=0.9)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("discriminator output")
    axes[1].set_title("Adversarial balance — the plot that says\nwhether a detector survived")
    axes[1].legend(fontsize=8, loc="best"); axes[1].grid(alpha=0.25)

    axes[2].plot(h["epoch"], h["psi_spread"], lw=1.4, color=C_ID, label="IQR of $\\Psi$")
    axes[2].plot(h["epoch"], h["psi_median"], lw=1.2, ls="--", color="#999999",
                 label="median $\\Psi$")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel(r"$\Psi$ on held-out ID windows")
    axes[2].set_yscale("log")
    axes[2].set_title("Score spread — a collapse toward zero\nmeans the score stopped ranking")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.25)

    fig.tight_layout(); _save(fig, out, "gan_training_curve")
    return h


def fig_generated(run: Path, data: GanData, out: Path, device=None, seed: int = 0,
                  n_show: int = 3):
    """Generated windows against real ones, per channel, plus a per-channel scale check.

    The generator is not the deliverable — ``Psi`` never calls it at scoring time. It matters
    because it was the discriminator's entire negative class: if its output is obviously
    unlike hip data, then separating real from generated required learning nothing about gait,
    and ``D`` will call every real window real. The right-hand panel is the cheap numerical
    version of that question — per-channel standard deviation, generated against real.
    """
    device = device or pick_device()
    model, _ = load_gan_run(run, data, device)
    rng = np.random.default_rng(seed)

    idx = rng.choice(len(data.test), size=512, replace=False)
    real = data.test.tensors(np.sort(idx)).numpy()
    with torch.no_grad():
        fake = model.generator(model.generator.noise(512, device)).cpu().numpy()

    show = [0, 1, 2, 3, 4, 5, 6, 7]     # left-leg block: accel xyz, gyro xyz, angle, velocity
    t = np.arange(data.window) / data.config.get("sample_rate_hz", 200)
    fig = plt.figure(figsize=(16, 1.35 * len(show)))
    gs = fig.add_gridspec(len(show), 3, width_ratios=[1, 1, 0.85], wspace=0.28)

    for col, (arr, title, colour) in enumerate(
            ((real, "real windows", "#333333"), (fake, "generated windows", "#C44E52"))):
        for row, ch in enumerate(show):
            ax = fig.add_subplot(gs[row, col])
            for k in range(n_show):
                ax.plot(t, arr[k, ch], lw=1.0, alpha=0.8, color=colour)
            ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
            ax.set_ylim(-4, 4)
            if col == 0:
                ax.set_ylabel(data.channels[ch], fontsize=6.5, rotation=0, ha="right",
                              va="center")
            if row == 0:
                ax.set_title(f"{title} ({n_show} overlaid)", fontsize=9)
            if row == len(show) - 1:
                ax.set_xlabel("time (s)", fontsize=8)
            else:
                ax.set_xticklabels([])

    ax = fig.add_subplot(gs[:, 2])
    y = np.arange(data.n_channels)
    ax.barh(y - 0.2, real.std(axis=(0, 2)), 0.38, color="#333333", label="real")
    ax.barh(y + 0.2, fake.std(axis=(0, 2)), 0.38, color="#C44E52", label="generated")
    ax.axvline(1.0, color="k", ls=":", lw=1, label="standardized unit")
    ax.set_yticks(y); ax.set_yticklabels(data.channels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("standard deviation (standardized units)")
    ax.set_title("Per-channel scale:\ndoes the generator match the data's spread?", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="x")

    fig.suptitle("Generator output against real hip windows, left-leg channels "
                 "(standardized units)", y=1.005)
    _save(fig, out, "gan_generated")
    return {"real_std": real.std(axis=(0, 2)).tolist(),
            "fake_std": fake.std(axis=(0, 2)).tolist()}


def fig_score_distribution(s: GanScores, out: Path):
    """Where Psi actually sits, and how much of its range it uses.

    Two panels, because the two failure modes look different. Left: ID against OOD, which is
    the question the metrics answer. Right: the cumulative distribution over all windows,
    which shows *saturation* — a curve that jumps from 0 to 1 within a hair of ``Psi = 0``
    means the discriminator returns essentially one value and the ranking is noise, regardless
    of how the left panel reads.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    for lab, col, nm in ((0, C_ID, "ID"), (1, C_OOD, "held-out (OOD)")):
        v = s.psi_smooth[s.labels == lab]
        axes[0].hist(v, bins=90, density=True, alpha=0.6, color=col,
                     label=f"{nm} (n={int((s.labels==lab).sum()):,})")
    axes[0].axvline(s.threshold, color="k", ls="--", lw=1.1,
                    label=f"99.5th pct of ID train = {s.threshold:.4f}")
    axes[0].set_xlabel(r"$\Psi = 1 - D(x)$"); axes[0].set_ylabel("density")
    axes[0].set_title("Score distribution by label")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

    v = np.sort(s.psi_smooth)
    axes[1].plot(v, np.linspace(0, 1, len(v)), lw=1.6, color="#333333")
    axes[1].axvline(s.threshold, color="k", ls="--", lw=1.1, label="threshold")
    q1, q3 = np.percentile(s.psi_smooth, [25, 75])
    axes[1].axvspan(q1, q3, color=C_ID, alpha=0.15, label=f"IQR = {q3-q1:.4f}")
    axes[1].set_xlabel(r"$\Psi$"); axes[1].set_ylabel("cumulative fraction of windows")
    axes[1].set_title("Saturation check — a near-vertical curve at $\\Psi\\approx0$\n"
                      "means the discriminator has won and the score is dead")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

    fig.tight_layout(); _save(fig, out, "gan_score_distribution")


def fig_trial_scores(s: GanScores, out: Path, label_n: int = 22):
    """One point per trial — the traceable view, standing in for the AE's latent centroids.

    There is no latent space to project, so trials are placed by the two numbers that do exist
    for every recording: its median score and the fraction of its windows flagged. A trial in
    the top-right is one the detector rejected wholesale, and it is named, so it can be opened.
    """
    df = s.meta.copy()
    df["psi"] = s.psi_smooth
    df["flag"] = (s.psi_smooth > s.threshold).astype(float)
    g = (df.groupby(["subject", "trial", "mode", "kind"], observed=True)
         .agg(psi=("psi", "median"), flagged=("flag", "mean"), n=("psi", "size"))
         .reset_index())
    g["flagged"] *= 100

    fig, ax = plt.subplots(figsize=(11.5, 7))
    for mo in ORDER:
        sel = g["mode"] == mo
        if not sel.any():
            continue
        ax.scatter(g.loc[sel, "psi"], g.loc[sel, "flagged"],
                   s=np.clip(g.loc[sel, "n"] / 6, 18, 300), alpha=0.75,
                   c=MODE_COLS[mo], edgecolors="black", linewidths=0.4,
                   label=f"{mo} ({MODE_NAMES[mo]})")
    ax.axvline(s.threshold, color="k", ls="--", lw=1,
               label=f"threshold = {s.threshold:.4f}")
    for _, r in g.nlargest(label_n, "psi").iterrows():
        ax.annotate(f"{r['subject']}/{r['trial']}", (r["psi"], r["flagged"]),
                    fontsize=5.5, alpha=0.85, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(r"median $\Psi$ over the trial")
    ax.set_ylabel("% of the trial's windows flagged")
    ax.set_title("Per-trial scores — marker size is window count, labels are the\n"
                 f"{label_n} highest-scoring trials")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.25)
    fig.tight_layout(); _save(fig, out, "gan_trial_scores")
    return g


# ---------------------------------------------------------------- driver

def summary(s: GanScores, m: dict, run: Path, h: pd.DataFrame, trials: pd.DataFrame,
            scales: dict) -> str:
    rep = json.loads((run / "report.json").read_text())
    per = per_mode_table(s)
    health = m.get("degeneracy", degeneracy_report(s, verbose=False))
    last = h.iloc[-1]

    L = [f"# TCN GAN — results ({s.split} split)", "",
         "## Headline", "",
         "| metric | GAN | paper (GAN) | comparable? |", "|---|---|---|---|"]
    for k, lab in (("j_statistic", "J-statistic"), ("auroc", "AUROC"),
                   ("accuracy", "accuracy"), ("recall", "recall (OOD caught)"),
                   ("specificity", "specificity (ID kept)"), ("precision", "precision"),
                   ("f1", "F1")):
        ref = PAPER_GAN.get(k)
        comp = "yes" if k in ("j_statistic", "auroc") else "no"
        L.append(f"| {lab} | **{m[k]:.3f}** | {ref if ref is not None else '—'} | {comp} |")
    L += [f"| ECE | {m['ece']:.3f} | {PAPER_GAN['ece']} | partly |",
          f"| Brier | {m['brier']:.3f} | {PAPER_GAN['brier']} | partly |", "",
          f"Test set is **{m['pct_ood']:.1f}% OOD** against the paper's 80.1%, so accuracy, "
          "precision and F1 move with class balance and only J-statistic and AUROC compare "
          "directly.", "",
          "## Adversarial health", "",
          "The discriminator *is* the detector here, so this section decides whether the "
          "numbers above mean anything. Its training objective is separating real windows "
          "from generated ones, and every out-of-distribution window is still real hip data — "
          "a discriminator that wins outright scores every real window near zero and detects "
          "nothing.", "",
          f"- Final `D(real)` = **{last['d_real']:.3f}**, `D(fake)` = **{last['d_fake']:.3f}** "
          f"(a healthy run keeps these within roughly 0.4–0.7 of each other, not at 1 and 0).",
          f"- `Psi` on the {s.split} split: median {health['psi_median']:.5f}, "
          f"IQR {health['psi_iqr']:.5f}, range "
          f"[{health['psi_min']:.4f}, {health['psi_max']:.4f}].",
          f"- {100*health['fraction_near_zero']:.1f}% of windows score below 1e-3; "
          f"{health['distinct_values']:,} distinct filtered values.",
          f"- Verdict: **{'DEGENERATE — the score carries almost no information' if health['degenerate'] else 'the score retains usable spread'}**.",
          "",
          "## Per-task", "",
          "| task | kind | windows | flagged | median Psi |", "|---|---|---|---|---|"]
    for _, r in per.iterrows():
        L.append(f"| {MODE_NAMES[r['mode']]} ({r['mode']}) | {r['kind']} | {r['windows']:,} | "
                 f"{r['flagged_%']:.1f}% | {r['median_psi']:.5f} |")

    real_sd = np.mean(scales["real_std"]); fake_sd = np.mean(scales["fake_std"])
    L += ["", "## Training", "",
          f"- {rep['epochs_run']} epochs, **fixed** — Table IV specifies 500 with no early "
          "stopping, because a GAN has no validation signal that reliably identifies a best "
          "epoch.",
          f"- {rep['config']['g_steps_per_d']} generator steps per discriminator step "
          "(the paper's ratio, and deliberately the reverse of usual GAN practice — it is what "
          "keeps the discriminator weak enough to remain a novelty detector).",
          f"- Batch {rep['config']['batch_size']}, lr G {rep['config']['lr']:.1e} / "
          f"D {rep['config']['lr_d']:.1e}, exponential decay {rep['config']['lr_decay']} "
          "per epoch.",
          f"- Trained on {rep['n_train_windows']:,} windows — the Step-20 subsample "
          "(every second stored window). Evaluation scores every test window, unsubsampled, "
          "so the cross-model comparison rests on one identical test set.",
          f"- Validation participants: {', '.join(rep['val_subjects'])}.",
          f"- Threshold {s.threshold:.5f} (99.5th percentile of training Psi; "
          f"training median {rep['psi_train_median']:.5f}, IQR {rep['psi_train_iqr']:.5f}).",
          f"- Generator output scale: mean per-channel sd {fake_sd:.3f} against the real "
          f"data's {real_sd:.3f} (both in standardized units, so 1.0 is the target).",
          "", "## Figures", "",
          "- `gan_psi_by_task`, `gan_roc`, `gan_detection_by_task` — the paper's reporting",
          "- `gan_training_curve` — losses, the D(real)/D(fake) balance, and the spread of Psi",
          "- `gan_generated` — generator output against real windows, and per-channel scale",
          "- `gan_score_distribution` — ID vs OOD, and the saturation check",
          f"- `gan_trial_scores` — one labelled point per trial ({len(trials)} trials)"]
    return "\n".join(L)


def run(out: str = "ml/vanilla/runs/gan", split: str = "test", batch_size: int = 256,
        device=None, reuse: bool = False) -> int:
    """Generate every figure. Callable directly, so main.py needs no argv juggling."""
    run_dir = resolve_out(out)
    if not (run_dir / "final.pt").exists():
        raise SystemExit(f"no trained model at {run_dir/'final.pt'} — run "
                         f"`main.py train --model gan` first.")

    data = GanData(batch_size=batch_size)
    scores_path = run_dir / f"scores_{split}"
    if reuse and scores_path.with_suffix(".npz").exists():
        print(f"reusing {scores_path.name}.npz")
        s = GanScores.load(scores_path, split)
        payload = json.loads((run_dir / f"eval_{split}.json").read_text())
        m = payload.get("metrics", payload)
        m.setdefault("degeneracy", degeneracy_report(s, verbose=False))
    else:
        s, m = evaluate_gan(run_dir, data, split, pick_device(device), batch_size)
        s.save(scores_path)

    figs = run_dir / "figures"
    print(f"\nwriting figures to {figs}")
    fig_psi_by_task(s, figs)
    fig_roc(s, m, figs)
    fig_detection_by_task(s, figs)
    h = fig_training(run_dir, figs)
    fig_score_distribution(s, figs)
    trials = fig_trial_scores(s, figs)
    scales = fig_generated(run_dir, data, figs, pick_device(device))

    text = summary(s, m, run_dir, h, trials, scales)
    (run_dir / "summary.md").write_text(text)
    trials.to_csv(run_dir / f"trial_scores_{split}.csv", index=False)
    print(f"\nwrote {run_dir/'summary.md'} and trial_scores_{split}.csv\n")
    print(text)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="GAN figures and summary.")
    p.add_argument("--out", default="ml/vanilla/runs/gan")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--reuse", action="store_true",
                   help="load the saved scores instead of re-scoring")
    a = p.parse_args()
    return run(a.out, a.split, a.batch_size, a.device, a.reuse)


if __name__ == "__main__":
    raise SystemExit(main())
