"""Figures for the convolutional autoencoder — the paper's plots, plus latent analysis.

    python ml/vanilla/figures_ae.py                        # uses runs/autoencoder
    python ml/vanilla/figures_ae.py --out <run> --split test

Reproduces the ankle paper's reporting (Table I row, Fig. 4 per-task distributions, ROC) and
adds what the paper does not show but this project needs:

* **The latent space**, projected to 2-D and coloured by ambulation mode, by participant, and
  by score — so it can be seen whether the encoder organised the data by task at all.
* **Per-trial centroids**, each labelled with its trial ID. This is the traceable view: one
  point per recording rather than per window, so a cluster can be named.
* **Reconstruction examples**, showing which channels the autoencoder learned and which it gave
  up on.
* **LOF against reconstruction error**, reproducing the paper's decision to reject the latter.

Writes PNG + PDF to ``<run>/figures/`` and a Markdown summary to ``<run>/summary.md``.
Every figure is regenerated from the saved latent bundle, so re-running is cheap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

from dataset import AutoencoderData
from evaluation.autoencoder import LatentBundle, evaluate_autoencoder, load_ae_run, per_mode_table, trials_near
from training.common import pick_device, resolve_out
from training.autoencoder import encode_split

C_ID, C_OOD = "#4C72B0", "#DD8452"
MODE_NAMES = {"LG": "level ground", "RA": "ramp ascent", "RD": "ramp descent",
              "SA": "stair ascent", "SD": "stair descent", "ST": "standing",
              "TR": "transitions"}
MODE_COLS = {"LG": "#4C72B0", "RA": "#6FA8DC", "RD": "#9FC5E8",
             "SA": "#C44E52", "SD": "#DD8452", "ST": "#55A868", "TR": "#8172B3"}
ORDER = ("LG", "RA", "RD", "SA", "SD", "TR", "ST")
# Paper's autoencoder row, Table I. Its OOD set was genuinely non-cyclic and 80.1% of windows,
# so these are context rather than a like-for-like target.
PAPER_AE = {"accuracy": 69.8, "precision": 40.0, "recall": 94.7,
            "f1": 55.7, "j_statistic": 58.5, "ece": 0.26, "brier": 0.18}


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote {name}.png / .pdf")


# ---------------------------------------------------------------- paper figures

def fig_psi_by_task(b: LatentBundle, out: Path):
    """Per-task uncertainty distributions with the threshold — the paper's Fig. 4."""
    order = [m for m in ORDER if (b.meta["mode"] == m).any()]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, mo in enumerate(order):
        sel = (b.meta["mode"] == mo).to_numpy()
        s = np.clip(b.psi_smooth[sel], 1e-6, None)
        parts = ax.violinplot([np.log10(s)], positions=[i], orientation="horizontal",
                              widths=0.8, showextrema=False, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(C_OOD if b.labels[sel][0] else C_ID)
            body.set_alpha(0.65); body.set_edgecolor("none")
        parts["cmedians"].set_color("black")
        ax.text(np.log10(b.threshold) + 0.04, i + 0.33,
                f"{100*(b.psi_smooth[sel] > b.threshold).mean():.0f}% flagged", fontsize=8)
    ax.axvline(np.log10(b.threshold), color="k", ls="--", lw=1.3,
               label=f"threshold (99.5th pct of ID train) = {b.threshold:.3f}")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{MODE_NAMES[m]}\n({m})" for m in order], fontsize=9)
    ax.set_xlabel(r"$\log_{10}\ \Psi$   (LOF on the latent space, causal-median filtered)")
    ax.set_title("Autoencoder uncertainty by task — blue in-distribution, orange held-out")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25, axis="x")
    fig.tight_layout(); _save(fig, out, "ae_psi_by_task")


def fig_roc(b: LatentBundle, m: dict, out: Path):
    """ROC for both candidate scores, so the paper's rejection of one is visible."""
    fig, ax = plt.subplots(figsize=(6, 5.6))
    for score, lab, col in ((b.psi_smooth, "LOF on latent space", C_ID),
                            (b.recon_smooth, "reconstruction error", "#999999")):
        fpr, tpr, _ = roc_curve(b.labels, score)
        auc = m["auroc"] if lab.startswith("LOF") else m["recon"]["auroc"]
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{lab} (AUROC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="chance")
    ax.plot(1 - m["specificity"]/100, m["recall"]/100, "o", ms=9, color=C_OOD, zorder=5,
            label=f"99.5th-pct threshold\n(J = {m['j_statistic']:.1f})")
    ax.set_xlabel("false positive rate  (ID wrongly flagged)")
    ax.set_ylabel("true positive rate  (OOD caught)")
    ax.set_title("Autoencoder: ID vs held-out modes")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout(); _save(fig, out, "ae_roc")


def fig_detection_by_task(b: LatentBundle, out: Path):
    per = per_mode_table(b)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(per)); w = 0.38
    ax.bar(x - w/2, per["flagged_LOF_%"], w, label="LOF on latent space", color=C_ID)
    ax.bar(x + w/2, per["flagged_recon_%"], w, label="reconstruction error", color="#999999")
    for i, r in per.iterrows():
        ax.text(i - w/2, r["flagged_LOF_%"] + 1.5, f"{r['flagged_LOF_%']:.0f}", ha="center", fontsize=8)
        ax.text(i + w/2, r["flagged_recon_%"] + 1.5, f"{r['flagged_recon_%']:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODE_NAMES[m]}\n({m})\nn={n:,}"
                        for m, n in zip(per["mode"], per["windows"])], fontsize=8)
    ax.set_ylabel("% of windows flagged as OOD"); ax.set_ylim(0, 115)
    ax.set_title("Detection rate by task — for in-distribution modes lower is better")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); _save(fig, out, "ae_detection_by_task")


def fig_training(run: Path, out: Path):
    rep = json.loads((run / "report.json").read_text())
    h = pd.DataFrame(rep["history"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(h["epoch"], h["train_loss"], "o-", ms=4, color=C_ID, label="train")
    ax.plot(h["epoch"], h["val_loss"], "s-", ms=4, color=C_OOD, label="validation (held-out participants)")
    ax.axvline(rep["best_epoch"], color="k", ls="--", lw=1,
               label=f"best epoch {rep['best_epoch']}")
    ax.set_xlabel("epoch"); ax.set_ylabel("reconstruction MSE"); ax.set_yscale("log")
    ax.set_title(f"Single-stage training, patience {rep['config']['patience']}, "
                 f"lr {rep['config']['lr']:.4f}")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); _save(fig, out, "ae_training_curve")


# ---------------------------------------------------------------- latent analysis

def fig_latent_space(b: LatentBundle, out: Path, n_plot: int = 18_000, seed: int = 0):
    """The latent space in 2-D, three ways: by task, by participant, by score.

    PCA is fitted on **in-distribution latents only**, then everything is projected into that
    basis — the same discipline as the autoencoder's own training, and the only way to see
    whether held-out tasks fall outside the manifold the model learned.
    """
    rng = np.random.default_rng(seed)
    id_mask = b.labels == 0
    pca = PCA(n_components=3).fit(b.Z[id_mask])
    P = pca.transform(b.Z)
    evr = pca.explained_variance_ratio_

    sub = rng.choice(len(P), size=min(n_plot, len(P)), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # by mode
    for mo in ORDER:
        sel = sub[(b.meta["mode"].to_numpy()[sub] == mo)]
        if not len(sel):
            continue
        axes[0].scatter(P[sel, 0], P[sel, 1], s=2.5, alpha=0.16, c=MODE_COLS[mo],
                        edgecolors="none", rasterized=True, label=f"{mo} ({MODE_NAMES[mo]})")
    lg = axes[0].legend(fontsize=7, markerscale=6, loc="best", framealpha=0.9)
    for h in lg.legend_handles:
        h.set_alpha(1)
    axes[0].set_title("coloured by ambulation mode")

    # by participant
    subs = sorted(b.meta["subject"].unique())
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(subs):
        sel = sub[(b.meta["subject"].to_numpy()[sub] == s)]
        if not len(sel):
            continue
        axes[1].scatter(P[sel, 0], P[sel, 1], s=2.5, alpha=0.16, color=cmap(i % 20),
                        edgecolors="none", rasterized=True, label=s)
    lg = axes[1].legend(fontsize=7, markerscale=6, ncol=2, loc="best", framealpha=0.9)
    for h in lg.legend_handles:
        h.set_alpha(1)
    axes[1].set_title("coloured by participant")

    # by score
    sc = axes[2].scatter(P[sub, 0], P[sub, 1], s=2.5, c=np.log10(np.clip(b.psi_smooth[sub], 1e-6, None)),
                         cmap="magma", alpha=0.5, edgecolors="none", rasterized=True)
    plt.colorbar(sc, ax=axes[2], label=r"$\log_{10}\Psi$ (LOF)")
    axes[2].set_title("coloured by uncertainty score")

    for ax in axes:
        ax.set_xlabel(f"PC1 ({evr[0]:.1%})"); ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Autoencoder latent space ({b.Z.shape[1]} dims) projected to 2-D — "
                 f"PCA fitted on in-distribution latents only "
                 f"({evr[:2].sum():.0%} of variance shown)", y=1.02)
    fig.tight_layout(); _save(fig, out, "ae_latent_space")
    return pca, P


def fig_latent_3d(b: LatentBundle, P: np.ndarray, evr, out: Path, n_plot: int = 12_000, seed: int = 1):
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(P), size=min(n_plot, len(P)), replace=False)
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    for mo in ORDER:
        sel = sub[(b.meta["mode"].to_numpy()[sub] == mo)]
        if not len(sel):
            continue
        ax.scatter(P[sel, 0], P[sel, 1], P[sel, 2], s=2, alpha=0.15, c=MODE_COLS[mo],
                   edgecolors="none", rasterized=True, label=f"{mo}")
    lg = ax.legend(fontsize=8, markerscale=8, loc="upper left")
    for h in lg.legend_handles:
        h.set_alpha(1)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})"); ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_zlabel(f"PC3 ({evr[2]:.1%})")
    ax.set_title("Latent space, first three components")
    ax.view_init(elev=18, azim=45)
    fig.tight_layout(); _save(fig, out, "ae_latent_3d")


def fig_trial_centroids(b: LatentBundle, P: np.ndarray, evr, out: Path, label_n: int = 26):
    """One point per trial, labelled — the traceable view of the latent space.

    Per-window scatter shows structure but cannot be read back to a recording. Collapsing each
    trial to its latent centroid gives a plot where every point is a named trial, and the
    labelled outliers are the ones worth opening.
    """
    df = b.meta.copy()
    df["pc1"], df["pc2"] = P[:, 0], P[:, 1]
    df["psi"] = b.psi_smooth
    g = (df.groupby(["subject", "trial", "mode", "kind"], observed=True)
         .agg(pc1=("pc1", "mean"), pc2=("pc2", "mean"),
              psi=("psi", "median"), n=("pc1", "size")).reset_index())

    fig, ax = plt.subplots(figsize=(12, 8))
    for mo in ORDER:
        sel = g["mode"] == mo
        if not sel.any():
            continue
        ax.scatter(g.loc[sel, "pc1"], g.loc[sel, "pc2"],
                   s=np.clip(g.loc[sel, "n"] / 6, 18, 320), alpha=0.75,
                   c=MODE_COLS[mo], edgecolors="black", linewidths=0.4,
                   label=f"{mo} ({MODE_NAMES[mo]})")

    # label the trials furthest from the in-distribution centroid -- the interesting ones
    id_c = g.loc[g["kind"] == "ID", ["pc1", "pc2"]].mean().to_numpy()
    g["dist"] = np.linalg.norm(g[["pc1", "pc2"]].to_numpy() - id_c[None, :], axis=1)
    for _, r in g.nlargest(label_n, "dist").iterrows():
        ax.annotate(f"{r['subject']}/{r['trial']}", (r["pc1"], r["pc2"]),
                    fontsize=5.5, alpha=0.85,
                    xytext=(4, 3), textcoords="offset points")
    ax.scatter(*id_c, marker="X", s=200, c="black", zorder=6, label="ID centroid")

    ax.set_xlabel(f"PC1 ({evr[0]:.1%})"); ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("Latent centroid per trial — marker size is window count, labels are the\n"
                 f"{label_n} trials furthest from the in-distribution centre")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.25)
    fig.tight_layout(); _save(fig, out, "ae_trial_centroids")
    return g


def fig_reconstructions(run: Path, b: LatentBundle, data: AutoencoderData, out: Path,
                        device=None, seed: int = 0):
    """What the autoencoder actually learned, per channel, for an ID and an OOD window."""
    device = device or pick_device()
    model, _, _ = load_ae_run(run, data, device)
    rng = np.random.default_rng(seed)

    picks = []
    for kind, split_obj in (("ID", data.splits[b.split]), ("OOD", data.ood[b.split])):
        i = int(rng.integers(len(split_obj)))
        x = split_obj.tensors([i]).to(device)
        with torch.no_grad():
            x_hat, _ = model(x)
        picks.append((kind, split_obj.meta.iloc[i], x[0].cpu().numpy(), x_hat[0].cpu().numpy()))

    show = [0, 1, 2, 3, 4, 5, 6, 7]     # left-leg block: accel xyz, gyro xyz, angle, velocity
    names = data.channels
    fig, axes = plt.subplots(len(show), 2, figsize=(13, 1.35 * len(show)), sharex=True)
    t = np.arange(data.window) / data.config.get("sample_rate_hz", 200)
    for col, (kind, meta_row, x, x_hat) in enumerate(picks):
        err = float(((x - x_hat) ** 2).mean())
        for row, ch in enumerate(show):
            ax = axes[row, col]
            ax.plot(t, x[ch], lw=1.2, color="#333333", label="input")
            ax.plot(t, x_hat[ch], lw=1.1, ls="--", color="#C44E52", label="reconstruction")
            ax.set_ylabel(names[ch], fontsize=6.5, rotation=0, ha="right", va="center")
            ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(f"{kind}: {meta_row['subject']} / {meta_row['trial']}\n"
                             f"MSE {err:.4f}", fontsize=9)
                ax.legend(fontsize=7, loc="upper right")
        axes[-1, col].set_xlabel("time (s)")
    fig.suptitle("Reconstruction, left-leg channels (standardized units)", y=1.005)
    fig.tight_layout(); _save(fig, out, "ae_reconstructions")


def fig_score_comparison(b: LatentBundle, out: Path):
    """LOF against reconstruction error — the paper's rejected score, tested not assumed."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, (v, name, thr) in zip(
            axes[:2],
            [(b.psi_smooth, "LOF on latent space", b.threshold),
             (b.recon_smooth, "reconstruction error",
              float(np.percentile(b.recon_smooth[b.labels == 0], 99.5)))]):
        for lab, col, nm in ((0, C_ID, "ID"), (1, C_OOD, "OOD")):
            s = np.clip(v[b.labels == lab], 1e-6, None)
            ax.hist(np.log10(s), bins=80, density=True, alpha=0.6, color=col,
                    label=f"{nm} (n={int((b.labels==lab).sum()):,})")
        ax.axvline(np.log10(thr), color="k", ls="--", lw=1.1, label="99.5th pct of ID")
        ax.set_xlabel(rf"$\log_{{10}}$ {name}"); ax.set_ylabel("density")
        ax.set_title(name); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # standing is the paper's stated reason for rejecting reconstruction error
    st = (b.meta["mode"] == "ST").to_numpy()
    if st.any():
        rows = []
        for mo in ORDER:
            sel = (b.meta["mode"] == mo).to_numpy()
            if sel.any():
                rows.append({"mode": mo,
                             "LOF": np.median(b.psi_smooth[sel]),
                             "recon": np.median(b.recon_smooth[sel]),
                             "ood": bool(b.labels[sel][0])})
        d = pd.DataFrame(rows)
        x = np.arange(len(d)); w = 0.38
        ax = axes[2]
        ax.bar(x - w/2, d["LOF"] / d.loc[~d["ood"], "LOF"].median(), w,
               label="LOF (relative to ID median)", color=C_ID)
        ax.bar(x + w/2, d["recon"] / d.loc[~d["ood"], "recon"].median(), w,
               label="recon error (relative to ID median)", color="#999999")
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(d["mode"], fontsize=9)
        ax.set_ylabel("score / ID median"); ax.set_yscale("log")
        ax.set_title("Standing is the test case:\nit reconstructs easily but should be anomalous")
        ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); _save(fig, out, "ae_score_comparison")


def fig_latent_dims(b: LatentBundle, out: Path):
    """Which latent dimensions carry the ID/OOD difference, in the (filter x time) layout."""
    id_m, ood_m = b.labels == 0, b.labels == 1
    mu_id, mu_ood = b.Z[id_m].mean(0), b.Z[ood_m].mean(0)
    sd_id = b.Z[id_m].std(0) + 1e-9
    effect = (mu_ood - mu_id) / sd_id

    # the latent is stored flattened; recover the (filters, time) layout it came from
    from models.autoencoder import LATENT_FILTERS
    n_filters = LATENT_FILTERS
    n_time = b.Z.shape[1] // n_filters
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    im = axes[0].imshow(effect.reshape(n_filters, n_time), cmap="coolwarm",
                        vmin=-np.abs(effect).max(), vmax=np.abs(effect).max(), aspect="auto")
    plt.colorbar(im, ax=axes[0], label="(OOD mean − ID mean) / ID sd")
    axes[0].set_xlabel("latent timestep"); axes[0].set_ylabel("latent filter")
    axes[0].set_title(f"Where OOD differs in the latent, {n_filters}x{n_time} layout")

    axes[1].bar(np.arange(len(effect)), effect, color=np.where(effect > 0, C_OOD, C_ID))
    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].set_xlabel("latent dimension"); axes[1].set_ylabel("standardized difference")
    axes[1].set_title("Per-dimension effect size")
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout(); _save(fig, out, "ae_latent_dims")


# ---------------------------------------------------------------- driver

def summary(b: LatentBundle, m: dict, run: Path, centroids: pd.DataFrame) -> str:
    rep = json.loads((run / "report.json").read_text())
    per = per_mode_table(b)
    L = [f"# Convolutional autoencoder — results ({b.split} split)", "",
         "## Headline", "",
         "| metric | LOF on latent | reconstruction error | paper (AE) | comparable? |",
         "|---|---|---|---|---|"]
    for k, lab in (("j_statistic", "J-statistic"), ("auroc", "AUROC"), ("accuracy", "accuracy"),
                   ("recall", "recall (OOD caught)"), ("specificity", "specificity (ID kept)"),
                   ("precision", "precision"), ("f1", "F1")):
        ref = PAPER_AE.get(k)
        comp = "yes" if k in ("j_statistic", "auroc") else "no"
        L.append(f"| {lab} | **{m[k]:.3f}** | {m['recon'][k]:.3f} | "
                 f"{ref if ref is not None else '—'} | {comp} |")
    L += [f"| ECE | {m['ece']:.3f} | — | {PAPER_AE['ece']} | partly |",
          f"| Brier | {m['brier']:.3f} | — | {PAPER_AE['brier']} | partly |", "",
          f"Test set is **{m['pct_ood']:.1f}% OOD** against the paper's 80.1%, so accuracy, "
          "precision and F1 move with class balance and only J-statistic and AUROC compare "
          "directly.", "",
          "## Per-task", "",
          "| task | kind | windows | flagged (LOF) | flagged (recon) | median Psi |",
          "|---|---|---|---|---|---|"]
    for _, r in per.iterrows():
        L.append(f"| {MODE_NAMES[r['mode']]} ({r['mode']}) | {r['kind']} | {r['windows']:,} | "
                 f"{r['flagged_LOF_%']:.1f}% | {r['flagged_recon_%']:.1f}% | {r['median_psi']:.3f} |")
    L += ["", "## Training", "",
          f"- Single stage, {rep['epochs_run']} epochs run, best at epoch {rep['best_epoch']} "
          f"(patience {rep['config']['patience']}).",
          f"- Validation participants: {', '.join(rep['val_subjects'])}.",
          f"- Learning rate {rep['config']['lr']:.4f} (declared deviation; Table IV specifies 0.064, which collapses).",
          f"- Latent dimension {m['latent_dim']}.",
          f"- Threshold {b.threshold:.4f} (99.5th percentile of training Psi).", "",
          "## Latent space", "",
          f"- {len(centroids)} trials, {len(b.Z):,} windows projected.",
          "- `ae_latent_space` — 2-D projection by mode, participant and score.",
          "- `ae_trial_centroids` — one labelled point per trial (the traceable view).",
          "- `ae_latent_dims` — which latent dimensions separate ID from OOD.", "",
          "## Figures", "",
          "- `ae_psi_by_task`, `ae_roc`, `ae_detection_by_task` — the paper's reporting",
          "- `ae_training_curve` — single-stage training",
          "- `ae_reconstructions` — per-channel input vs reconstruction",
          "- `ae_score_comparison` — LOF against the paper's rejected reconstruction error"]
    return "\n".join(L)


def run(out: str = "ml/vanilla/runs/autoencoder", split: str = "test",
        batch_size: int = 1024, device=None, reuse: bool = False) -> int:
    """Generate every figure. Callable directly, so main.py needs no argv juggling."""
    run_dir = resolve_out(out)
    if not (run_dir / "final.pt").exists():
        raise SystemExit(f"no trained model at {run_dir/'final.pt'} — run train_ae.py first.")

    data = AutoencoderData(batch_size=batch_size)
    bundle_path = run_dir / f"latents_{split}"
    if reuse and bundle_path.with_suffix(".npz").exists():
        print(f"reusing {bundle_path.name}.npz")
        b = LatentBundle.load(bundle_path, split)
        m = json.loads((run_dir / f"eval_{split}.json").read_text())
        m["recon"] = m.pop("recon_score")
    else:
        b, m = evaluate_autoencoder(run_dir, data, split, pick_device(device), batch_size)
        b.save(bundle_path)

    figs = run_dir / "figures"
    print(f"\nwriting figures to {figs}")
    fig_psi_by_task(b, figs)
    fig_roc(b, m, figs)
    fig_detection_by_task(b, figs)
    fig_training(run_dir, figs)
    pca, P = fig_latent_space(b, figs)
    fig_latent_3d(b, P, pca.explained_variance_ratio_, figs)
    centroids = fig_trial_centroids(b, P, pca.explained_variance_ratio_, figs)
    fig_latent_dims(b, figs)
    fig_score_comparison(b, figs)
    fig_reconstructions(run_dir, b, data, figs, pick_device(device))

    text = summary(b, m, run_dir, centroids)
    (run_dir / "summary.md").write_text(text)
    centroids.to_csv(run_dir / f"trial_centroids_{split}.csv", index=False)
    print(f"\nwrote {run_dir/'summary.md'} and trial_centroids_{split}.csv\n")
    print(text)

    print("\nTo ask the latent space which trials occupy a region:")
    print("    from eval_ae import LatentBundle, trials_near")
    print(f"    b = LatentBundle.load('{bundle_path}', '{split}')")
    print("    # project with the same PCA, then:")
    print("    trials_near(b, P, centre=(x, y), radius=1.0)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Autoencoder figures and summary.")
    p.add_argument("--out", default="ml/vanilla/runs/autoencoder")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default=None)
    p.add_argument("--reuse", action="store_true",
                   help="load the saved latent bundle instead of re-scoring")
    a = p.parse_args()
    return run(a.out, a.split, a.batch_size, a.device, a.reuse)


if __name__ == "__main__":
    raise SystemExit(main())
