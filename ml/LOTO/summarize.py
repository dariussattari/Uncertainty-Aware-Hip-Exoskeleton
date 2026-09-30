"""Collect every finished leave-one-task-out run into one table.

    python ml/LOTO/summarize.py                  # writes summary.md and summary.csv
    python ml/LOTO/summarize.py --split val

Discovery is by **walking the tree**, not by consulting a list: any `eval_<split>.json` found
under `ml/LOTO/<TASK>/<anything>/` is picked up. Adding a model to the matrix therefore needs
no change here — train it into a new leaf and it appears. The two evaluation schemas in the
project are both understood:

* the vanilla schema, one score per run, under ``metrics``
* the probabilistic schema, four candidate scores, under ``scores``

For the probabilistic runs the row reported is ``epistemic``, which is the score comparable to
the vanilla ensembles; the others are kept in the CSV with a ``score`` column so nothing is
thrown away.

The participant-split results are loaded alongside, from ``ml/vanilla/runs/`` and
``ml/prob_ensemble/runs/``, so every LOTO number sits next to the number it should be read
against. **Do not compare the F1 or precision columns across the two protocols**: the OOD share
of the test set is 27.1% on the participant split and 0.7-12% here, and those metrics move with
prevalence. AUROC and best-J do not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from splits import HOLDOUTS, held_out_modes  # noqa: E402
REPO = next(d for d in HERE.parents if (d / "data").is_dir())

# the participant-split counterpart of each LOTO leaf, for the side-by-side column
BASELINE = {
    "vanilla/AE": "ml/vanilla/runs/autoencoder",
    "vanilla/GAN": "ml/vanilla/runs/gan",
    "vanilla/Ensemble_GP": "ml/vanilla/runs/ensemble_gp",
    "vanilla/Ensemble_synthetic": "ml/vanilla/runs/synthetic",
    "vanilla/Ensemble_forecast": "ml/vanilla/runs/forecast_angle",
    "prob_ensemble": "ml/prob_ensemble/runs/prob_forecast",
}
COLS = ["auroc", "best_j", "j_statistic", "f1", "recall", "specificity", "precision",
        "ece", "brier", "n_id", "n_ood"]


def _rows_from(path: Path, **tag) -> list[dict]:
    """One row per score in an ``eval_<split>.json``, whichever schema it uses."""
    r = json.loads(path.read_text())
    common = dict(tag, model=r.get("model", "?"), split=r.get("split", "?"))

    if "scores" in r:                                   # probabilistic schema
        out = []
        for name, s in r["scores"].items():
            # this schema has no n_id/n_ood field; the confusion counts carry the same fact
            c = s.get("counts", {})
            out.append(common | {"score": name, "primary": name == "epistemic",
                                 "n_id": c.get("tn", 0) + c.get("fp", 0),
                                 "n_ood": c.get("tp", 0) + c.get("fn", 0)}
                       | {k: s[k] for k in COLS if k in s})
        return out

    m = r.get("metrics", {})                            # vanilla schema
    return [common | {"score": "psi", "primary": True} | {c: m.get(c) for c in COLS}]


def collect(split: str = "test") -> pd.DataFrame:
    rows: list[dict] = []
    for task_dir in sorted(p for p in HERE.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for f in sorted(task_dir.rglob(f"eval_{split}.json")):
            leaf = f.parent.relative_to(task_dir).as_posix()
            leaf = leaf[:-4] if leaf.endswith("/run") else leaf   # prob_ensemble/run -> ...
            rows += _rows_from(f, held_out=task_dir.name, leaf=leaf or ".")
    for leaf, rel in BASELINE.items():
        f = REPO / rel / f"eval_{split}.json"
        if f.exists():
            rows += _rows_from(f, held_out="participant-split", leaf=leaf)
    if not rows:
        return pd.DataFrame(columns=["held_out", "leaf", "score", *COLS])
    df = pd.DataFrame(rows)
    order = {l: i for i, l in enumerate(BASELINE)}
    df["_o"] = df["leaf"].map(order).fillna(99)
    return df.sort_values(["_o", "leaf", "held_out", "score"]).drop(columns="_o")


def to_markdown(df: pd.DataFrame, split: str) -> str:
    if df.empty:
        return f"# LOTO results ({split})\n\nNothing evaluated yet.\n"
    p = df[df["primary"]].copy()
    tasks = [t for t in HOLDOUTS if t in set(p["held_out"])]

    lines = [f"# Leave-one-task-out — {split} split", "",
             "AUROC, and best achievable J in brackets. `base` is the participant-split run "
             "(LG/RA/RD in training, four tasks held out at once).", "",
             "Precision and F1 are **not** comparable across the two protocols: the OOD share "
             "of the test set differs by more than an order of magnitude. AUROC and best-J are "
             "prevalence-independent.", "",
             "| model | base | " + " | ".join(f"−{t}" for t in tasks) + " |",
             "|---|---|" + "---|" * len(tasks)]

    def cell(sub):
        if sub.empty or pd.isna(sub["auroc"].iloc[0]):
            return "—"
        r = sub.iloc[0]
        j = r.get("best_j")
        return f"{r['auroc']:.3f}" + (f" [{j:.1f}]" if pd.notna(j) else "")

    for leaf in BASELINE:
        sub = p[p["leaf"] == leaf]
        if sub.empty:
            continue
        row = [leaf, cell(sub[sub["held_out"] == "participant-split"])]
        row += [cell(sub[sub["held_out"] == t]) for t in tasks]
        lines.append("| " + " | ".join(row) + " |")

    extra = sorted(set(p["leaf"]) - set(BASELINE))
    for leaf in extra:                          # models added after this file was written
        sub = p[p["leaf"] == leaf]
        lines.append("| " + " | ".join([leaf, "—", *(cell(sub[sub["held_out"] == t])
                                                     for t in tasks)]) + " |")

    lines += ["", "## Test-set composition", "",
              "| held out | ID windows | OOD windows | OOD share |", "|---|---|---|---|"]
    for t in ["participant-split", *tasks]:
        sub = p[p["held_out"] == t]
        if sub.empty:
            continue
        n_id, n_ood = int(sub["n_id"].max()), int(sub["n_ood"].max())
        lines.append(f"| {t} | {n_id:,} | {n_ood:,} | {100 * n_ood / (n_id + n_ood):.1f}% |")

    done = len(p[p["held_out"] != "participant-split"])
    lines += ["", f"_{done} leave-one-task-out runs evaluated._", ""]
    return "\n".join(lines)


def figure(df: pd.DataFrame, split: str, out: Path) -> Path | None:
    """Grouped bars: AUROC per model, one bar per held-out task plus the participant split.

    One figure is what the weekly update needs, and AUROC is the only headline number that
    survives the change in OOD prevalence between the two protocols.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    p = df[df["primary"] & df["auroc"].notna()]
    if p.empty:
        return None
    leaves = [l for l in BASELINE if l in set(p["leaf"])] + \
             sorted(set(p["leaf"]) - set(BASELINE))
    groups = ["participant-split"] + [t for t in HOLDOUTS if t in set(p["held_out"])]
    x = np.arange(len(leaves))
    w = 0.8 / len(groups)

    below = p[p["auroc"] < 0.5]
    note = ("shaded band = below chance; inverting the score would detect better"
            + (f"  —  {len(below)} of {len(p)} cells fall there" if len(below) else ""))
    fig, ax = plt.subplots(figsize=(1.6 * len(leaves) + 3, 4.4))
    for i, g in enumerate(groups):
        vals = [p[(p["leaf"] == l) & (p["held_out"] == g)]["auroc"].max() for l in leaves]
        label = "participant split" if g == "participant-split" else f"held out {g}"
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=label,
               color="0.35" if g == "participant-split" else None)
    ax.axhline(0.5, color="crimson", lw=1, ls="--")
    ax.text(len(leaves) - 0.5, 0.507, "chance", color="crimson", fontsize=8, ha="right")
    ax.set_xticks(x, [l.replace("vanilla/", "") for l in leaves], rotation=15, ha="right")
    ax.set_ylabel("AUROC, ID vs OOD")
    # The lower limit MUST come from the data. A fixed floor of 0.4 silently rendered the two
    # most important bars -- Ensemble_GP and Ensemble_forecast with standing held out, at 0.236
    # and 0.320 -- as nothing at all, so the figure implied those cells had not been run.
    # Below-chance values are the finding here, not an artefact to be cropped away.
    lo = float(p["auroc"].min())
    ax.set_ylim(max(0.0, lo - 0.08), 1.06)
    ax.axhspan(ax.get_ylim()[0], 0.5, color="crimson", alpha=0.05, zorder=0)

    ax.set_title(f"Leave-one-task-out vs the participant split ({split})\n{note}",
                 pad=24, fontsize=11)
    ax.legend(fontsize=8, ncol=len(groups), loc="lower center",
              bbox_to_anchor=(0.5, 1.02), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


MODES = ("LG", "RA", "RD", "SA", "SD", "TR", "ST")


def per_mode(split: str) -> pd.DataFrame:
    """Flag rate and median Psi for every (model, holdout, mode), from the eval files.

    This is the table the leave-one-task-out design exists to produce: the same mode appears
    as in-distribution in some runs and out-of-distribution in others, so a score that tracks
    familiarity and a score that tracks signal amplitude finally look different.
    """
    rows = []
    for task_dir in sorted(d for d in HERE.iterdir() if d.is_dir()):
        for f in sorted(task_dir.rglob(f"eval_{split}.json")):
            r = json.loads(f.read_text())
            pm = r.get("per_mode")
            # Two schemas: the vanilla models write one list of rows; the probabilistic one
            # writes {score_name: rows} because it scores four candidates. Take `epistemic`,
            # the score comparable to the vanilla ensembles' branch variance.
            if isinstance(pm, dict):
                pm = pm.get("epistemic") or next(iter(pm.values()), None)
            if not isinstance(pm, list):
                continue
            leaf = f.parent.relative_to(task_dir).as_posix()
            leaf = leaf[:-4] if leaf.endswith("/run") else leaf
            for m in pm:
                key = next((k for k in m if k.startswith("flagged")), None)
                rows.append({"held_out": task_dir.name, "leaf": leaf, "mode": m["mode"],
                             "kind": m["kind"], "windows": m["windows"],
                             "flagged_pct": m.get(key),
                             "median_psi": m.get("median_psi", m.get("median"))})
    return pd.DataFrame(rows)


def figure_modes(pm: pd.DataFrame, split: str, out: Path) -> Path | None:
    """Flag rate per mode, one panel per model, one bar per holdout.

    Hatched bars mark the run in which that mode was the held-out task. If a score measures
    unfamiliarity, the hatched bar should stand well above its plain neighbours. If it measures
    amplitude, the bars are the same height and the hatching tells you nothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    leaves = [l for l in BASELINE if l in set(pm["leaf"])]
    leaves += sorted(set(pm["leaf"]) - set(BASELINE))
    if not leaves:
        return None
    tasks = [t for t in HOLDOUTS if t in set(pm["held_out"])]
    modes = [m for m in MODES if m in set(pm["mode"])]

    fig, axes = plt.subplots(len(leaves), 1, figsize=(9, 2.6 * len(leaves)), squeeze=False,
                             sharex=True)
    x = np.arange(len(modes))
    w = 0.8 / max(len(tasks), 1)
    for ax, leaf in zip(axes[:, 0], leaves):
        sub = pm[pm["leaf"] == leaf]
        for i, t in enumerate(tasks):
            d = sub[sub["held_out"] == t].set_index("mode")
            vals = [d["flagged_pct"].get(m, np.nan) for m in modes]
            hatch = ["//" if m in held_out_modes(t) else "" for m in modes]
            bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=f"−{t}")
            for b, h in zip(bars, hatch):
                b.set_hatch(h)
                if h:
                    b.set_edgecolor("black")
        ax.set_ylabel("flagged %")
        ax.set_title(leaf, fontsize=10, loc="left")
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, ncol=len(tasks), loc="upper left",
                      title="hatched = held out in that run", title_fontsize=8)
    axes[-1, 0].set_xticks(x, modes)
    fig.suptitle(f"Flag rate per mode, by which task was held out ({split})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def figure_psi(pm: pd.DataFrame, split: str, out: Path) -> Path | None:
    """Median Psi per mode, one line per held-out task, one panel per model.

    This is the sharper version of `figure_modes`. Flag rate is read through a threshold and
    can sit at 0% or 100% for every mode at once, saying nothing; the median score underneath
    it always says something. If a score measured unfamiliarity, the point for the held-out
    task (ringed) would lift away from the line it sits on in the other runs. If the lines lie
    on top of each other, the score is a fixed function of the signal and the run's AUROC is
    just reporting where the held-out task happens to fall in a ranking the model would have
    produced anyway.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    leaves = [l for l in BASELINE if l in set(pm["leaf"])]
    leaves += sorted(set(pm["leaf"]) - set(BASELINE))
    leaves = [l for l in leaves if pm[(pm["leaf"] == l)]["held_out"].nunique() >= 2]
    if not leaves:
        return None
    tasks = [t for t in MODES if t in set(pm["held_out"])]

    fig, axes = plt.subplots(1, len(leaves), figsize=(4.4 * len(leaves), 3.9), squeeze=False)
    for ax, leaf in zip(axes[0], leaves):
        sub = pm[pm["leaf"] == leaf]
        # order modes by their median psi in the first run, so a stable ranking reads as flat
        ref = sub[sub["held_out"] == tasks[0]].set_index("mode")["median_psi"]
        modes = [m for m in ref.sort_values().index if m in set(sub["mode"])]
        x = np.arange(len(modes))
        for t in tasks:
            d = sub[sub["held_out"] == t].set_index("mode")
            y = [d["median_psi"].get(m, np.nan) for m in modes]
            ln, = ax.plot(x, y, "o-", ms=5, lw=1.4, alpha=0.85, label=f"−{t}")
            if t in modes:                       # ring the held-out task's own point
                ax.plot(modes.index(t), d["median_psi"].get(t, np.nan), "o", ms=13,
                        mfc="none", mec=ln.get_color(), mew=2)
        ax.set_xticks(x, modes)
        ax.set_title(leaf.replace("vanilla/", ""), fontsize=10, loc="left")
        ax.set_ylabel(r"median $\Psi$")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, axis="y")
    axes[0][0].legend(fontsize=8, title="held out", title_fontsize=8)
    fig.suptitle(r"Median $\Psi$ per task — rings mark the held-out task in that run "
                 "(overlapping lines = the score ignores training membership)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    df = collect(a.split)
    md = to_markdown(df, a.split)
    (HERE / f"summary_{a.split}.md").write_text(md)
    df.to_csv(HERE / f"summary_{a.split}.csv", index=False)
    fig = figure(df, a.split, HERE / f"auroc_{a.split}.png")
    pm = per_mode(a.split)
    if not pm.empty:
        pm.to_csv(HERE / f"per_mode_{a.split}.csv", index=False)
        figure_modes(pm, a.split, HERE / f"flagrate_{a.split}.png")
        figure_psi(pm, a.split, HERE / f"medianpsi_{a.split}.png")
    if not a.quiet:
        print(md)
    print(f"wrote {HERE / f'summary_{a.split}.md'}, .csv" + (f" and {fig.name}" if fig else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
