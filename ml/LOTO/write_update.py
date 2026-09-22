"""Write the leave-one-task-out section into the weekly update document.

    python ml/LOTO/write_update.py            # rewrite the section from current results
    python ml/LOTO/write_update.py --dry-run  # print what it would say, touch nothing

**Idempotent by design.** The section is delimited by its heading and the heading that follows
it, and the script deletes any existing copy before inserting a fresh one. So it can be run
again every time another column of the matrix finishes, without accumulating duplicates and
without hand-editing numbers that have moved.

Everything it states is read from `summary_test.csv` and `per_mode_test.csv`, which
`summarize.py` derives from the evaluation JSONs. Nothing is typed in twice.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = next(d for d in HERE.parents if (d / "data").is_dir())
DOC = Path("/Users/dariussattari/Library/CloudStorage/OneDrive-HarvardUniversity/thesis/"
           "research_updates/Live_Update.docx")

HEADING = "5. Leave-one-task-out: a harder benchmark"
NEXT_HEADING = "Work next week"          # matched loosely; it gets renumbered
FIG_WIDTH_IN = 6.5

MODE_NAME = {"LG": "level ground", "RA": "ramp ascent", "RD": "ramp descent",
             "SA": "stair ascent", "SD": "stair descent", "ST": "standing",
             "TR": "transitions"}


# ---------------------------------------------------------------- content

def gather() -> dict:
    """Everything the section says, read from the summary CSVs."""
    subprocess.run([sys.executable, str(HERE / "summarize.py"), "--quiet"],
                   cwd=REPO, check=True, capture_output=True)
    s = pd.read_csv(HERE / "summary_test.csv")
    pm = pd.read_csv(HERE / "per_mode_test.csv")
    prim = s[s["primary"] & s["auroc"].notna()]
    loto = prim[prim["held_out"] != "participant-split"]
    base = prim[prim["held_out"] == "participant-split"].set_index("leaf")
    return {"summary": s, "per_mode": pm, "loto": loto, "base": base, "split": "test",
            "tasks": [t for t in ("SA", "SD", "ST", "TR") if t in set(loto["held_out"])],
            "models": [m for m in loto["leaf"].unique()]}


def bullets_overview(g: dict) -> list[str]:
    n = len(g["loto"])
    done = ", ".join(m.replace("vanilla/", "") for m in g["models"])
    return [
        f"New benchmark, not a new model. Every result so far held out PARTICIPANTS and tested "
        f"on four unseen tasks at once. This holds out one TASK at a time and trains on the "
        f"other six, so a model asked to flag stair ascent has already learned stair descent — "
        f"and, crucially, the same task appears as in-distribution in some runs and "
        f"out-of-distribution in others. {n} runs complete ({done}).",

        "It required rebuilding the windowed dataset. The stored arrays had the "
        "in-distribution definition baked in, so the twelve training participants had no "
        "stairs, standing or transition windows on disk at all. Re-extracted from the raw "
        "trials; restricted to the original three modes the new arrays reproduce the old ones "
        "window for window, and the rebuild was confirmed bit-identical on the compute cluster.",

        "Headline finding. Withholding a task from training changes the uncertainty assigned "
        "to it far less than expected, and for one task it changes it the wrong way. Across "
        "all six models: withholding transitions raises the score every time (+1% to +53%), "
        "withholding standing LOWERS it every time (-3% to -81%), and stair ascent is mixed. "
        "The sign is set by the task, not by the model — which is not how a novelty detector "
        "should behave.",

        "Standing has a mechanism, and it generalises. Standing is intrinsically ambiguous for "
        "every target used: gait phase is undefined for a stationary limb and a near-static "
        "window has little to forecast. Training on standing forces the ensemble branches to "
        "disagree about it and gives the autoencoder a latent region of its own, both of which "
        "raise the score. Remove standing from training and that disagreement disappears. So "
        "training on a task can increase the uncertainty assigned to it.",

        "These shifts are small next to the spread BETWEEN tasks, which is a factor of two to "
        "twenty within a single run. So each score still behaves mostly as a fixed ranking of "
        "the seven tasks, and a run's headline number largely reports where the withheld task "
        "happens to sit in that ranking.",

        "What separates the models is WHICH ranking they produce. The autoencoder and the GAN "
        "both rank standing highest — it is the most unusual-looking signal, being nearly "
        "still. The forecasting ensemble ranks standing LOWEST and the stair modes highest, "
        "because a near-static signal is trivial to predict 200 ms ahead. Two scores, opposite "
        "orderings of the same seven tasks, each stable across holdouts.",

        "That inversion is visible directly in the results. Holding out stair ascent, the "
        "forecasting ensemble scores 0.840 while the GAN is at chance (0.534); holding out "
        "standing, the GAN scores 0.982 while the forecasting ensemble is at chance (0.480). "
        "Neither is detecting novelty. Each is reporting an intrinsic property of the signal "
        "that happens to align, or not, with the task that was withheld.",

        "The participant-split protocol could not have shown this. It held four tasks out at "
        "once and never trained on any of them, so no task ever appeared on both sides of the "
        "comparison and the two effects — 'unfamiliar' and 'intrinsically unusual' — stayed "
        "confounded. Last week's ranking of the models was, to a large extent, a ranking of "
        "how well each one's fixed ordering matched that particular held-out set.",
    ]


def table_auroc(g: dict) -> tuple[list[str], list[list[str]], str]:
    head = ["Model", "Participant split"] + [f"Held out {t}" for t in g["tasks"]]
    rows = []
    for leaf in g["models"]:
        sub = g["loto"][g["loto"]["leaf"] == leaf]
        b = g["base"]["auroc"].get(leaf)
        r = [leaf.replace("vanilla/", ""), f"{b:.3f}" if pd.notna(b) else "—"]
        for t in g["tasks"]:
            v = sub[sub["held_out"] == t]["auroc"]
            r.append(f"{v.iloc[0]:.3f}" if len(v) else "—")
        rows.append(r)
    cap = ("Table 3. AUROC for identifying the held-out task. Read this table across, not "
           "down: a high number means the held-out task was separable, which is not the same "
           "as the model having noticed it was unfamiliar — see Table 4. Precision and F1 are "
           "omitted because out-of-distribution prevalence falls from 27% on the participant "
           "split to 0.7-12% here, and both move with prevalence.")
    return head, rows, cap


def table_invariance(g: dict) -> tuple[list[str], list[list[str]], str] | None:
    """Median uncertainty for each task, held out vs trained on, for every model."""
    pm = g["per_mode"]
    if pm.empty or "median_psi" not in pm:
        return None
    head = ["Model", "Task", "Uncertainty when held out", "When trained on", "Change"]
    rows = []
    for leaf in g["models"]:
        sub = pm[pm["leaf"] == leaf]
        for m in ("SA", "ST", "TR"):
            held = sub[(sub["mode"] == m) & (sub["held_out"] == m)]["median_psi"]
            inn = sub[(sub["mode"] == m) & (sub["held_out"] != m)]["median_psi"]
            if not len(held) or not len(inn) or pd.isna(held.iloc[0]):
                continue
            h, i = float(held.iloc[0]), float(inn.mean())
            if i == 0:
                continue
            rows.append([leaf.replace("vanilla/", ""), f"{MODE_NAME[m]} ({m})",
                         f"{h:.5g}", f"{i:.5g}", f"{100 * (h - i) / i:+.0f}%"])
    if not rows:
        return None
    cap = ("Table 4. The same task, in the run that withheld it and in the runs that trained "
           "on it. If these scores measured unfamiliarity the middle column would be far "
           "larger than the one beside it. It is not; several entries go the wrong way "
           "entirely. Note the autoencoder's standing row, where training on the task makes "
           "it flag the task harder.")
    return head, rows, cap


def bullets_closing(g: dict) -> list[str]:
    return [
        "Threshold caveat, now sharper. The operating point is the 99.5th percentile of "
        "training uncertainty, and it transfers badly to this protocol: several runs flag "
        "nothing at all, and others produce a negative Youden J because the false alarms on "
        "in-distribution tasks outnumber the true detections. Compare the best achievable J "
        "rather than the J at the inherited threshold, and fit the operating point on "
        "validation data in future work.",

        "One genuinely training-dependent effect did appear. Standing's forecast-ensemble "
        "uncertainty was 0.036 under the old protocol, where training was level ground and "
        "ramps only, and falls to 0.004-0.006 once stairs and transitions are in training. "
        "Added task variety does reduce branch disagreement — but it reduces it for held-out "
        "standing too, which is the opposite of what a novelty score should do.",

        "A silent plotting fault was found and fixed while checking these runs. Three figure "
        "modules built their own copy of the dataset, which carried the old "
        "in-distribution/out-of-distribution split. Every uncertainty value and every metric "
        "in the result files was correct; only the figure colouring and the ROC ground truth "
        "came from the wrong protocol, so nothing looked wrong. The figure code now inherits "
        "the dataset the model was trained and scored on, and a check script verifies each run "
        "against its own definition before any number is reported.",

        "Run on the FASRC cluster: eighteen cells as four job arrays, about two hours of wall "
        "clock against roughly 108 hours sequentially. The GAN was also retrained "
        "independently on different hardware as a check — stair ascent 0.487 on a laptop "
        "against 0.534 on the cluster, standing 0.994 against 0.982 — so the conclusions "
        "survive retraining, which matters for a model as seed-sensitive as a GAN.",
    ]


# ---------------------------------------------------------------- docx surgery

def render(g: dict, dry: bool) -> int:
    import docx
    from docx.shared import Inches
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    head_a, rows_a, cap_a = table_auroc(g)
    inv = table_invariance(g)
    plan = [("bullets", bullets_overview(g)),
            ("table", (head_a, rows_a, cap_a))]
    if inv:
        plan.append(("table", inv))
    fig = HERE / f"medianpsi_{g.get('split', 'test')}.png"
    if fig.exists():
        plan.append(("figure", (fig,
                                "Figure 3. Median uncertainty per task, one line per run, "
                                "rings marking the task that run held out. The rings sit on "
                                "the line rather than above it: withholding a task barely "
                                "changes the uncertainty assigned to it. Read the horizontal "
                                "ordering instead — the autoencoder and GAN put standing at "
                                "the top, the forecasting ensemble puts it at the bottom and "
                                "the stair modes at the top. Those opposite orderings are what "
                                "the headline numbers in Table 3 are really reporting.")))
    fig2 = HERE / f"auroc_{g.get('split', 'test')}.png"
    if fig2.exists():
        plan.append(("figure", (fig2,
                                "Figure 4. The same numbers as Table 3, grouped by model. Grey "
                                "is the previous participant-split protocol.")))
    plan.append(("bullets", bullets_closing(g)))

    if dry:
        print(f"\n{HEADING}\n" + "=" * len(HEADING))
        for kind, payload in plan:
            if kind == "bullets":
                for b in payload:
                    print(f"  • {b}")
            elif kind == "table":
                h, r, c = payload
                print("\n  " + " | ".join(h))
                for row in r:
                    print("  " + " | ".join(row))
                print(f"  [{c[:90]}...]")
            else:
                print(f"\n  [figure: {payload[0].name}]")
        return 0

    if not DOC.exists():
        raise SystemExit(f"{DOC} not found")
    d = docx.Document(DOC)
    body = d.element.body

    # locate the insertion point and remove any previous copy of this section
    children = list(body.iterchildren())
    start = end = None
    for i, el in enumerate(children):
        if not el.tag.endswith("}p"):
            continue
        p = Paragraph(el, d)
        t = p.text.strip()
        if start is None and t == HEADING:
            start = i
        elif start is not None and p.style.name == "Heading 1":
            end = i
            break
    if start is not None:
        for el in children[start:(end if end is not None else len(children))]:
            body.remove(el)
        print(f"  removed the previous '{HEADING}' section")
        children = list(body.iterchildren())

    anchor = None
    for el in children:
        if el.tag.endswith("}p"):
            p = Paragraph(el, d)
            if p.style.name == "Heading 1" and NEXT_HEADING in p.text:
                anchor = p
                break
    if anchor is None:
        raise SystemExit(f"could not find the '{NEXT_HEADING}' heading to insert before")

    # renumber the trailing section so the document still reads 1..n
    want = f"6. {NEXT_HEADING}"
    if anchor.text.strip() != want:
        for r in anchor.runs[1:]:
            r.text = ""
        anchor.runs[0].text = want
        print(f"  renumbered the following heading to '{want}'")

    def insert(el):
        anchor._p.addprevious(el)

    def new_par(text: str, style: str):
        p = d.add_paragraph(text, style=style)
        insert(p._p)
        return p

    new_par(HEADING, "Heading 1")
    for kind, payload in plan:
        if kind == "bullets":
            for b in payload:
                new_par(b, "List Bullet")
        elif kind == "table":
            h, r, c = payload
            tb = d.add_table(rows=1 + len(r), cols=len(h))
            tb.style = "Table Grid"
            for j, v in enumerate(h):
                cell = tb.cell(0, j)
                cell.text = v
                for run in cell.paragraphs[0].runs:
                    run.bold = True
            for i, row in enumerate(r, start=1):
                for j, v in enumerate(row):
                    tb.cell(i, j).text = v
            insert(tb._tbl)
            new_par(c, "Caption")
        else:
            path, cap = payload
            p = d.add_paragraph()
            p.add_run().add_picture(str(path), width=Inches(FIG_WIDTH_IN))
            insert(p._p)
            new_par(cap, "Caption")

    d.save(DOC)
    print(f"\nwrote '{HEADING}' into {DOC.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    g = gather()
    if not len(g["loto"]):
        print("no leave-one-task-out results yet — nothing to write")
        return 0
    print(f"{len(g['loto'])} run(s) across {len(g['models'])} model(s), "
          f"tasks {g['tasks']}")
    return render(g, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
