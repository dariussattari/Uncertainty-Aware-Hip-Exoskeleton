"""Check that every artefact in a leave-one-task-out leaf really is leave-one-task-out.

    python ml/LOTO/verify.py                 # all leaves, test split
    python ml/LOTO/verify.py --split val

This exists because of a bug that produced no error and no implausible number. Several plot
modules built their own data object from ``data/processed/``, whose ID/OOD partition is the
participant-split one — four modes out at once. Under a task holdout the model, the windows and
every Psi value were correct, so the figures looked right; only the ID/OOD *labelling* came
from the wrong protocol. Standing appeared orange in a run that trained on standing, the saved
score bundle claimed 23,043 OOD windows where the metrics said 10,273, and the ROC curve was
drawn against the wrong ground truth.

An eyeball scrub cannot be relied on to catch that again, and more models are still to be run.
So each check below compares an artefact against the one fact that defines the run — the
directory name — and the checks are cheap enough to run every time.

Per leaf:

1.  the evaluation JSON marks exactly one mode OOD, and it is the directory's mode
2.  its OOD window count equals that mode's window count in the rebuilt test split, and its
    ID count equals the rest
3.  the saved score bundle's labels agree with the JSON, window for window
4.  every figure is newer than the bundle it should have been drawn from

Exit status is non-zero if anything fails, so this can gate a commit or a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = next(d for d in HERE.parents if (d / "data").is_dir())
sys.path.insert(0, str(HERE))

from splits import ALL_MODES, DATA_ROOT, HOLDOUTS, held_out_modes, trained_modes     # noqa: E402

# bundle basename -> the family whose test meta it should match
BUNDLES = {"latents": "AE_GAN", "scores": "AE_GAN"}


def expected_counts(split: str, held_out: str) -> dict[str, dict[str, int]]:
    """Windows per mode in each rebuilt family, for this split."""
    out = {}
    for fam in ("AE_GAN", "ensemble"):
        p = DATA_ROOT / fam / f"{split}_meta.parquet"
        if p.exists():
            out[fam] = pd.read_parquet(p, columns=["mode"])["mode"].value_counts().to_dict()
    return out


def check_leaf(task: str, leaf: Path, split: str, counts: dict,
               skipped: list[str] | None = None) -> list[str]:
    """Return a list of failure messages; empty means the leaf is sound."""
    bad: list[str] = []
    skipped = [] if skipped is None else skipped
    ev = leaf / f"eval_{split}.json"
    if not ev.exists():
        ev = leaf / "run" / f"eval_{split}.json"
    if not ev.exists():
        return []                                   # not evaluated yet; nothing to check
    r = json.loads(ev.read_text())
    name = leaf.relative_to(HERE / task).as_posix()

    # --- 1. the OOD class is the held-out task, and only that ---------------------------
    out = sorted(held_out_modes(task))
    pm = r.get("per_mode")
    if isinstance(pm, list):
        ood = sorted(m["mode"] for m in pm if m["kind"] == "OOD")
        if ood != out:
            bad.append(f"{task}/{name}: per_mode marks {ood or 'nothing'} as OOD, expected "
                       f"{out} — the evaluation used the wrong protocol")
        ident = sorted(m["mode"] for m in pm if m["kind"] == "ID")
        if ident and ident != sorted(trained_modes(task)):
            bad.append(f"{task}/{name}: ID modes {ident} != trained modes "
                       f"{sorted(trained_modes(task))}")

    # --- 2. counts match the rebuilt arrays ---------------------------------------------
    m = r.get("metrics") or next(iter(r.get("scores", {}).values()), {})
    n_ood = m.get("n_ood")
    if n_ood is None and "counts" in m:
        n_ood = m["counts"]["tp"] + m["counts"]["fn"]
    n_id = m.get("n_id")
    if n_id is None and "counts" in m:
        n_id = m["counts"]["tn"] + m["counts"]["fp"]
    fam = "ensemble" if "Ensemble_GP" in name else "AE_GAN"
    want = counts.get(fam, {})
    if want and n_ood is not None:
        want_ood = sum(v for k, v in want.items() if k in out)
        if n_ood != want_ood:
            bad.append(f"{task}/{name}: n_ood {n_ood:,} != {want_ood:,} "
                       f"{'/'.join(out)} windows in {fam}/{split}")
        want_id = sum(v for k, v in want.items() if k not in out)
        if n_id != want_id:
            bad.append(f"{task}/{name}: n_id {n_id:,} != {want_id:,} non-{task} windows "
                       f"in {fam}/{split}")

    # --- 3. the saved bundle agrees with the JSON ---------------------------------------
    for stem in BUNDLES:
        npz, pq = leaf / f"{stem}_{split}.npz", leaf / f"{stem}_{split}.parquet"
        if not npz.exists():
            continue
        z = np.load(npz)
        if "labels" not in z:
            continue
        lab = z["labels"]
        if n_ood is not None and int(lab.sum()) != n_ood:
            bad.append(f"{task}/{name}: {npz.name} has {int(lab.sum()):,} OOD labels but the "
                       f"metrics say {n_ood:,} — the figures were drawn from other data")
        if pq.exists():
            meta = pd.read_parquet(pq, columns=["mode"])
            if len(meta) == len(lab):
                flagged = sorted(pd.Series(meta["mode"])[lab == 1].unique().tolist())
                if flagged and flagged != out:
                    bad.append(f"{task}/{name}: {npz.name} labels modes {flagged} as OOD, "
                               f"expected {out}")
        else:
            # Only the autoencoder and the GAN save a per-window meta table beside their
            # scores; the ensembles save labels alone. The count check above still applies and
            # is the one that catches a wrong-protocol bundle, but say so rather than letting
            # a weaker check pass as a full one.
            skipped.append(f"{task}/{name}: {npz.name} has no companion .parquet, so the "
                           f"mode-level check was skipped (counts still verified)")

        # --- 4. no figure predates the bundle it should come from ----------------------
        figs = sorted((leaf / "figures").glob("*.png"))
        stale = [f.name for f in figs if f.stat().st_mtime < npz.stat().st_mtime - 1]
        if stale:
            bad.append(f"{task}/{name}: {len(stale)} figure(s) older than {npz.name} "
                       f"({', '.join(stale[:3])}{'...' if len(stale) > 3 else ''}) — rerun "
                       f"`run.py --held-out {task} --model <model> --stage figures`")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="test", choices=("val", "test"))
    a = ap.parse_args()

    tasks = [d.name for d in sorted(HERE.iterdir())
             if d.is_dir() and d.name in HOLDOUTS]
    if not tasks:
        print("no task directories under ml/LOTO/")
        return 0

    failures, skipped, checked = [], [], 0
    for task in tasks:
        counts = expected_counts(a.split, task)
        for ev in sorted((HERE / task).rglob(f"eval_{a.split}.json")):
            leaf = ev.parent
            checked += 1
            failures += check_leaf(task, leaf, a.split, counts, skipped)

    if checked == 0:
        print(f"no eval_{a.split}.json found yet — nothing to verify")
        return 0
    print(f"checked {checked} evaluated leaf/leaves on the {a.split} split")
    if failures:
        print(f"\n{len(failures)} PROBLEM(S):")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("  ✓ OOD class is the held-out task everywhere")
    print("  ✓ window counts match the rebuilt arrays")
    print("  ✓ saved score bundles agree with the metrics")
    print("  ✓ no figure predates its bundle")
    for m in skipped:
        print(f"  – {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
