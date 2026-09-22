"""Run any completed model under leave-one-task-out, writing into ``ml/LOTO/<TASK>/...``.

    python ml/LOTO/run.py --list                        # the matrix and what is done
    python ml/LOTO/run.py --held-out SA --model autoencoder
    python ml/LOTO/run.py --held-out SA --model all     # every model, one held-out task
    python ml/LOTO/run.py --all                         # the whole 18-run matrix

Nothing about the models is reimplemented here. ``ml/vanilla/main.py`` and
``ml/prob_ensemble/{train,evaluate,plots}.py`` are driven exactly as they are on the
participant split; the only substitution is the data object, which ``splits.build`` replaces
with a task-restricted view of the same arrays. That is deliberate: if a LOTO number differs
from its participant-split counterpart, the holdout is the only thing it can be attributable
to.

Each run writes ``final.pt``, ``report.json``, ``eval_<split>.json`` and ``figures/`` into its
own leaf, so the eighteen runs never contend for a path.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(d for d in HERE.parents if (d / "data").is_dir())
sys.path.insert(0, str(HERE))

from splits import HELD_OUT_TASKS, build, trained_modes   # noqa: E402

# leaf name -> (package, vanilla model name, data family, extra data kwargs)
#
# ADDING A MODEL. If it is registered in ml/vanilla/registry.py, one line here is the whole
# job: give it a leaf name, its registry name, and which data family it reads
# ("ensemble" = labelled gait phase, "autoencoder"/"gan"/"synthetic" = the AE_GAN arrays).
# Add an HOURS entry so --list can total the matrix, and a BASELINE line in summarize.py if
# it has a participant-split counterpart to sit beside. Nothing else needs to change:
# summarize.py finds results by walking the tree, so a new leaf appears on its own.
#
# A model in its own package (like prob_ensemble) needs a run_* function in the shape of
# run_prob below -- patch its data builder, then drive its CLI.
MODELS: dict[str, tuple] = {
    "vanilla/AE":                ("vanilla", "autoencoder",    "autoencoder", {}),
    "vanilla/GAN":               ("vanilla", "gan",            "gan",         {}),
    "vanilla/Ensemble_GP":       ("vanilla", "ensemble",       "ensemble",    {}),
    "vanilla/Ensemble_synthetic": ("vanilla", "synthetic",     "synthetic",
                                   {"target": "correlation"}),
    "vanilla/Ensemble_forecast": ("vanilla", "forecast-angle", "synthetic",
                                  {"target": "forecast_angle", "horizon": 40}),
    "prob_ensemble":             ("prob_ensemble", None,       "synthetic",
                                  {"target": "forecast_angle", "horizon": 40}),
}

# Rough wall-clock on this laptop's MPS, scaled up by the ~1.4x larger training set that a
# six-mode LOTO split brings. Printed by --list so a full matrix is never launched blind.
HOURS = {"vanilla/AE": 0.1, "vanilla/GAN": 6.0, "vanilla/Ensemble_GP": 8.0,
         "vanilla/Ensemble_synthetic": 10.0, "vanilla/Ensemble_forecast": 9.0,
         "prob_ensemble": 8.0}
# Deferred to the HPC cluster, not run locally. All four use the 12-fold leave-one-subject-out
# stage, which is what the SLURM job array in ml/prob_ensemble/slurm/ exists to parallelise: on
# one laptop the column is ~20-30 h, on the cluster it collapses to the slowest single fold.
# `--all` skips them; `--include-hpc` or naming one explicitly still runs it.
HPC_DEFERRED = {"vanilla/Ensemble_GP", "vanilla/Ensemble_synthetic",
                "vanilla/Ensemble_forecast", "prob_ensemble"}
TIMINGS = HERE / "timings.json"     # measured hours, written as cells finish; --list prefers it


def _tune_backend() -> None:
    """GPU settings that are safe for a reproduction, applied once per process.

    ``cudnn.benchmark`` lets cuDNN pick an algorithm per shape and cache it. Every shape here
    is fixed, and it changes no arithmetic, so it is on.

    TF32 is **off** by default and gated behind ``$HIPEXO_TF32=1``. Measured on an A100 it cuts
    an autoencoder epoch from 16.4 s to 6.6 s, which is tempting -- but it truncates the
    mantissa of every convolution, and these runs exist to be compared against participant-split
    runs trained on a laptop in full precision. A 2.5x speed-up is not worth making that
    comparison approximate, when the actual win here is running eighteen cells at once.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    if os.environ.get("HIPEXO_TF32") == "1":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print("HIPEXO_TF32=1: TF32 enabled -- results are NOT bit-comparable with the "
              "full-precision participant-split runs")
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def leaf(task: str, model: str) -> Path:
    return HERE / task / model


def _record(task: str, model: str, rc: int, hours: float) -> None:
    """Append a measured runtime, so the next --list quotes fact rather than my guess."""
    import json
    try:
        d = json.loads(TIMINGS.read_text()) if TIMINGS.exists() else {}
    except json.JSONDecodeError:
        d = {}
    d.setdefault(model, {})[task] = {"hours": round(hours, 3), "ok": rc == 0}
    TIMINGS.write_text(json.dumps(d, indent=2, sort_keys=True))


def _measured(model: str) -> float | None:
    """Mean measured hours for a model, over the cells that have finished."""
    import json
    if not TIMINGS.exists():
        return None
    try:
        got = [v["hours"] for v in json.loads(TIMINGS.read_text()).get(model, {}).values()
               if v.get("ok")]
    except json.JSONDecodeError:
        return None
    return sum(got) / len(got) if got else None


# ---------------------------------------------------------------- vanilla

def run_vanilla(task: str, model: str, extra: list[str], stage: str = "all") -> int:
    """Drive ``ml/vanilla/main.py all``, with its registry pointed at the LOTO data."""
    _, name, family, kw = MODELS[model]
    out = leaf(task, model)
    sys.path.insert(0, str(REPO / "ml" / "vanilla"))

    import registry
    import main as vanilla_main

    def builder(batch_size=1024, horizon=None, **rest):
        args = dict(kw)
        if horizon is not None and "horizon" in args:
            args["horizon"] = horizon
        return build(family, task, batch_size=batch_size, **args, **rest)

    # the registry resolves `data` exactly once, and only wants something callable that
    # returns the data object -- so binding the holdout here needs no change in vanilla
    registry.REGISTRY[name] = replace(registry.REGISTRY[name], data=lambda: builder)

    argv = ["main.py", stage, "--model", name, "--out", str(out), *extra]
    print(f"\n$ {' '.join(argv)}")
    saved, sys.argv = sys.argv, argv
    try:
        return vanilla_main.main()
    finally:
        sys.argv = saved


# ---------------------------------------------------------------- prob_ensemble

N_FOLDS = 12


def prob_stages(task: str, out: Path, extra: list[str], only: str | None = None,
                fold: int | None = None) -> list[tuple[Path, list[str]]]:
    """The ordered (script, argv) pairs of the probabilistic ensemble's protocol.

    ``only`` selects one stage so that a SLURM array can run the twelve folds concurrently --
    which is the whole reason this model is worth putting on a cluster. Locally every stage
    runs in sequence.
    """
    pkg = REPO / "ml" / "prob_ensemble"
    folds = [(pkg / "train.py",
              ["train.py", "fold", "--index", str(i), "--out", str(out), *extra])
             for i in range(N_FOLDS)]
    rest = [("collect", pkg / "train.py", ["train.py", "collect", "--out", str(out)]),
            ("final", pkg / "train.py", ["train.py", "final", "--out", str(out), *extra]),
            ("eval", pkg / "evaluate.py",
             ["evaluate.py", "--out", str(out), "--split", "test"]),
            ("figures", pkg / "plots.py", ["plots.py", "--out", str(out), "--split", "test"])]
    if only == "fold":
        if fold is None:
            raise ValueError("--prob-stage fold needs --fold-index")
        return [folds[fold]]
    if only:
        return [(sc, av) for name, sc, av in rest if name == only]
    return folds + [(sc, av) for _, sc, av in rest]


def run_prob(task: str, extra: list[str], only: str | None = None,
             fold: int | None = None) -> int:
    """Drive ``ml/prob_ensemble`` through its full protocol, one stage per subprocess.

    Its stages are scripts, not functions -- everything lives under ``if __name__ ==
    "__main__"`` so that a SLURM array task can run exactly one of them. Driving them the same
    way here keeps the local and cluster paths identical, and gives each fold a fresh
    process-wide seed instead of inheriting whatever the previous fold left behind.

    Each stage is preceded by a prelude that rebinds ``data.build`` to the task-restricted
    view. The script's own ``import data as D`` then resolves to the already-patched module.
    """
    out = leaf(task, "prob_ensemble") / "run"
    pkg = REPO / "ml" / "prob_ensemble"
    prelude = (
        "import sys, runpy\n"
        f"sys.path.insert(0, {str(pkg)!r})\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import splits, data as pdata\n"
        "pdata.build = lambda batch_size=1024, scale=True: splits.build(\n"
        f"    'synthetic', {task!r}, batch_size=batch_size, scale=scale,\n"
        "    target=pdata.TARGET, horizon=pdata.HORIZON)\n"
    )

    stages = prob_stages(task, out, extra, only, fold)

    for script, argv in stages:
        code = prelude + f"sys.argv = {argv!r}\n" + \
            f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
        print(f"\n$ python {' '.join(argv)}   [held out {task}]")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO))
        if r.returncode:
            return r.returncode
    return 0


# ---------------------------------------------------------------- dispatch

def done(task: str, model: str) -> bool:
    d = leaf(task, model)
    return (d / "eval_test.json").exists() or (d / "run" / "eval_test.json").exists()


def cmd_list() -> int:
    print(f"leave-one-task-out matrix -- {len(HELD_OUT_TASKS)} tasks x {len(MODELS)} models\n")
    total = 0.0
    for task in HELD_OUT_TASKS:
        print(f"  {task} held out; trained on {' '.join(trained_modes(task))}")
        for m in MODELS:
            mark = "done" if done(task, m) else "    "
            meas = _measured(m)
            h = meas if meas is not None else HOURS[m]
            total += 0.0 if done(task, m) else h
            src = "measured" if meas is not None else "estimate"
            where = "  HPC" if m in HPC_DEFERRED else ""
            print(f"    [{mark}] {m:28s} ~{h:5.2f} h  ({src}){where}")
    print(f"\n  outstanding: ~{total:.0f} h of compute "
          f"({total / 24:.1f} days sequentially on this laptop)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--held-out", choices=HELD_OUT_TASKS)
    p.add_argument("--model", choices=(*MODELS, "all"))
    p.add_argument("--all", action="store_true", help="the whole matrix, sequentially")
    p.add_argument("--list", action="store_true")
    p.add_argument("--prob-stage", default=None,
                   choices=("fold", "collect", "final", "eval", "figures"),
                   help="prob_ensemble only: run one stage (for a SLURM array task)")
    p.add_argument("--fold-index", type=int, default=None,
                   help="with --prob-stage fold: which of the 12 LOSO folds")
    p.add_argument("--include-hpc", action="store_true",
                   help="also run the LOSO-based models that are otherwise left for the cluster")
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--force", action="store_true", help="rerun even if eval_test.json exists")
    p.add_argument("--stage", default="all", choices=("all", "train", "eval", "figures", "smoke"),
                   help="vanilla models only; `smoke` is a two-epoch wiring check")
    a, extra = p.parse_known_args()
    _tune_backend()

    if a.list or not (a.all or a.held_out):
        return cmd_list()

    tasks = HELD_OUT_TASKS if a.all else (a.held_out,)
    models = list(MODELS) if (a.all or a.model == "all") else [a.model]
    if a.all and not a.include_hpc:
        skipped = [m for m in models if m in HPC_DEFERRED]
        models = [m for m in models if m not in HPC_DEFERRED]
        if skipped:
            print(f"deferred to the cluster, not run here: {', '.join(skipped)}\n"
                  f"  (pass --include-hpc to run them locally anyway)\n")
    if not models or models == (None,):
        p.error("--model is required unless --all is given")
    # Cheapest first, and all three holdouts of one model before moving on. A matrix this long
    # should produce a complete, comparable row early rather than one finished column and two
    # empty ones -- and if something is wrong with the protocol, the AE finds out in 30 min.
    models.sort(key=lambda m: HOURS[m])

    results: list[tuple[str, str, int, float]] = []
    for m in models:
        for task in tasks:
            if (done(task, m) and not a.force and a.stage == "all"
                    and a.prob_stage is None):
                print(f"[skip] {task}/{m} already has eval_test.json")
                continue
            leaf(task, m).mkdir(parents=True, exist_ok=True)
            print(f"\n{'=' * 78}\n{task} held out | {m}\n{'=' * 78}")
            t0 = time.time()
            try:
                rc = (run_prob(task, extra, a.prob_stage, a.fold_index)
                      if m == "prob_ensemble"
                      else run_vanilla(task, m, extra, stage=a.stage))
            except Exception as e:                      # noqa: BLE001
                import traceback
                traceback.print_exc()
                rc, note = 1, f"{type(e).__name__}: {e}"
            else:
                note = ""
            hours = (time.time() - t0) / 3600
            print(f"\n{task}/{m}: {'ok' if rc == 0 else f'FAILED rc={rc} {note}'} "
                  f"in {hours:.2f} h", flush=True)
            # A matrix this long must not lose four finished days to one bad cell. Failures are
            # recorded and the run continues; the exit code still reflects that some cell failed.
            results.append((task, m, rc, hours))
            if a.stage == "all" and a.prob_stage is None:
                _record(task, m, rc, hours)

    bad = [(t, m) for t, m, rc, _ in results if rc]
    print(f"\n{'=' * 78}\n{len(results) - len(bad)}/{len(results)} cells ok, "
          f"{sum(h for *_, h in results):.1f} h total")
    for t, m in bad:
        print(f"  FAILED  {t}/{m}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
