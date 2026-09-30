"""Leave-one-task-out views over the mode-complete arrays.

The participant split is unchanged (seed 0, 12/4/4). What changes is the **task** partition:

    train        the 12 training participants, every mode EXCEPT the held-out one
    val  / test  the 4 val / 4 test participants, the same six modes -> in-distribution
    ood_val/ood_test   those same participants, the held-out mode only -> OOD

So a run with ``SA`` held out trains on level ground, both ramps, stair *descent*, standing and
transitions, and is asked to flag stair ascent. That is a much sharper question than the
participant-split protocol, which held out four tasks at once and therefore never tested whether
the model can flag a task whose nearest neighbour it has seen.

Nothing is copied. ``restrict()`` wraps a split so that row ``i`` of the view reads row
``keep[i]`` of the shared memmap; ``build()`` then swaps those views into a normally-constructed
``ml/vanilla`` data object. Every downstream consumer -- the two-stage trainer, the evaluator,
the plots -- sees the ordinary interface and does not know a holdout happened.

    from splits import build
    data = build("ensemble", held_out="SA")          # EnsembleGaitPhase, SA removed
    data = build("synthetic", held_out="TR", target="forecast_angle", horizon=40)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = next(d for d in Path(__file__).resolve().parents if (d / "data").is_dir())
sys.path.insert(0, str(REPO / "ml" / "vanilla"))

# $HIPEXO_LOTO_DATA lets the cluster keep the 9.9 GB of arrays on scratch while the code
# lives in $HOME. It must be the directory containing AE_GAN/ and ensemble/.
DATA_ROOT = Path(os.environ.get("HIPEXO_LOTO_DATA") or (REPO / "data" / "processed_all"))
ALL_MODES = ("LG", "RA", "RD", "SA", "SD", "ST", "TR")
# A holdout is a NAME mapping to the set of modes removed from training and scored as OOD.
# The first three are the original single-task holdouts; ``DOWN`` holds out a whole regime:
# both descents plus standing and transitions, so training sees level ground and ascents only
# (LG, RA, SA) and the question is whether "descending" can be flagged without ever being seen.
HOLDOUTS: dict[str, tuple[str, ...]] = {
    "SA": ("SA",), "ST": ("ST",), "TR": ("TR",),
    "DOWN": ("RD", "SD", "ST", "TR"),
}
HELD_OUT_TASKS = tuple(HOLDOUTS)

__all__ = ["ALL_MODES", "HOLDOUTS", "HELD_OUT_TASKS", "DATA_ROOT", "held_out_modes",
           "trained_modes", "restrict", "build"]


def held_out_modes(held_out: str) -> tuple[str, ...]:
    """The modes a holdout name removes from training (a single mode for SA/ST/TR)."""
    if held_out not in HOLDOUTS:
        raise ValueError(f"held_out must be one of {tuple(HOLDOUTS)}, got {held_out!r}")
    return HOLDOUTS[held_out]


def trained_modes(held_out: str) -> list[str]:
    out = set(held_out_modes(held_out))
    return [m for m in ALL_MODES if m not in out]


# ---------------------------------------------------------------- the row-restricted view

def restrict(split, keep: np.ndarray, as_inputs_only: bool = False):
    """A view of ``split`` limited to rows ``keep``, sharing its memmaps.

    ``as_inputs_only`` turns a labelled ``Split`` into something that quacks like an
    ``OodSplit``: ``take`` returns ``x`` alone, which is all a Psi score ever needs.
    """
    keep = np.asarray(keep, dtype=np.int64)
    parent = getattr(split, "_parent", None)
    if parent is not None:
        # A view of a view: compose the row maps and build directly on the original split, so
        # ``take`` applies exactly one row map. (Nesting the classes instead would apply the
        # inner map twice and index past the end of the inner view.)
        keep = split._keep[keep]
        as_inputs_only = as_inputs_only or bool(getattr(split, "_inputs_only", False))
        split = parent
    base = type(split)

    class _View(base):                                      # type: ignore[misc, valid-type]
        def __len__(self) -> int:
            return len(self._keep)

        def __repr__(self) -> str:
            tag = "ood" if as_inputs_only else "id"
            return (f"{base.__name__}View({self.split!r}/{tag}, n={len(self):,}, "
                    f"modes={sorted(self.meta['mode'].unique())})")

        def take(self, idx):
            out = base.take(self, self._keep[np.atleast_1d(np.asarray(idx, dtype=np.int64))])
            return out[0] if as_inputs_only and isinstance(out, tuple) else out

        def tensors(self, idx):
            import torch
            out = self.take(idx)
            return (torch.from_numpy(out) if isinstance(out, np.ndarray)
                    else tuple(torch.from_numpy(a) for a in out))

        def __getitem__(self, i: int):
            t = self.tensors([i])
            return tuple(a[0] for a in t) if isinstance(t, tuple) else t[0]

        def __getitems__(self, items):
            t = self.tensors(items)
            return list(t) if not isinstance(t, tuple) else list(zip(*t))

    v = _View.__new__(_View)
    v.__dict__.update(split.__dict__)        # shares X / y / mask memmaps -- no copy
    v._keep = keep
    v._parent = split
    v._inputs_only = as_inputs_only
    v.meta = split.meta.iloc[keep].reset_index(drop=True)
    return v


# ---------------------------------------------------------------- the holdout itself

def _apply(data, held_out: str) -> None:
    """Replace ``data``'s splits with task-restricted views, in place."""
    keep_modes = trained_modes(held_out)
    out_modes = held_out_modes(held_out)
    old_splits, old_ood = dict(data.splits), dict(getattr(data, "ood", {}))
    keeps: dict[str, np.ndarray] = {}
    new_ood = {}

    for name, sp in old_splits.items():
        mode = sp.meta["mode"].to_numpy()
        id_rows = np.flatnonzero(np.isin(mode, keep_modes))
        keeps[name] = id_rows
        data.splits[name] = restrict(sp, id_rows)
        if name in ("val", "test"):
            ood_rows = np.flatnonzero(np.isin(mode, out_modes))
            if len(ood_rows) == 0:
                raise ValueError(
                    f"no {'/'.join(out_modes)} windows in the {name} split -- the {name} "
                    f"participants never performed that task, so it cannot be scored as OOD")
            keeps[f"ood_{name}"] = ood_rows
            new_ood[name] = restrict(sp, ood_rows, as_inputs_only=True)

    data.train, data.val, data.test = (data.splits[s] for s in ("train", "val", "test"))
    data.ood = new_ood
    data.held_out_task = held_out
    data.held_out_modes = list(out_modes)
    data.trained_modes = keep_modes

    # SyntheticEnsembleData caches a per-row target ingredient keyed by id(split), and a
    # per-split map from each row to the row holding its future value. Both are row-indexed,
    # so both have to follow the restriction rather than survive it.
    cache = getattr(data, "_cache", None)
    if cache is not None:
        pairs = [(n, old_splits[n], data.splits[n], keeps[n]) for n in old_splits]
        pairs += [(f"ood_{n}", old_ood.get(n, old_splits[n]), new_ood[n], keeps[f"ood_{n}"])
                  for n in new_ood]
        data._cache = {id(new): cache[id(old)][k]
                       for _, old, new, k in pairs if id(old) in cache}
        if getattr(data, "_fwd", None):
            # recomputed, not subset: row numbers change, and a partner row that fell outside
            # the kept set must become -1 (masked out) rather than point at the wrong window
            data._fwd = {n: data._T.forecast_rows(new.meta, data.horizon)
                         for n, _, new, _ in pairs}
        if data.target == "correlation":
            data.target_stats = data._fit_target_stats()


# ---------------------------------------------------------------- construction

def build(family: str, held_out: str, batch_size: int = 1024, scale: bool = True,
          data_root: "str | Path | None" = None, **kw):
    """A vanilla data object over the mode-complete arrays, with ``held_out`` removed.

    ``family`` is one of ``ensemble`` (gait phase, labelled), ``autoencoder``, ``gan`` or
    ``synthetic`` (label-free targets; pass ``target=`` and ``horizon=``). ``data_root``
    replaces ``DATA_ROOT`` for one call, e.g. the 22-channel pelvis-IMU arrays of the ablation.
    """
    held_out_modes(held_out)          # validates the name
    root = Path(data_root) if data_root else DATA_ROOT
    # Check the family this call will actually read, not a fixed one. The gait-phase ensemble
    # reads `ensemble/` and everything else reads `AE_GAN/`, and on a cluster only the needed
    # family may have been staged to node-local disk -- so demanding AE_GAN unconditionally
    # fails a perfectly good Ensemble_GP run.
    need = "ensemble" if family == "ensemble" else "AE_GAN"
    if not (root / need / "train_X.npy").exists():
        raise FileNotFoundError(
            f"{root}/{need} has not been built (needed for family={family!r}). The arrays "
            f"under data/processed/ hold no SA/SD/ST/TR windows for the training participants, "
            f"so leave-one-task-out is impossible from them. Run: python ml/LOTO/build_data.py "
            f"-- or, on a cluster, check $HIPEXO_STAGE_FAMILY includes {need!r}.")

    import dataset as D
    if family == "ensemble":
        return _built(D.EnsembleGaitPhase(root=root / "ensemble",
                                          batch_size=batch_size, scale=scale), held_out)
    cls = {"autoencoder": D.AutoencoderData, "gan": D.GanData,
           "synthetic": D.SyntheticEnsembleData}[family]
    return _built(cls(root=root, batch_size=batch_size, scale=scale, **kw), held_out)


def _built(data, held_out: str):
    _apply(data, held_out)
    return data


if __name__ == "__main__":                       # python ml/LOTO/splits.py
    import argparse

    p = argparse.ArgumentParser(description="Inspect the leave-one-task-out views.")
    p.add_argument("--held-out", default=None, choices=HELD_OUT_TASKS)
    p.add_argument("--family", default="autoencoder",
                   choices=("ensemble", "autoencoder", "gan", "synthetic"))
    a = p.parse_args()

    for task in ([a.held_out] if a.held_out else HELD_OUT_TASKS):
        d = build(a.family, task, batch_size=256)
        print(f"\nheld out {task} = {' '.join(held_out_modes(task))}  trained on {' '.join(trained_modes(task))}")
        print(f"  {d}")
        for name in ("train", "val", "test"):
            sp = d.splits[name]
            by = sp.meta.groupby("mode").size().to_dict()
            print(f"    {name:5s} {len(sp):7,}  " +
                  "  ".join(f"{m}={by.get(m, 0):6,}" for m in trained_modes(task)))
        for name, sp in d.ood.items():
            by = sp.meta.groupby("mode").size().to_dict()
            print(f"    ood_{name:3s} {len(sp):7,}  " +
                  "  ".join(f"{m}={by.get(m, 0):6,}" for m in held_out_modes(task)))
