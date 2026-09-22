"""Rebuild the processed windows mode-complete, so leave-one-task-out is possible at all.

**Why this exists.** The arrays under ``data/processed/`` were partitioned by the *original*
in-distribution definition, and that definition is baked into the files rather than applied at
load time:

    data/processed/AE_GAN/{train,val,test}   LG, RA, RD only -- all three splits
    data/processed/ood/{val,test}            SA, SD, ST, TR  -- val and test only

The twelve training participants therefore have **no** SA/SD/ST/TR windows anywhere on disk.
Leave-one-task-out needs the opposite arrangement: with stair ascent held out, the other six
modes -- including stair descent, standing and transitions, from the training participants --
have to be available *for training*. Those windows have never been extracted.

So this is not a re-split of existing arrays. It is a re-extraction from the raw trials, which
are still present under ``data/molinaro-raw`` (6.1 GB).

**What it writes.** One mode-complete set per family, under ``data/processed_all/``:

    data/processed_all/ensemble/{train,val,test}_{X,y,mask}.npy + _meta.parquet
    data/processed_all/AE_GAN/{train,val,test}_X.npy           + _meta.parquet

Each split now holds **all seven modes** for its own participants. Nothing is labelled ID or
OOD on disk; ``splits.py`` applies the task holdout at load time, which is what makes three
different holdouts reusable from one build.

The participant assignment is unchanged (seed 0, 12/4/4), so a LOTO result and a
participant-split result differ only in the task partition.

**Correctness check.** After building, the LG/RA/RD subset of each new split is compared
against the corresponding existing array. They should match window for window; the extraction
logic below is lifted from ``data_exploration/02_build_windows.ipynb`` (cells 2, 7, 9) and any
divergence means the two have drifted.

    python ml/LOTO/build_data.py --dry-run     # sizes and counts, writes nothing
    python ml/LOTO/build_data.py               # the real build
    python ml/LOTO/build_data.py --verify      # re-check against the existing arrays
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants (notebook 02 cell 2)


def _side_block(side: str) -> list[str]:
    return ([f"thigh_accel_{a}_{side}" for a in "xyz"]
            + [f"thigh_gyro_{a}_{side}" for a in "xyz"]
            + [f"enc_angle_{side}", f"enc_velo_{side}"])


INPUT_16 = _side_block("l") + _side_block("r")
TARGET_GP = ["gp_hs_l", "gp_hs_r"]
SAMPLE_RATE_HZ, WINDOW, STRIDE = 200, 200, 10
ALL_MODES = ["LG", "RA", "RD", "SA", "SD", "ST", "TR"]
ORIGINAL_ID = ["LG", "RA", "RD"]


def repo_root() -> Path:
    for d in Path(__file__).resolve().parents:
        if (d / "data" / "molinaro-raw").is_dir():
            return d
    raise FileNotFoundError("could not locate the repo root (expected data/molinaro-raw)")


REPO = repo_root()
DATA_ROOT = REPO / "data" / "molinaro-raw"
OUT_ROOT = REPO / "data" / "processed_all"


# ---------------------------------------------------------------- extraction (notebook 02 cell 7)
# Lifted verbatim in behaviour from data_exploration/02_build_windows.ipynb. If that notebook
# changes, this must change with it -- --verify is what catches the drift.

def extract_windows(subject: str, trial: str, target_cols: list[str] | None = None,
                    window: int = WINDOW, stride: int = STRIDE, require_both: bool = False):
    d = DATA_ROOT / subject / trial
    X_full = pd.read_csv(d / "exo.csv", usecols=INPUT_16)[INPUT_16].to_numpy(np.float32)

    supervised = target_cols is not None
    n_t = len(target_cols) if supervised else 0
    empty = (np.empty((0, 16, window), np.float32), np.empty((0, n_t), np.float32),
             np.empty((0, n_t), np.float32), np.empty(0, np.int32))

    n = len(X_full)
    if n < window:
        return empty

    ends = np.arange(window - 1, n, stride, dtype=np.int32)
    bad = np.isnan(X_full).any(axis=1)
    cum = np.concatenate([[0], np.cumsum(bad)])
    keep = (cum[ends + 1] - cum[ends + 1 - window]) == 0

    valid = None
    if supervised:
        tgt_file = "gp" if target_cols[0].startswith("gp_") else "moment"
        y_full = pd.read_csv(d / f"{tgt_file}.csv",
                             usecols=target_cols)[target_cols].to_numpy(np.float32)
        valid = ~np.isnan(y_full[ends])
        keep = keep & (valid.all(axis=1) if require_both else valid.any(axis=1))

    ends_kept = ends[keep]
    if len(ends_kept) == 0:
        return empty

    starts = ends_kept - window + 1
    idx = starts[:, None] + np.arange(window)[None, :]
    X = np.ascontiguousarray(np.transpose(X_full[idx], (0, 2, 1)))
    if not supervised:
        return X, empty[1], empty[2], ends_kept
    y = np.nan_to_num(y_full[ends_kept], nan=0.0)
    mask = valid[keep].astype(np.float32)
    return X, y, mask, ends_kept


def to_sine_phase(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """sin(2*pi*phase), the paper's target -- avoids the heel-strike discontinuity."""
    return np.where(mask > 0, np.sin(2.0 * np.pi * y), 0.0).astype(np.float32)


# ---------------------------------------------------------------- build

def build_split(name: str, trials: pd.DataFrame, out_dir: Path,
                target_cols: list[str] | None = None, overwrite: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    supervised = target_cols is not None
    paths = {"X": out_dir / f"{name}_X.npy", "meta": out_dir / f"{name}_meta.parquet"}
    if supervised:
        paths |= {"y": out_dir / f"{name}_y.npy", "mask": out_dir / f"{name}_mask.npy"}
    if paths["X"].exists() and not overwrite:
        print(f"  [{out_dir.name}/{name}] exists; pass --overwrite to rebuild")
        return paths

    counts, kept = [], []
    for r in trials.itertuples():
        *_, ends = extract_windows(r.subject, r.trial, target_cols)
        if len(ends):
            counts.append(len(ends)); kept.append(r)
    total = int(sum(counts))
    print(f"  [{out_dir.name}/{name}] {len(kept)}/{len(trials)} trials -> {total:,} windows "
          f"({total * 16 * WINDOW * 4 / 1e9:.2f} GB)")

    n_t = len(target_cols) if supervised else 0
    mm = {"X": np.lib.format.open_memmap(paths["X"], mode="w+", dtype=np.float32,
                                         shape=(total, 16, WINDOW))}
    if supervised:
        for k in ("y", "mask"):
            mm[k] = np.lib.format.open_memmap(paths[k], mode="w+", dtype=np.float32,
                                              shape=(total, n_t))

    rows, off = [], 0
    for r, n in zip(kept, counts):
        X, y, mask, ends = extract_windows(r.subject, r.trial, target_cols)
        mm["X"][off:off + n] = X
        if supervised:
            mm["y"][off:off + n] = to_sine_phase(y, mask)
            mm["mask"][off:off + n] = mask
        rows.append(pd.DataFrame({
            "subject": r.subject, "trial": r.trial, "mode": r.mode,
            "condition": r.condition, "speed": r.speed, "controller": r.controller,
            "end_idx": ends, "row": np.arange(off, off + n, dtype=np.int64)}))
        off += n
    for a in mm.values():
        a.flush()
    pd.concat(rows, ignore_index=True).to_parquet(paths["meta"], index=False)
    return paths


def trial_pool() -> tuple[pd.DataFrame, dict]:
    inv = pd.read_parquet(REPO / "data" / "cache" / "inventory.parquet")
    pool = inv.query("complete and phase in [3, 4] and exo_hardware == 'custom'").copy()
    subjects = sorted(pool["subject"].unique().tolist())
    rng = np.random.default_rng(0)                       # same seed as notebook 02
    shuffled = [str(s) for s in rng.permutation(subjects)]
    splits = {"train": sorted(shuffled[:12]), "val": sorted(shuffled[12:16]),
              "test": sorted(shuffled[16:])}
    return pool[pool["mode"].isin(ALL_MODES)], splits


def verify(splits: dict) -> bool:
    """The LG/RA/RD subset of the new arrays must reproduce the existing ones."""
    ok = True
    for fam, old_dir in (("AE_GAN", REPO / "data/processed/AE_GAN"),
                         ("ensemble", REPO / "data/processed")):
        for name in ("train", "val", "test"):
            new_m = OUT_ROOT / fam / f"{name}_meta.parquet"
            old_m = old_dir / f"{name}_meta.parquet"
            if not (new_m.exists() and old_m.exists()):
                print(f"  {fam}/{name}: missing, skipped"); continue
            nm, om = pd.read_parquet(new_m), pd.read_parquet(old_m)
            sub = nm[nm["mode"].isin(ORIGINAL_ID)]
            key = lambda d: set(zip(d.subject, d.trial, d.end_idx))
            same = key(sub) == key(om)
            print(f"  {fam}/{name}: new ID-subset {len(sub):,} vs existing {len(om):,}  "
                  f"identical={same}")
            ok &= same
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true", help="counts only, writes nothing")
    p.add_argument("--verify", action="store_true", help="check against the existing arrays")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--family", default="both", choices=("both", "AE_GAN", "ensemble"))
    a = p.parse_args()

    pool, splits = trial_pool()
    print(f"repo {REPO}")
    print(f"pool {len(pool):,} trials, modes {sorted(pool['mode'].unique())}")
    for k, v in splits.items():
        print(f"  {k:5s} {len(v):2d} participants {v}")

    if a.verify:
        return 0 if verify(splits) else 1

    print("\nwindows per split and mode (what the rebuild will produce):")
    for name, subs in splits.items():
        d = pool[pool["subject"].isin(subs)]
        by = d.groupby("mode").size().to_dict()
        print(f"  {name:5s} {sum(by.values()):4d} trials  " +
              "  ".join(f"{m}={by.get(m, 0):3d}" for m in ALL_MODES))
    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    fams = [("AE_GAN", None), ("ensemble", TARGET_GP)]
    if a.family != "both":
        fams = [f for f in fams if f[0] == a.family]
    for fam, target in fams:
        print(f"\n=== {fam} ===")
        for name, subs in splits.items():
            build_split(name, pool[pool["subject"].isin(subs)], OUT_ROOT / fam,
                        target_cols=target, overwrite=a.overwrite)
        # The scaler is **reused from the participant-split build**, not refit. It was fit on
        # training windows of LG/RA/RD, which is a population these arrays still contain, and
        # copying it makes a LOTO run and a participant-split run scale their inputs by exactly
        # the same numbers. Refitting would introduce a small, pointless divergence into the one
        # comparison this directory exists to make. It also keeps scaling independent of which
        # task a run holds out.
        src = REPO / "data" / "processed" / ("AE_GAN" if fam == "AE_GAN" else "") / "scaler.npz"
        if src.exists():
            (OUT_ROOT / fam / "scaler.npz").write_bytes(src.read_bytes())
            print(f"  copied the scaler from {src.relative_to(REPO)}")
        else:
            meta = pd.read_parquet(OUT_ROOT / fam / "train_meta.parquet")
            X = np.load(OUT_ROOT / fam / "train_X.npy", mmap_mode="r")
            idx = meta.index[meta["mode"].isin(ORIGINAL_ID)].to_numpy()
            rng = np.random.default_rng(0)
            pick = np.sort(rng.choice(idx, size=min(60_000, len(idx)), replace=False))
            sub = np.asarray(X[pick], dtype=np.float64)
            mean, std = sub.mean(axis=(0, 2)), sub.std(axis=(0, 2))
            std[std < 1e-8] = 1.0
            np.savez(OUT_ROOT / fam / "scaler.npz", mean=mean.astype(np.float32),
                     std=std.astype(np.float32), channels=np.array(INPUT_16))
            print("  refit the scaler on LG/RA/RD training windows "
                  "(data/processed/ had none to copy)")
        (OUT_ROOT / fam / "config.json").write_text(json.dumps({
            "source": "Molinaro et al. 2024, Sci. Robot. 9 eadi8852 (Zenodo 10849318)",
            "set": f"{fam}, mode-complete for leave-one-task-out",
            "input_channels": INPUT_16, "n_channels": 16,
            "sample_rate_hz": SAMPLE_RATE_HZ, "window_samples": WINDOW,
            "window_seconds": WINDOW / SAMPLE_RATE_HZ, "stride_samples": STRIDE,
            "modes": ALL_MODES, "splits": splits,
            "scaler_fit_on": f"train windows of {ORIGINAL_ID} only, so that scaling is "
                             f"independent of which task a run holds out",
            "note": "no ID/OOD partition on disk; ml/LOTO/splits.py applies it at load time",
        }, indent=2))
        print(f"  wrote scaler.npz and config.json")

    print("\nverifying against the existing arrays:")
    return 0 if verify(splits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
