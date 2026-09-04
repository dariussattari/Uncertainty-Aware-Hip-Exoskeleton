"""Data interface for the vanilla gait-phase ensemble.

Reads the arrays built by ``data_exploration/02_build_windows.ipynb`` from
``data/processed/``. Windows are already cut — nothing here re-windows anything.

    from dataset import EnsembleGaitPhase

    data = EnsembleGaitPhase()
    for x, y, mask in data.loader("train", shuffle=True):
        loss = data.masked_mse(model(x), y, mask)

Three details this module exists to get right:

1. **Arrays on disk are unscaled.** The ``StandardScaler`` lives beside them and is applied
   on read.
2. **``mask`` is not optional.** A window can have ground truth for one leg and not the other
   (inverse dynamics was only valid on the force plates). The missing leg carries ``y = 0``,
   which in the tanh range reads as mid-stride — a plausible value that is not a label. Use
   :meth:`EnsembleGaitPhase.masked_mse`, never plain MSE.
3. **The arrays are memory-mapped** (``train_X.npy`` is 2.2 GB). :meth:`Split.take` sorts
   indices before gathering, because scattered reads across a file that size are much slower
   than monotonic ones, then restores the caller's ordering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

__all__ = ["EnsembleGaitPhase", "Split", "OodSplit", "repo_root"]

SPLITS = ("train", "val", "test")


def repo_root(marker: str = "data/processed") -> Path:
    """Locate the repo root, so paths never depend on the caller's cwd."""
    for d in Path(__file__).resolve().parents:
        if (d / marker).is_dir():
            return d
    raise FileNotFoundError(
        f"could not find {marker!r} above {Path(__file__).resolve()}. "
        "Run data_exploration/02_build_windows.ipynb first."
    )


class Split(Dataset):
    """One split's memory-mapped windows, with scaling and index filtering.

    Also a ``torch.utils.data.Dataset``: indexing with an int gives one ``(x, y, mask)``
    sample, and ``__getitems__`` lets PyTorch fetch a whole batch through a single gather.
    """

    def __init__(self, root: Path, split: str, mean: np.ndarray, std: np.ndarray, scale: bool):
        self.root, self.split, self.scale = root, split, scale
        self._mean, self._std = mean, std

        self.X = np.load(root / f"{split}_X.npy", mmap_mode="r")
        self.y = np.load(root / f"{split}_y.npy", mmap_mode="r")
        self.mask = np.load(root / f"{split}_mask.npy", mmap_mode="r")
        self.meta = pd.read_parquet(root / f"{split}_meta.parquet")

        if not len(self.X) == len(self.y) == len(self.mask) == len(self.meta):
            raise ValueError(f"length mismatch in split {split!r}")

    # -- basics ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:
        return (f"Split({self.split!r}, n={len(self):,}, shape={tuple(self.X.shape[1:])}, "
                f"subjects={len(self.subjects)})")

    @property
    def subjects(self) -> list[str]:
        return sorted(self.meta["subject"].unique().tolist())

    # -- reading --------------------------------------------------------------

    def take(self, idx: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gather ``(x, y, mask)`` for the given rows, preserving the requested order."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        order = np.argsort(idx, kind="stable")
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        s = idx[order]

        x = np.asarray(self.X[s], dtype=np.float32)[inverse]
        y = np.asarray(self.y[s], dtype=np.float32)[inverse]
        m = np.asarray(self.mask[s], dtype=np.float32)[inverse]
        if self.scale:
            x = (x - self._mean) / self._std
        return x, y, m

    def tensors(self, idx: Sequence[int] | np.ndarray) -> tuple[torch.Tensor, ...]:
        return tuple(torch.from_numpy(a) for a in self.take(idx))

    def __getitem__(self, i: int):
        return tuple(t[0] for t in self.tensors([i]))

    def __getitems__(self, items: Sequence[int]):
        return list(zip(*self.tensors(items)))

    def indices(
        self,
        subjects: Sequence[str] | None = None,
        modes: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Row indices matching the given subject / mode filters."""
        keep = np.ones(len(self), dtype=bool)
        if subjects is not None:
            keep &= self.meta["subject"].isin(list(subjects)).to_numpy()
        if exclude is not None:
            keep &= ~self.meta["subject"].isin(list(exclude)).to_numpy()
        if modes is not None:
            keep &= self.meta["mode"].isin(list(modes)).to_numpy()
        return np.flatnonzero(keep)


class OodSplit(Dataset):
    """Held-out-mode windows for out-of-distribution evaluation — inputs only.

    Lives in ``data/processed/ood/`` and is shared by every architecture: OOD scoring needs
    only ``Psi(x)``, never a label. Scaled with the *ensemble's* scaler, since that is what
    the model was trained under.
    """

    def __init__(self, root: Path, split: str, mean: np.ndarray, std: np.ndarray, scale: bool):
        self.root, self.split, self.scale = root, split, scale
        self._mean, self._std = mean, std
        self.X = np.load(root / f"{split}_X.npy", mmap_mode="r")
        self.meta = pd.read_parquet(root / f"{split}_meta.parquet")
        if len(self.X) != len(self.meta):
            raise ValueError(f"length mismatch in ood/{split}")

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:
        return (f"OodSplit({self.split!r}, n={len(self):,}, "
                f"modes={sorted(self.meta['mode'].unique())})")

    def take(self, idx: Sequence[int] | np.ndarray) -> np.ndarray:
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        order = np.argsort(idx, kind="stable")
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        x = np.asarray(self.X[idx[order]], dtype=np.float32)[inverse]
        return (x - self._mean) / self._std if self.scale else x

    def __getitem__(self, i: int):
        return torch.from_numpy(self.take([i])[0])

    def __getitems__(self, items: Sequence[int]):
        return list(torch.from_numpy(self.take(items)))


class EnsembleGaitPhase:
    """Everything the vanilla gait-phase ensemble needs from the data.

    Parameters
    ----------
    root
        Directory holding the arrays. Defaults to ``<repo>/data/processed``.
    batch_size
        Default batch size for :meth:`loader`. 1024 matches the paper.
    scale
        Apply the stored scaler on read. Leave ``True`` unless you have a specific reason.
    """

    def __init__(self, root: Path | str | None = None, batch_size: int = 1024, scale: bool = True):
        self.root = Path(root) if root is not None else repo_root() / "data" / "processed"
        self.batch_size = batch_size

        missing = [n for n in ("scaler.npz", *(f"{s}_X.npy" for s in SPLITS))
                   if not (self.root / n).exists()]
        if missing:
            raise FileNotFoundError(
                f"missing {missing} in {self.root}. "
                "Run data_exploration/02_build_windows.ipynb to build them."
            )

        sc = np.load(self.root / "scaler.npz")
        mean = sc["mean"].astype(np.float32)[None, :, None]   # broadcast over (n, C, T)
        std = sc["std"].astype(np.float32)[None, :, None]
        self.channels = [str(c) for c in sc["channels"]]

        cfg = self.root / "config.json"
        self.config = json.loads(cfg.read_text()) if cfg.exists() else {}

        self.splits = {s: Split(self.root, s, mean, std, scale) for s in SPLITS}
        self.train, self.val, self.test = (self.splits[s] for s in SPLITS)

        # Pseudo-OOD (held-out ambulation modes). Absent until 02_build_windows.ipynb
        # has been re-run with the ood/ section, so this stays optional.
        ood_root = self.root / "ood"
        self.ood = {
            s: OodSplit(ood_root, s, mean, std, scale)
            for s in ("val", "test")
            if (ood_root / f"{s}_X.npy").exists()
        }

    def __repr__(self) -> str:
        sizes = ", ".join(f"{s}={len(v):,}" for s, v in self.splits.items())
        ood = ", ".join(f"ood_{s}={len(v):,}" for s, v in self.ood.items())
        return f"EnsembleGaitPhase({sizes}{', ' + ood if ood else ''}, shape={self.shape})"

    def __getitem__(self, split: str) -> Split:
        return self.splits[split]

    # -- shape ----------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """``(channels, window)`` of one sample."""
        return tuple(self.train.X.shape[1:])

    @property
    def n_channels(self) -> int:
        return self.shape[0]

    @property
    def window(self) -> int:
        return self.shape[1]

    @property
    def n_targets(self) -> int:
        """2 — left and right leg."""
        return self.train.y.shape[1]

    # -- loading --------------------------------------------------------------

    def loader(
        self,
        split: str | Split,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> DataLoader:
        """A ``DataLoader`` yielding ``(x, y, mask)`` batches.

        Batching happens at the sampler level, so each step is one sorted gather from the
        memmap rather than ``batch_size`` separate seeks.

        ``num_workers`` is deliberately 0: the bottleneck is memmap I/O, and forking workers
        would duplicate the mapping per process for no gain.
        """
        sp = self.splits[split] if isinstance(split, str) else split
        idx = (np.arange(len(sp), dtype=np.int64) if indices is None
               else np.asarray(indices, dtype=np.int64))
        bs = batch_size or self.batch_size

        class _Batched(Dataset):
            def __len__(self) -> int:
                return len(idx)

            def __getitem__(self, batch: Sequence[int]):
                return sp.tensors(idx[np.asarray(batch, dtype=np.int64)])

        base = _Batched()
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        sampler = RandomSampler(base, generator=gen) if shuffle else SequentialSampler(base)

        return DataLoader(
            base,
            sampler=BatchSampler(sampler, batch_size=bs, drop_last=drop_last),
            batch_size=None,           # the sampler already yields batches
            collate_fn=lambda b: b,    # _Batched returns a finished batch
            num_workers=0,
        )

    def ood_loader(self, split: str, batch_size: int | None = None) -> DataLoader:
        """A ``DataLoader`` over OOD windows, yielding ``x`` only (no labels exist)."""
        if split not in self.ood:
            raise KeyError(
                f"no OOD data for {split!r} in {self.root / 'ood'}. "
                "Re-run data_exploration/02_build_windows.ipynb."
            )
        sp = self.ood[split]
        idx = np.arange(len(sp), dtype=np.int64)
        bs = batch_size or self.batch_size

        class _Batched(Dataset):
            def __len__(self) -> int:
                return len(idx)

            def __getitem__(self, batch: Sequence[int]):
                return torch.from_numpy(sp.take(idx[np.asarray(batch, dtype=np.int64)]))

        base = _Batched()
        return DataLoader(
            base,
            sampler=BatchSampler(SequentialSampler(base), batch_size=bs, drop_last=False),
            batch_size=None,
            collate_fn=lambda b: b,
            num_workers=0,
        )

    # -- training protocol ----------------------------------------------------

    def loso_folds(self, split: str = "train") -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
        """Leave-one-subject-out folds, yielding ``(subject, train_idx, heldout_idx)``.

        The paper rotates one subject out as the early-stopping set to determine an epoch
        count, then retrains on everyone with that count fixed.
        """
        sp = self.splits[split]
        for s in sp.subjects:
            yield s, sp.indices(exclude=[s]), sp.indices(subjects=[s])

    @staticmethod
    def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """MSE over labelled entries only.

        All three are ``(batch, 2)`` — left and right leg. Where ``mask == 0`` the target is a
        placeholder 0, not a label; including it would train the model toward a fabricated
        mid-stride value on whichever leg lacked force-plate coverage.
        """
        return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


if __name__ == "__main__":  # python ml/vanilla/dataset.py
    data = EnsembleGaitPhase()
    print(data, "\n")
    for name, sp in data.splits.items():
        x, y, m = sp.tensors(np.random.default_rng(0).choice(len(sp), 2048, replace=False))
        print(f"  {sp}")
        print(f"      x{tuple(x.shape)} {x.dtype} | mean {x.mean():+.3f} std {x.std():.3f} "
              f"| labelled L {m[:, 0].mean():.0%} R {m[:, 1].mean():.0%}")
