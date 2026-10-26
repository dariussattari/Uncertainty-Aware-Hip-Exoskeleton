"""Gait-phase targets for the ankle paper's ensemble baseline on the Nature data, built once on the laptop.

    python ml/nature/awareness/ensemble_data.py        # needs data/processed_nature/ae (ae_data.py) and data/nature

The ankle paper's best uncertainty estimator is an ensemble of gait-phase regressors (Tourk et al. 2025, Table I), so it
needs a gait phase for every training sample. The Nature data publishes none; each recording carries the insoles'
vertical force (``insole_{l,r}_force_y`` in Exo.csv, N), from which heel strikes are read here.

Rule (fixed before any model was trained)
  contact      a foot is in contact once its insole force rises above ``ON_N`` and until it falls below ``OFF_N``
               (hysteresis, so a force hovering at the threshold cannot fire twice)
  heel strike  the first sample of a contact, unless the swing before it (since the previous contact ended) is shorter
               than ``MIN_SWING_S``: that is a bounce of the force, not a step
  stride       heel strike to the next heel strike of the same leg, kept if it lasts ``MIN_STRIDE_S`` to ``MAX_STRIDE_S``
  phase        within a kept stride, elapsed time over the stride's duration, in [0, 1); NaN everywhere else

Writes ``data/processed_nature/ae/gait_phase.npy``, float32 (n_samples, 2): each leg's phase, aligned sample by sample
with ``signals.npy`` (same recording order and offsets), and ``gait_phase.json`` (the rule's constants, the strides and
coverage per task, the file's hash). Non-cyclic tasks get a phase only where the insoles happen to show a step-like
contact pattern: that is the ankle paper's known limit (phase is undefined off the cycle), not something to fix here.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # ml/nature
import nature_data as nt  # noqa: E402
import ae_data as ad  # noqa: E402

OUT = ad.OUT / 'gait_phase.npy'
ON_N, OFF_N = 50.0, 20.0                   # insole force thresholds (N): standing is about 400 N per foot, swing about 0
MIN_SWING_S = 0.1                          # a contact starting sooner after the previous one ended is a bounce
MIN_STRIDE_S, MAX_STRIDE_S = 0.6, 2.5      # walking and stairs take about 1 to 1.6 s per stride (nature notebook 02)
WORKERS = 8
FORCE_COLUMNS = [f'insole_{s}_force_y' for s in nt.SIDES]


def get_contact(force: npt.NDArray[np.float64], on_n: float = ON_N, off_n: float = OFF_N) -> npt.NDArray[np.bool_]:
    """Foot contact with hysteresis: set above ``on_n``, cleared below ``off_n``, otherwise carried (NaN carries too)."""
    event = np.where(force > on_n, 1, np.where(force < off_n, -1, 0))           # (n,) set, clear or carry
    last = np.maximum.accumulate(np.where(event != 0, np.arange(len(force)), -1))   # index of the last set or clear
    return (last >= 0) & (event[np.maximum(last, 0)] == 1)


def get_heel_strikes(contact: npt.NDArray[np.bool_], rate: int = nt.RATE_HZ, min_swing_s: float = MIN_SWING_S
                     ) -> npt.NDArray[np.int64]:
    """Samples where contact begins, dropping those that follow the previous contact's end by less than the swing floor."""
    edge = np.diff(contact.astype(np.int8), prepend = 0)                          # +1 rise, -1 fall
    rises, falls = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
    swing = np.full(len(rises), np.inf)                                           # no earlier fall: nothing to bounce from
    previous_fall = np.searchsorted(falls, rises) - 1                             # index of the last fall before each rise
    has_fall = previous_fall >= 0
    swing[has_fall] = rises[has_fall] - falls[previous_fall[has_fall]]
    return rises[swing >= min_swing_s * rate]


def get_phase(heel_strikes: npt.NDArray[np.int64], n: int, rate: int = nt.RATE_HZ, min_stride_s: float = MIN_STRIDE_S,
              max_stride_s: float = MAX_STRIDE_S) -> npt.NDArray[np.float32]:
    """Gait phase in [0, 1) over each kept stride (heel strike to the next), NaN elsewhere. (K,) -> (n,)."""
    starts, ends = heel_strikes[:-1], heel_strikes[1:]
    keep = (ends - starts >= min_stride_s * rate) & (ends - starts <= max_stride_s * rate)
    starts, lengths = starts[keep], (ends - starts)[keep]
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)   # 0..length-1 within each stride
    phase = np.full(n, np.nan, dtype = np.float32)
    phase[np.repeat(starts, lengths) + offsets] = offsets / np.repeat(lengths, lengths)
    return phase


def get_recording_phase(job: tuple[int, str, int]) -> tuple[int, npt.NDArray[np.float32], list[int]]:
    """Worker: both legs' phase for one recording (rec, path under data/nature, expected samples), and strides per leg."""
    rec, path, n_samples = job
    force = pd.read_csv(nt.NATURE / path / 'Exo.csv', usecols = FORCE_COLUMNS)[FORCE_COLUMNS].to_numpy(dtype = np.float64)
    if len(force) != n_samples:
        raise ValueError(f'{path}: Exo.csv has {len(force)} rows, the sample cache {n_samples}; rebuild the cache')
    phases, strides = [], []
    for leg in range(force.shape[1]):
        strikes = get_heel_strikes(get_contact(force[:, leg]))
        phase = get_phase(strikes, len(force))
        phases.append(phase)
        strides.append(int(np.sum(phase == 0)))                                   # every kept stride starts at phase 0
    return rec, np.stack(phases, axis = 1), strides                                 # (n, 2)


def build(out: Path = OUT, workers: int = WORKERS) -> pd.DataFrame:
    """Write gait_phase.npy and gait_phase.json; returns strides and coverage per task."""
    inventory = pd.read_parquet(nt.CACHE / 'samples_inventory.parquet')
    windows = pd.read_parquet(ad.OUT / 'windows.parquet', columns = ['rec', 'end', 'task', 'in_task'])
    offsets = np.r_[0, np.cumsum(inventory['n_samples'].to_numpy())]              # as ae_data.build_arrays fills signals.npy
    target = np.lib.format.open_memmap(out, mode = 'w+', dtype = np.float32, shape = (int(offsets[-1]), len(nt.SIDES)))
    rows = []
    jobs = [(int(r.rec), r.path, int(r.n_samples)) for r in inventory.itertuples()]
    with ProcessPoolExecutor(workers) as pool:
        for rec, phase, strides in pool.map(get_recording_phase, jobs, chunksize = 8):
            target[offsets[rec]:offsets[rec + 1]] = phase
            rows.append({'rec': rec, 'strides_l': strides[0], 'strides_r': strides[1]})
    target.flush()
    del target
    # coverage: share of in-task 1 s windows whose end sample has a phase for at least one leg, per task
    ends = pd.DataFrame(rows).merge(inventory[['rec', 'task']], on = 'rec')
    in_task = windows[windows['in_task']]
    phase_at_end = np.load(out, mmap_mode = 'r')[offsets[in_task['rec'].to_numpy()] + in_task['end'].to_numpy()]   # (W, 2)
    coverage = pd.Series(np.isfinite(phase_at_end).any(axis = 1)).groupby(in_task['task'].astype(str).to_numpy()).mean()
    table = ends.groupby('task')[['strides_l', 'strides_r']].sum()
    table['coverage'] = coverage
    table['controller_class'] = [nt.get_controller_class(t) for t in table.index]
    table = table.sort_values('coverage', ascending = False)
    meta = {'rule': {'on_n': ON_N, 'off_n': OFF_N, 'min_swing_s': MIN_SWING_S, 'min_stride_s': MIN_STRIDE_S,
                     'max_stride_s': MAX_STRIDE_S}, 'source': 'Exo.csv insole_{l,r}_force_y',
            'version': ad.get_data_version(out), 'per_task': table.reset_index().to_dict(orient = 'records')}
    out.with_suffix('.json').write_text(json.dumps(meta, indent = 2))
    return table


if __name__ == '__main__':
    table = build()
    print(table.to_string(float_format = lambda v: f'{v:.2f}'))
    print(f'-> {OUT} (version {json.loads(OUT.with_suffix(".json").read_text())["version"]})')
