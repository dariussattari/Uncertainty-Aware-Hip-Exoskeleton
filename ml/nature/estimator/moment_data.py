"""Training arrays for the Nature joint moment estimator (Model A, the paper's sensors), built once on the laptop.

    python ml/nature/estimator/moment_data.py        # needs data/nature and the sample cache (nature notebook 01)

Molinaro, Scherpereel et al. (Nature 2024, Methods "Neural network architecture optimization"): the TCN input is a
sequence of *unilateral* hip and knee encoder data, thigh, shank and foot IMU data and insole vertical force and centre
of pressure, at 200 Hz, standardized with training-set statistics; the outputs are the total (biological + exoskeleton)
hip and knee flexion/extension moments in N m/kg, the hip estimated 50 ms behind the input sequence.

Writes ``data/processed_nature/moment/``:

  inputs.npy       float32 (n_samples, 2, 25): every recording end to end, per leg (0 left, 1 right) that leg's 25
                   channels (``CHANNELS``). Encoder velocities are recomputed causally from the angles (the published
                   channels are half scale, hip and knee alike). The left leg is mirrored into the right leg's frame
                   (``MIRRORED``), so one unilateral model serves both legs.
  targets.npy      float32 (n_samples, 2, 2): per leg, total hip moment HIP_DELAY samples before the sample and total
                   knee moment at it (Joint_Moments_Filt, OpenSim signs: flexion positive at the hip), N m/kg; NaN
                   where inverse dynamics is undefined.
  estimate.npy     float32 (n_samples, 2, 2): the deployed estimator's output (Exo.csv ``*_torque_estimated``, N m/kg,
                   as published), at the sample, for comparison.
  windows.parquet  one row per (window end, leg) every STRIDE samples with WINDOW clean input samples, the leg's activity
                   flag set and at least one target: ``start`` (first row of inputs.npy), ``leg``, recording, participant,
                   task, direction, controller class, participant split (the awareness models' 14 / 4 / 4) and
                   ``paper_train`` (in the paper's 19 selected training tasks).
  meta.json        channels, mirrored channels, window, stride, hip delay, the task selection, the split, data version.

Left/right frames (checked on one walking recording of each of the 22 participants, best lagged left/right correlation
within 0.3-0.9 s): the sensors share one frame, and the legs differ as a body mirrored in the sagittal plane does.
Lateral accelerometer (z), gyro x and gyro y of all three IMUs and insole COP z correlate negatively across legs in
22 of 22 people (21 of 22 for COP z); every other channel positively. Mirroring negates those.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                 # ml/nature
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'awareness'))   # ae_data: the shared participant split
import nature_data as nt  # noqa: E402
import features as ft  # noqa: E402  (ml/common, put on the path by nature_data)
import ae_data as ad  # noqa: E402

OUT = nt.REPO / 'data' / 'processed_nature' / 'moment'
WINDOW = 249                     # samples: the TCN's receptive field (ml/common/models/moment_tcn.py)
STRIDE = 5                       # samples between window ends (25 ms)
# ponytail: stride 5 keeps 1 in 5 near-duplicate 200 Hz windows; build at 1 if validation error still falls with data
HIP_DELAY = 10                   # samples: the paper trains the hip head 50 ms behind the input sequence
WORKERS = 8
IMUS = ('thigh', 'shank', 'foot')
# per leg, {s} = l or r; velocities are recomputed from the angle
CHANNELS = (['hip_angle_{s}', 'hip_angle_{s} (causal derivative)', 'knee_angle_{s}', 'knee_angle_{s} (causal derivative)']
            + [f'{imu}_imu_{{s}}_{kind}_{a}' for imu in IMUS for kind in ('accel', 'gyro') for a in 'xyz']
            + ['insole_{s}_force_y', 'insole_{s}_cop_x', 'insole_{s}_cop_z'])
MIRRORED = [f'{imu}_imu_{{s}}_{c}' for imu in IMUS for c in ('accel_z', 'gyro_x', 'gyro_y')] + ['insole_{s}_cop_z']
MOMENTS = ('hip_flexion_{s}_moment', 'knee_angle_{s}_moment')
ESTIMATES = ('hip_angle_{s}_torque_estimated', 'knee_angle_{s}_torque_estimated')
# The paper's 19 selected training tasks (Extended Data Fig. 3b: level-ground walking, then the 18 tasks added before
# the 95% generalization cut-off), as Nature recording folders, with the movement direction kept where a folder holds
# both directions (incline_walk = inclined + declined walk, stairs = stair ascent + descent; None keeps every sample).
# jump holds jump in place and jump across, both selected. dynamic_walk (toe and heel walk) and side_shuffle
# (calisthenics) are matched by elimination: they are the two folders whose names match no task in the paper, and the
# two paper tasks without a matching folder are toe and heel walk and calisthenics. Both are selected, so the
# assignment does not change the training set.
PAPER_TASKS = {'normal_walk': None, 'poses': None, 'side_shuffle': None, 'push': None, 'jump': None, 'turn_and_step': None,
               'cutting': None, 'sit_to_stand': None, 'walk_backward': None, 'weighted_walk': None, 'lift_weight': None,
               'tug_of_war': None, 'tire_run': None, 'dynamic_walk': None, 'twister': None, 'meander': None,
               'incline_walk': 'up', 'stairs': 'down'}


def get_leg_inputs(exo: pd.DataFrame, side: str) -> npt.NDArray[np.float32]:
    """One leg's 25 input channels (n, 25), velocities recomputed, the left leg mirrored into the right leg's frame."""
    columns = []
    for name in CHANNELS:
        if name.endswith('(causal derivative)'):
            x = ft.get_causal_velocity(exo[name.split(' ')[0].format(s = side)].to_numpy(dtype = float))
        else:
            x = exo[name.format(s = side)].to_numpy(dtype = float)
        columns.append(-x if side == 'l' and name in MIRRORED else x)
    return np.stack(columns, axis = 1).astype(np.float32)


def get_valid_ends(x: npt.NDArray[np.float32], flag: npt.NDArray[np.bool_], y: npt.NDArray[np.float32],
                   window: int = WINDOW, stride: int = STRIDE) -> npt.NDArray[np.int64]:
    """Window ends (every ``stride`` from ``window - 1``) with ``window`` finite input samples, the flag set at the end
    and at least one finite target. x (n, C), flag (n,), y (n, 2)."""
    bad = np.r_[0, np.cumsum(~np.isfinite(x).all(axis = 1))]                    # (n + 1,) running count of bad samples
    ends = np.arange(window - 1, len(x), stride)
    clean = (bad[ends + 1] - bad[ends + 1 - window]) == 0
    return ends[clean & flag[ends] & np.isfinite(y[ends]).any(axis = 1)]


def shift_back(x: npt.NDArray[np.float64], delay: int) -> npt.NDArray[np.float64]:
    """x[t - delay] at t, NaN for the first ``delay`` samples."""
    return np.r_[np.full(delay, np.nan), x[:len(x) - delay]]


def get_recording(job: tuple[int, str, int, str, str | None]) -> tuple[int, npt.NDArray, npt.NDArray, npt.NDArray, pd.DataFrame]:
    """Worker: inputs (n, 2, 25), targets (n, 2, 2), deployed estimates (n, 2, 2) and the window rows of one recording."""
    rec, path, n_samples, task, stair_dir = job
    folder = nt.NATURE / path
    exo = pd.read_csv(folder / 'Exo.csv')
    moments = pd.read_csv(folder / 'Joint_Moments_Filt.csv')
    cache = pd.read_parquet(nt.CACHE / 'samples' / f'{rec:05d}.parquet', columns = ['flag_l', 'flag_r', 'limb_power_l', 'limb_power_r'])
    if not len(exo) == len(moments) == len(cache) == n_samples:
        raise ValueError(f'{path}: Exo {len(exo)}, moments {len(moments)}, cache {len(cache)} rows; inventory says {n_samples}')
    x = np.stack([get_leg_inputs(exo, s) for s in nt.SIDES], axis = 1)                    # (n, 2, 25)
    y = np.stack([np.stack([shift_back(moments[MOMENTS[0].format(s = s)].to_numpy(dtype = float), HIP_DELAY),
                            moments[MOMENTS[1].format(s = s)].to_numpy(dtype = float)], axis = 1)
                  for s in nt.SIDES], axis = 1).astype(np.float32)                        # (n, 2, 2)
    est = np.stack([exo[[e.format(s = s) for e in ESTIMATES]].to_numpy(dtype = float) for s in nt.SIDES],
                   axis = 1).astype(np.float32)                                           # (n, 2, 2)
    direction = nt.get_direction(task, stair_dir, cache['flag_l'].to_numpy(), cache['limb_power_l'].to_numpy(),
                                 cache['limb_power_r'].to_numpy())
    rows = []
    for leg, side in enumerate(nt.SIDES):
        ends = get_valid_ends(x[:, leg], cache[f'flag_{side}'].to_numpy() > 0, y[:, leg])
        rows.append(pd.DataFrame({'rec': np.int32(rec), 'end': ends.astype(np.int32), 'leg': np.int8(leg),
                                  'direction': direction[ends]}))
    return rec, x, y, est, pd.concat(rows, ignore_index = True)


def is_paper_train(task: pd.Series, direction: pd.Series) -> npt.NDArray[np.bool_]:
    """Rows of the paper's 19 selected tasks, in the kept direction where the folder holds two."""
    keep = [t in PAPER_TASKS and PAPER_TASKS[t] in (None, d)
            for t, d in zip(task.to_numpy(dtype = object), direction.to_numpy(dtype = object))]
    return np.asarray(keep, dtype = bool)


def build(out: Path = OUT, workers: int = WORKERS) -> pd.DataFrame:
    """Write inputs.npy, targets.npy, estimate.npy, windows.parquet and meta.json; returns the window table."""
    inventory = pd.read_parquet(nt.CACHE / 'samples_inventory.parquet')
    split = ad.get_participant_split(inventory)
    offsets = np.r_[0, np.cumsum(inventory['n_samples'].to_numpy())]
    out.mkdir(parents = True, exist_ok = True)
    shape = (int(offsets[-1]), len(nt.SIDES))
    arrays = {name: np.lib.format.open_memmap(out / f'{name}.npy', mode = 'w+', dtype = np.float32, shape = (*shape, n))
              for name, n in (('inputs', len(CHANNELS)), ('targets', 2), ('estimate', 2))}
    tables = []
    jobs = [(int(r.rec), r.path, int(r.n_samples), r.task, r.stair_dir) for r in inventory.itertuples()]
    with ProcessPoolExecutor(workers) as pool:
        for rec, x, y, est, rows in pool.map(get_recording, jobs, chunksize = 4):
            for name, values in (('inputs', x), ('targets', y), ('estimate', est)):
                arrays[name][offsets[rec]:offsets[rec + 1]] = values
            tables.append(rows.assign(start = offsets[rec] + rows['end'].to_numpy(np.int64) - (WINDOW - 1)))
    for a in arrays.values():
        a.flush()
    del arrays
    windows = pd.concat(tables, ignore_index = True).merge(inventory[['rec', 'subject', 'phase', 'task']], on = 'rec')
    windows = windows.sort_values(['rec', 'leg', 'end'], kind = 'stable').reset_index(drop = True)
    windows['controller_class'] = windows['task'].map(nt.get_controller_class)
    windows['split'] = windows['subject'].map(split)
    windows['paper_train'] = is_paper_train(windows['task'], windows['direction'])
    for c in ('subject', 'phase', 'task', 'direction', 'controller_class', 'split'):
        windows[c] = windows[c].astype('category')
    windows = windows[['start', 'rec', 'end', 'leg', 'subject', 'phase', 'task', 'direction', 'controller_class', 'split',
                       'paper_train']]
    windows.to_parquet(out / 'windows.parquet', index = False)
    meta = {'channels': [c.format(s = '<leg>') for c in CHANNELS], 'mirrored_on_left': [c.format(s = '<leg>') for c in MIRRORED],
            'legs': list(nt.SIDES), 'targets': ['total hip moment (t - hip_delay)', 'total knee moment (t)'],
            'units': 'N m/kg, OpenSim signs', 'window': WINDOW, 'stride': STRIDE, 'hip_delay': HIP_DELAY, 'rate_hz': nt.RATE_HZ,
            'paper_tasks': PAPER_TASKS, 'split_seed': ad.SPLIT_SEED,
            'split': {k: sorted(p for p, s in split.items() if s == k) for k in ('train', 'val', 'test')},
            'n_samples': int(offsets[-1]),
            'data_version': ad.get_data_version(*(out / f for f in ('inputs.npy', 'targets.npy', 'windows.parquet')))}
    (out / 'meta.json').write_text(json.dumps(meta, indent = 2))
    return windows


if __name__ == '__main__':
    table = build()
    print(f'{len(table):,} windows -> {OUT} (version {json.loads((OUT / "meta.json").read_text())["data_version"]})')
    print(table.pivot_table(index = 'paper_train', columns = 'split', values = 'start', aggfunc = 'size', observed = True))
