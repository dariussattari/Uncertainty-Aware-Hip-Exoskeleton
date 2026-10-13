"""Training arrays for the Nature awareness autoencoders, built once on the laptop from the sample cache.

    python ml/nature/awareness/ae_data.py          # needs data/cache/nature/samples (nature notebook 01)

Writes ``data/processed_nature/ae/``:

  signals.npy      float32 (n_samples, 16): every recording end to end, in the Science Robotics pipeline's channel
                   order (per side: thigh accel xyz, thigh gyro xyz, hip angle, hip velocity). IMU axes are remapped
                   (pipeline x, y, z = Nature y, z, x; signs as published) and hip velocity is recomputed causally
                   from the angle, because Nature's published velocity is half scale in most sessions.
  windows.parquet  one row per 1 s window (stride 10) with no missing sample: ``start`` (its first row in
                   signals.npy), recording, participant, task, in-task flag, family, controller class, direction,
                   Science Robotics mode, and the participant split.
  windows_raw_3s_spectral.parquet
                   the same for 3 s windows ending at the same samples (those with 3 s of clean history), plus the
                   seven spectral features at the window end from the Nature feature cache (nature notebook 02): the
                   input setting the boosted-tree probe and the PCA / UMAP ablation both found best (notebooks 02, 03).
  meta.json        channel names and sources, the input settings (table, window, features), stride, the split.

Windows are kept continuous rather than materialized (1.9 M windows x 16 x 200 would be 24 GB); the trainer
gathers them from ``signals.npy`` by ``start``.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # ml/nature
import nature_data as nt  # noqa: E402
import features as ft  # noqa: E402  (ml/common, put on the path by nature_data)

OUT = nt.REPO / 'data' / 'processed_nature' / 'ae'
WINDOW = 200                       # samples (1 s), as in the Science Robotics pipeline
STRIDE = 10                        # samples between window ends (50 ms)
SPLIT_SEED = 0
N_TEST, N_VAL = 2, 2               # participants per stratum (with and without a Phase 3 session)
# (pipeline channel, Nature source column) per side; velocity is recomputed from the angle
CHANNELS = [(f'thigh_{kind}_{p}_{{s}}', f'thigh_imu_{{s}}_{kind}_{n}')
            for kind in ('accel', 'gyro') for p, n in zip('xyz', nt.NATURE_ANATOMICAL)] \
    + [('enc_angle_{s}', 'hip_angle_{s}'), ('enc_velo_{s}', 'hip_angle_{s} (causal derivative)')]
CHANNEL_NAMES = [name.format(s = s) for s in nt.SIDES for name, _ in CHANNELS]
CHANNEL_SOURCES = [source.format(s = s) for s in nt.SIDES for _, source in CHANNELS]
SPECTRAL = ['dominant_hz_l', 'dominant_hz_r', 'spectral_entropy_l', 'spectral_entropy_r', 'stride_regularity_l',
            'stride_regularity_r', 'step_regularity']
# input settings: window (samples) of the 16 raw channels, and per-window features fed as constant extra channels
INPUTS = {'raw_1s': {'table': 'windows.parquet', 'window': WINDOW, 'features': []},
          'raw_3s_spectral': {'table': 'windows_raw_3s_spectral.parquet', 'window': 3 * WINDOW, 'features': SPECTRAL}}


def get_valid_ends(x: npt.NDArray[np.float32], window: int = WINDOW, stride: int = STRIDE) -> npt.NDArray[np.int64]:
    """Window ends (every ``stride``, from ``window - 1``) whose ``window`` samples are finite in every channel."""
    bad = np.r_[0, np.cumsum(~np.isfinite(x).all(axis = 1))]               # (n + 1,) running count of bad samples
    ends = np.arange(window - 1, len(x), stride)
    return ends[(bad[ends + 1] - bad[ends + 1 - window]) == 0]


def get_participant_split(inventory: pd.DataFrame, seed: int = SPLIT_SEED) -> dict[str, str]:
    """Participant -> 'train' / 'val' / 'test', stratified by whether the person has a Phase 3 session.

    BT01, BT02 and BT13 appear in both phases, so the split is by person, never by session.
    """
    rng = np.random.default_rng(seed)
    has_phase3 = inventory.groupby('subject')['phase'].agg(lambda p: 'phase3' in set(p))
    split = {}
    for _, people in has_phase3.groupby(has_phase3):
        order = rng.permutation(sorted(people.index))
        for i, person in enumerate(order):
            split[person] = 'test' if i < N_TEST else 'val' if i < N_TEST + N_VAL else 'train'
    return split


def get_recording(path: Path) -> tuple[npt.NDArray[np.float32], pd.DataFrame]:
    """One cached recording's 16 inputs (pipeline order) and the columns its labels need."""
    df = pd.read_parquet(path)
    columns = []
    for s in nt.SIDES:
        for name, source in CHANNELS:
            if name.startswith('enc_velo'):
                columns.append(ft.get_causal_velocity(df[f'hip_angle_{s}'].to_numpy(dtype = float)))
            else:
                columns.append(df[source.format(s = s)].to_numpy(dtype = float))
    x = np.stack(columns, axis = 1).astype(np.float32)                    # (n, 16)
    return x, df[['flag_l', 'limb_power_l', 'limb_power_r']]


def build_arrays(out: Path = OUT) -> pd.DataFrame:
    """Write signals.npy, the window table of every input setting and meta.json; returns the 1 s table."""
    nt.build_feature_cache(stride = STRIDE)                           # current spectral features (rebuilt if stale)
    inventory = pd.read_parquet(nt.CACHE / 'samples_inventory.parquet')
    split = get_participant_split(inventory)
    out.mkdir(parents = True, exist_ok = True)
    signals = np.lib.format.open_memmap(out / 'signals.npy', mode = 'w+', dtype = np.float32,
                                        shape = (int(inventory['n_samples'].sum()), len(CHANNEL_NAMES)))
    tables, long_ends, offset = [], [], 0
    long_window = INPUTS['raw_3s_spectral']['window']
    for r in inventory.itertuples():
        x, labels = get_recording(nt.CACHE / 'samples' / f'{r.rec:05d}.parquet')
        signals[offset:offset + len(x)] = x
        ends = get_valid_ends(x)
        long_ends.append(pd.DataFrame({'rec': np.int32(r.rec), 'end': get_valid_ends(x, long_window).astype(np.int32)}))
        direction = nt.get_direction(r.task, r.stair_dir, labels['flag_l'].to_numpy(),
                                     labels['limb_power_l'].to_numpy(), labels['limb_power_r'].to_numpy())[ends]
        tables.append(pd.DataFrame({'start': offset + ends - (WINDOW - 1), 'rec': np.int32(r.rec), 'end': ends.astype(np.int32),
                                    'subject': r.subject, 'phase': r.phase, 'task': r.task,
                                    'in_task': labels['flag_l'].to_numpy()[ends] > 0, 'direction': direction}))
        offset += len(x)
    signals.flush()
    windows = pd.concat(tables, ignore_index = True)
    windows['family'] = windows['task'].map(nt.get_family)
    windows['controller_class'] = windows['task'].map(nt.get_controller_class)
    windows['scirob_mode'] = [nt.SCIROB_MODE.get(key, 'Nature only') for key in zip(windows['task'], windows['direction'])]
    windows['split'] = windows['subject'].map(split)
    for c in ('subject', 'phase', 'task', 'direction', 'family', 'controller_class', 'scirob_mode', 'split'):
        windows[c] = windows[c].astype('category')
    windows.to_parquet(out / INPUTS['raw_1s']['table'], index = False)
    build_long_table(windows, pd.concat(long_ends, ignore_index = True), long_window).to_parquet(
        out / INPUTS['raw_3s_spectral']['table'], index = False)
    meta = {'channels': CHANNEL_NAMES, 'sources': CHANNEL_SOURCES, 'window': WINDOW, 'stride': STRIDE, 'rate_hz': nt.RATE_HZ,
            'inputs': INPUTS, 'split_seed': SPLIT_SEED,
            'split': {k: sorted(p for p, s in split.items() if s == k) for k in ('train', 'val', 'test')},
            'n_samples': int(offset), 'id_tasks': list(nt.ID_SAFE_TASKS),
            'data_version': get_data_version(out / 'signals.npy', *(out / i['table'] for i in INPUTS.values()))}
    (out / 'meta.json').write_text(json.dumps(meta, indent = 2))
    return windows


def build_long_table(windows: pd.DataFrame, long_ends: pd.DataFrame, window: int) -> pd.DataFrame:
    """The 3 s windows: every 1 s window whose ``window`` samples are all clean, its labels, and its spectral features.

    Each ends at the same sample as its 1 s window, so labels (taken at the window end) carry over and the two settings
    can be compared window by window. Windows whose spectral features are missing (a flat hip angle) are dropped.
    """
    table = windows.merge(long_ends, on = ['rec', 'end'], how = 'inner', validate = 'one_to_one')
    table['start'] = table['start'] - (window - WINDOW)
    spectral = pd.read_parquet(nt.CACHE / 'features', columns = ['rec', 'end', *SPECTRAL])
    table = table.merge(spectral, on = ['rec', 'end'], how = 'left', validate = 'one_to_one')
    return table.dropna(subset = SPECTRAL).reset_index(drop = True)


def get_data_version(*paths: Path) -> str:
    """Content hash of the array files, logged beside every result trained on them."""
    digest = hashlib.sha256()
    for path in paths:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''):
                digest.update(chunk)
    return digest.hexdigest()[:16]


if __name__ == '__main__':
    table = build_arrays()
    long = pd.read_parquet(OUT / INPUTS['raw_3s_spectral']['table'])
    print(f'3 s + spectral: {len(long):,} windows ({len(long) / len(table):.0%} of the 1 s windows)')
    print(f'{len(table):,} windows ({table.in_task.mean():.0%} in task) -> {OUT}')
    print(table[table.in_task].pivot_table(index = 'controller_class', columns = 'split', values = 'start', aggfunc = 'size',
                                           observed = True))
    print(json.loads((OUT / 'meta.json').read_text())['split'])
