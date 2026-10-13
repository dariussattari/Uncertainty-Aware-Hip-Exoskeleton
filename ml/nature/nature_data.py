"""Data access for the Nature 2024 dataset, and for the Science Robotics study pool it is compared with.

Nature 2024: Molinaro et al., *Task-agnostic exoskeleton control via biological joint moment
estimation*, dataset doi:10.35090/gatech/75759, Complete variant under ``data/nature/{phase1-2,phase3}``.
Science Robotics 2024: Molinaro, Kang & Young, Zenodo 10849318, under ``data/molinaro-raw``; only the
study pool the models use (phase 3-4 custom hardware, complete trials, the seven modes).

Read-only on the raw files. ``build_cache`` / ``build_scirob_cache`` copy the columns the notebook needs
into one Parquet file per recording under ``data/cache/nature/`` (float32, native units and sign
conventions, nothing filtered or resampled), so the analysis loads in seconds instead of re-parsing
88 GB of CSV.

Conventions kept as published:
  Nature exo signals (hip angle, velocity, torques)  extension positive, deg, deg/s, Nm (estimate Nm/kg)
  Nature OpenSim signals (moments, powers, angles)   hip FLEXION positive, Nm/kg, W/kg, deg
  Science Robotics (all)                            extension positive, rad, rad/s, Nm/kg
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import urllib.request
import zipfile
import zlib

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'data').is_dir())
if str(REPO / 'ml' / 'common') not in sys.path:
    sys.path.insert(0, str(REPO / 'ml' / 'common'))
import features as ft  # noqa: E402  (ml/common/features.py: the engineered-feature definitions)
import signals as sg  # noqa: E402  (ml/common/signals.py: bouts)
NATURE = REPO / 'data' / 'nature'
SCIROB = REPO / 'data' / 'molinaro-raw'
CACHE = REPO / 'data' / 'cache' / 'nature'
RATE_HZ = 200
SIDES = ('l', 'r')
PHASES = {'phase1-2': 'Phase1And2_Complete.zip', 'phase3': 'Phase3_Complete.zip'}
BITSTREAMS_API = ('https://repository.gatech.edu/server/api/core/bundles/'
                  '6ad2308e-56ae-4730-ab40-0fd162e4ccc8/bitstreams?size=100')
HTTP_HEADERS = {'User-Agent': 'Mozilla/5.0'}

# The existing pipeline's 16 inputs, per side: accel xyz, gyro xyz, hip angle, hip velocity.
SCIROB_16 = [c for s in SIDES for c in ([f'thigh_accel_{a}_{s}' for a in 'xyz'] + [f'thigh_gyro_{a}_{s}' for a in 'xyz']
                                        + [f'enc_angle_{s}', f'enc_velo_{s}'])]
NATURE_16 = [c for s in SIDES for c in ([f'thigh_imu_{s}_accel_{a}' for a in 'xyz'] + [f'thigh_imu_{s}_gyro_{a}' for a in 'xyz']
                                        + [f'hip_angle_{s}', f'hip_angle_{s}_velocity'])]

# source file -> {source column: cache column}
NATURE_FILES = {
    'Activity_Flag': {'left': 'flag_l', 'right': 'flag_r'},
    'Exo': {**{c: c for c in NATURE_16},
            **{f'hip_angle_{s}_velocity_filt': f'hip_angle_{s}_velocity_filt' for s in SIDES},
            **{f'hip_angle_{s}_torque_{k}': f'tq_{t}_{s}' for s in SIDES
               for k, t in (('measured', 'meas'), ('estimated', 'est'), ('interaction', 'int'))}},
    'Joint_Moments_Filt': {f'hip_flexion_{s}_moment': f'net_{s}' for s in SIDES},
    'Joint_Moments': {f'hip_flexion_{s}_moment': f'net_raw_{s}' for s in SIDES},
    'Bio_Joint_Moments': {f'hip_flexion_{s}_moment': f'bio_{s}' for s in SIDES},
    'Joint_Angles_Filt': {f'hip_flexion_{s}': f'hip_flexion_{s}' for s in SIDES},
    'Net_Joint_Power': {f'{j}_{s}_power': f'_{j}_power_{s}' for s in SIDES
                        for j in ('hip_flexion', 'knee_angle', 'ankle_angle')},
}
SCIROB_FILES = {
    'exo': {**{c: c for c in SCIROB_16}, **{f'trq_{k}_{s}': f'trq_{k}_{s}' for s in SIDES for k in ('mea', 'est')}},
    'moment': {f'hip_moment_{s}': f'moment_{s}' for s in SIDES},
    'angle': {f'hip_angle_{s}': f'hip_angle_{s}' for s in SIDES},
}
SCIROB_MODES = ('LG', 'RA', 'RD', 'SA', 'SD', 'ST', 'TR')


# ----------------------------------------------------------------------------- labels

def parse_task(recording):
    """Base task of a Complete recording folder, e.g. ``stairs_2_10`` -> ``stairs``."""
    return re.sub(r'(_\d+)+$', '', recording)


def get_stair_direction(recording):
    """Stair recordings alternate ascent and descent by their last index (readme: _1_up, _2_down, ...)."""
    result = None
    if parse_task(recording) == 'stairs':
        result = 'up' if int(recording.rsplit('_', 1)[1]) % 2 else 'down'
    return result


# Task taxonomy used across the Nature notebooks. Controller classes follow notebook 01's estimator-benchmark rule
# (section 5.5) and are provisional: its thresholds are a choice.
TASK_FAMILIES = {'walking': ('normal_walk', 'incline_walk', 'walk_backward', 'weighted_walk', 'dynamic_walk', 'obstacle_walk'),
                 'stairs / curbs / step-ups': ('stairs', 'curb_up', 'curb_down', 'step_ups'),
                 'squat / lift / sit / lunge': ('squats', 'lift_weight', 'sit_to_stand', 'lunges'),
                 'jump / run / cut / shuffle': ('jump', 'tire_run', 'cutting', 'side_shuffle'),
                 'standing / unstructured': ('poses', 'tug_of_war', 'push', 'twister', 'meander', 'start_stop',
                                             'turn_and_step', 'ball_toss')}
ID_SAFE_TASKS = ('normal_walk', 'incline_walk', 'stairs', 'weighted_walk')
OOD_DANGEROUS_TASKS = ('ball_toss', 'step_ups', 'tug_of_war', 'sit_to_stand')
# The Science Robotics mode a Nature task (and direction) corresponds to (notebook 01, section 2); others are new.
SCIROB_MODE = {('normal_walk', 'level'): 'LG', ('incline_walk', 'up'): 'RA', ('incline_walk', 'down'): 'RD',
               ('stairs', 'up'): 'SA', ('stairs', 'down'): 'SD', ('curb_up', 'up'): 'SA', ('curb_down', 'down'): 'SD',
               ('start_stop', 'none'): 'TR', ('poses', 'none'): 'ST'}
INCLINE_POWER_W_KG = 0.1          # mean limb power (both legs) beyond which a ramp bout is ascent or descent
BOUT_MIN_S = 2.0                  # in-task bouts shorter than this are too short to judge a direction


def get_family(task: str) -> str:
    """Task family of a Nature task."""
    return next(family for family, tasks in TASK_FAMILIES.items() if task in tasks)


def get_controller_class(task: str) -> str:
    """'ID-safe' (cyclic: spline controller), 'OOD-dangerous' (exo off) or 'OOD-safe' (moment estimator)."""
    result = 'OOD-safe'
    if task in ID_SAFE_TASKS:
        result = 'ID-safe'
    elif task in OOD_DANGEROUS_TASKS:
        result = 'OOD-dangerous'
    return result


def get_direction(task: str, stair_dir: str | None, flag: np.ndarray, limb_power_l: np.ndarray,
                  limb_power_r: np.ndarray) -> np.ndarray:
    """Per-sample movement direction of one recording: 'up', 'down', 'level', 'unclear' or 'none'.

    Stairs take the readme's index parity, curbs their task name, level treadmill walking is 'level'. A ramp's
    direction is not labelled, so each in-task bout (left flag, at least ``BOUT_MIN_S``) is 'up' or 'down' by the sign
    of its mean limb power (W/kg, both legs; notebook 01, section 4) and 'unclear' in between or outside a bout.
    Every other task is 'none'.
    """
    n = len(flag)
    fixed = {'stairs': stair_dir, 'curb_up': 'up', 'curb_down': 'down', 'normal_walk': 'level'}
    result = np.full(n, fixed.get(task) or 'none', dtype = object)
    if task == 'incline_walk':
        result[:] = 'unclear'
        legs = np.stack([limb_power_l, limb_power_r])                                  # (2, n)
        count = np.isfinite(legs).sum(axis = 0)
        power = np.where(count > 0, np.nansum(legs, axis = 0) / np.maximum(count, 1), np.nan)   # (n,)
        for start, end in sg.get_bouts(np.asarray(flag) > 0, int(BOUT_MIN_S * RATE_HZ)):
            mean = np.nanmean(power[start:end]) if np.isfinite(power[start:end]).any() else 0.0
            if abs(mean) > INCLINE_POWER_W_KG:
                result[start:end] = 'up' if mean > 0 else 'down'
    return result


# ----------------------------------------------------------------------------- reading

def read_columns(folder, files, extension = '.csv'):
    """One frame from several per-recording CSVs that share a time axis; raises if the axes disagree."""
    folder = Path(folder)
    frames, time = [], None
    for name, columns in files.items():
        df = pd.read_csv(folder / f'{name}{extension}', usecols = ['time', *columns])
        if time is None:
            time = df['time'].to_numpy()
        if len(df) != len(time) or not np.allclose(df['time'].to_numpy(), time, atol = 1e-6, equal_nan = True):
            raise ValueError(f'{folder}: {name} does not share the time axis of the other files')
        frames.append(df[list(columns)].rename(columns = columns))
    return pd.concat(frames, axis = 1).astype(np.float32)


def read_recording(folder):
    """The cache columns of one Nature recording; the three joint powers are summed into ``limb_power``."""
    df = read_columns(folder, NATURE_FILES)
    for s in SIDES:
        parts = [f'_{j}_power_{s}' for j in ('hip_flexion', 'knee_angle', 'ankle_angle')]
        df[f'hip_power_{s}'] = df[parts[0]]
        df[f'limb_power_{s}'] = df[parts].sum(axis = 1, min_count = 3)
    df = df.drop(columns = [c for c in df if c.startswith('_')])
    df[['flag_l', 'flag_r']] = df[['flag_l', 'flag_r']].fillna(0)
    return df


def read_scirob_trial(folder):
    return read_columns(folder, SCIROB_FILES)


# ----------------------------------------------------------------------------- inventories

def build_inventory():
    """Every Nature recording folder, in a stable order; ``rec`` is the cache key."""
    rows = []
    for phase in PHASES:
        for folder in sorted(p for p in (NATURE / phase).glob('BT*/*') if p.is_dir()):
            rows.append(dict(phase = phase, subject = folder.parent.name, recording = folder.name,
                             task = parse_task(folder.name), stair_dir = get_stair_direction(folder.name),
                             path = str(folder.relative_to(NATURE))))
    inventory = pd.DataFrame(rows)
    inventory.insert(0, 'rec', np.arange(len(inventory), dtype = np.int32))
    return inventory


def build_scirob_inventory():
    """The Science Robotics study pool: phase 3-4 custom hardware, complete trials, the seven modes."""
    inv = pd.read_parquet(REPO / 'data' / 'cache' / 'inventory.parquet')
    pool = inv.query("complete and phase in [3, 4] and exo_hardware == 'custom'")
    pool = pool[pool['mode'].isin(SCIROB_MODES)].sort_values(['subject', 'trial']).reset_index(drop = True)
    pool = pool[['subject', 'trial', 'mode', 'condition', 'speed', 'controller', 'phase']].copy()
    for c in ('subject', 'trial', 'mode', 'controller'):
        pool[c] = pool[c].astype(str)
    pool['path'] = pool['subject'] + '/' + pool['trial']
    pool.insert(0, 'rec', np.arange(len(pool), dtype = np.int32))
    return pool


# ----------------------------------------------------------------------------- caches

def write_cache_file(job):
    """Worker: read one recording, write it as Parquet with its ``rec`` id; returns (rec, rows)."""
    rec, folder, out, reader = job
    df = reader(folder)
    df.insert(0, 'rec', np.int32(rec))
    df.to_parquet(out, index = False, compression = 'zstd')
    return rec, len(df)


def build_cache_dir(inventory, root, reader, name, workers = 8, force = False):
    """Write one Parquet per inventory row under ``CACHE/name``; reuse it unless the inventory changed."""
    out = CACHE / name
    saved = CACHE / f'{name}_inventory.parquet'
    stale = saved.exists() and pd.read_parquet(saved)['path'].tolist() != inventory['path'].tolist()
    out.mkdir(parents = True, exist_ok = True)
    if force or stale:
        for f in out.glob('*.parquet'):
            f.unlink()
    jobs = [(r.rec, root / r.path, out / f'{r.rec:05d}.parquet', reader)
            for r in inventory.itertuples() if not (out / f'{r.rec:05d}.parquet').exists()]
    if jobs:
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(write_cache_file, jobs, chunksize = 4))
    rows = [pq.read_metadata(out / f'{r:05d}.parquet').num_rows for r in inventory['rec']]
    inventory = inventory.assign(n_samples = rows, seconds = np.asarray(rows) / RATE_HZ)
    inventory.to_parquet(saved, index = False)
    return inventory


def build_cache(workers = 8, force = False):
    """Nature cache (first run reads every recording: several minutes); returns the inventory with lengths."""
    return build_cache_dir(build_inventory(), NATURE, read_recording, 'samples', workers, force)


def build_scirob_cache(workers = 8, force = False):
    return build_cache_dir(build_scirob_inventory(), SCIROB, read_scirob_trial, 'scirob_samples', workers, force)


def load_samples(columns, recs = None, name = 'samples'):
    """Cached samples (always with ``rec``), optionally only for some recordings, in recording order."""
    dataset = ds.dataset(CACHE / name, format = 'parquet')
    cols = ['rec', *[c for c in columns if c != 'rec']]
    flt = None if recs is None else ds.field('rec').isin(np.asarray(list(recs), dtype = np.int32))
    df = dataset.to_table(columns = cols, filter = flt).to_pandas()
    return df.sort_values('rec', kind = 'stable').reset_index(drop = True)


# ----------------------------------------------------------------------------- engineered features (nature notebook 02)

# Nature thigh IMU axes in anatomical order (up, mediolateral flexion +, anterior): nature notebook 01, section 3.4.
NATURE_ANATOMICAL = ('y', 'z', 'x')
RAW_STATS = ('last', 'mean', 'sd', 'min', 'max')
FEATURE_SOURCES = (Path(__file__), REPO / 'ml' / 'common' / 'features.py')


def get_leg_signals(df, side):
    """One leg's hip angle and anatomical-order accelerometer and gyro from a cached Nature recording."""
    return {'angle': df[f'hip_angle_{side}'].to_numpy(dtype = float),
            'acc': df[[f'thigh_imu_{side}_accel_{a}' for a in NATURE_ANATOMICAL]].to_numpy(dtype = float),
            'gyr': df[[f'thigh_imu_{side}_gyro_{a}' for a in NATURE_ANATOMICAL]].to_numpy(dtype = float)}


def get_raw_summaries(df, ends, length = ft.WINDOW['short']):
    """Last value, mean, SD, min and max over each window of the 16 inputs, velocity recomputed from the angle."""
    raw = {}
    for c in NATURE_16:
        x = ft.get_causal_velocity(df[c.replace('_velocity', '')].to_numpy(dtype = float)) if c.endswith('_velocity') \
            else df[c].to_numpy(dtype = float)
        w = ft.get_windows(x, ends, length)
        for stat, value in zip(RAW_STATS, (w[:, -1], w.mean(axis = 1), w.std(axis = 1), w.min(axis = 1), w.max(axis = 1))):
            raw[f'raw_{c}_{stat}'] = value
    return pd.DataFrame(raw)


def write_feature_file(job):
    """Worker: engineered features, raw summaries and end-of-window labels for one recording."""
    rec, out, stride = job
    df = pd.read_parquet(CACHE / 'samples' / f'{rec:05d}.parquet')
    ends = np.arange(ft.WINDOW['short'] - 1, len(df), stride)
    table = pd.concat([pd.DataFrame({'rec': np.int32(rec), 'end': ends.astype(np.int32)}),
                       ft.get_window_features(get_leg_signals(df, 'l'), get_leg_signals(df, 'r'), ends),
                       get_raw_summaries(df, ends).astype(np.float32),
                       df[['flag_l', 'flag_r', 'bio_l', 'bio_r', 'net_l', 'net_r']].iloc[ends].reset_index(drop = True)], axis = 1)
    table.to_parquet(out, index = False, compression = 'zstd')
    return rec, len(table)


def build_feature_cache(stride = 10, workers = 8, force = False):
    """Feature table for every Nature recording (needs ``build_cache`` first); rebuilt when the feature code changes."""
    inventory = pd.read_parquet(CACHE / 'samples_inventory.parquet')
    out = CACHE / 'features'
    stamp = hashlib.sha256(b''.join(p.read_bytes() for p in FEATURE_SOURCES) + str(stride).encode()).hexdigest()[:16]
    stamp_file = CACHE / 'features_version.txt'          # beside, not inside, the Parquet folder
    stale = not stamp_file.exists() or stamp_file.read_text() != stamp
    out.mkdir(parents = True, exist_ok = True)
    if force or stale:
        for f in out.glob('*.parquet'):
            f.unlink()
    jobs = [(r, out / f'{r:05d}.parquet', stride) for r in inventory['rec'] if not (out / f'{r:05d}.parquet').exists()]
    if jobs:
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(write_feature_file, jobs, chunksize = 4))
    stamp_file.write_text(stamp)
    return pd.DataFrame({'rec': inventory['rec'], 'windows': [pq.read_metadata(out / f'{r:05d}.parquet').num_rows for r in inventory['rec']]})


# ----------------------------------------------------------------------------- download verification

class RangeFile:
    """Seekable read-only view of a remote file over HTTP range requests, enough for ``zipfile``."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence = 0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n = -1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        data = b''
        if end > self.pos:
            req = urllib.request.Request(self.url, headers = {**HTTP_HEADERS, 'Range': f'bytes={self.pos}-{end - 1}'})
            with urllib.request.urlopen(req) as r:
                data = r.read()
            self.pos += len(data)
        return data


def fetch_manifest(phase):
    """The published zip's file list (path below the participant folder, size, CRC32), read without downloading it."""
    req = urllib.request.Request(BITSTREAMS_API, headers = {**HTTP_HEADERS, 'Accept': 'application/json'})
    with urllib.request.urlopen(req) as r:
        streams = {b['name']: b for b in json.load(r)['_embedded']['bitstreams']}
    item = streams[PHASES[phase]]
    with zipfile.ZipFile(RangeFile(item['_links']['content']['href'], item['sizeBytes'])) as z:
        rows = [(i.filename, i.file_size, i.CRC) for i in z.infolist() if not i.is_dir()]
    m = pd.DataFrame(rows, columns = ['name', 'size', 'crc'])
    m = m[~m['name'].str.contains(r'__MACOSX|/\._|\.DS_Store', regex = True)]
    m['rel'] = m['name'].str.extract(r'(BT\d\d/.*)$')[0]
    return m[['rel', 'size', 'crc']].reset_index(drop = True)


def load_manifest(phase, refresh = False):
    path = CACHE / f'manifest_{phase}.parquet'
    if refresh or not path.exists():
        CACHE.mkdir(parents = True, exist_ok = True)
        fetch_manifest(phase).to_parquet(path, index = False)
    return pd.read_parquet(path)


def compare_with_manifest(phase):
    """Files missing from disk, extra on disk, and size mismatches against the published zip."""
    m = load_manifest(phase)
    root = NATURE / phase
    disk = {str(p.relative_to(root)): p.stat().st_size for p in root.rglob('*') if p.is_file() and p.name != '.DS_Store'}
    published = dict(zip(m['rel'], m['size']))
    common = set(published) & set(disk)
    return dict(phase = phase, published_files = len(published), files_on_disk = len(disk),
                missing = sorted(set(published) - set(disk)), extra = sorted(set(disk) - set(published)),
                size_mismatch = sorted(k for k in common if published[k] != disk[k]),
                published_gb = sum(published.values()) / 1e9, disk_gb = sum(disk.values()) / 1e9)


def get_crc(path):
    crc = 0
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            crc = zlib.crc32(block, crc)
    return crc


def verify_crc(phase, workers = 8):
    """CRC32 of every file on disk against the zip's (reads every byte: ~1 min per 30 GB)."""
    m = load_manifest(phase)
    paths = [str(NATURE / phase / r) for r in m['rel']]
    with ProcessPoolExecutor(workers) as ex:
        got = list(ex.map(get_crc, paths, chunksize = 16))
    return m.assign(crc_disk = got)[lambda d: d['crc'] != d['crc_disk']]
