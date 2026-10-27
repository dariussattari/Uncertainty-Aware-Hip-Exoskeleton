"""Sci-Rob trials in B1's input conventions, for the Jetson replay (stage A).

    .venv/bin/python deployment/scirob_replay.py          # writes data/replay/scirob/AB*.npz (needs the Sci-Rob cache)

Pool: the Sci-Rob cache (``nature_data.build_scirob_cache``): 20 participants (AB15-AB34, the custom hip exo), 1505
trials, seven modes. None of them is in B1's training data. Decisions (moment sign, calibration variant) are made on
``DEV``; the other 16 participants are scored once.

Conventions, measured on walking in both datasets (encoder vs mocap angle, gravity, sagittal gyro vs encoder velocity):

    Sci-Rob            -> B1 (Nature)
    enc_angle  rad     -> hip angle deg (both extension-positive), velocity recomputed (features.get_causal_velocity)
    accel      m/s^2   -> m/s^2; Nature (x, y, z) = Sci-Rob (z, x, y): up = +x, mediolateral = +y, anterior = +z
    gyro       rad/s   -> deg/s, same axis map

Two variants of the thigh IMU rotation, both written:

    nominal      the axis map alone. Sci-Rob's left thigh IMU sits rolled ~23 deg about the anterior axis (right ~6 deg),
                 beyond B1's 15 deg training randomization.
    calibrated   the E41 deployment design: per person and leg, ``signals.solve_imu_frame`` on one walking trial (their
                 slowest level-ground trial; "up" is the mean accelerometer reading while walking, as no quiet standing
                 is recorded) gives the anatomical axes; they are rotated onto the average Nature IMU frame, solved the
                 same way on the Nature training participants' walking. v_nature = R_nature_mean^T R_person v_scirob.

Legs are written unmirrored (left, right); the replay runner applies B1's left-leg mirroring from moment.json, as the
device will. Ground truth and the online estimate ride along for the analysis: ``moment`` (inverse dynamics, N m/kg,
NaN off the force plates) and ``trq_est`` (the estimator that ran on the exo during collection).
"""
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'ml' / 'nature'))
import nature_data as nt  # noqa: E402  (puts ml/common on the path)
import features as ft  # noqa: E402
import signals as sg  # noqa: E402

OUT = REPO / 'data' / 'replay' / 'scirob'
DEV = ('AB15', 'AB19', 'AB25', 'AB29')                  # two per collection phase; decisions are made on these only
SCIROB_TO_NATURE = np.eye(3)[[2, 0, 1]]                 # rows: Nature (x, y, z) from Sci-Rob (z, x, y)
VARIANTS = ('nominal', 'calibrated')
CHANNELS = ('hip_angle_<leg>', 'hip_angle_<leg> (causal derivative)', *[f'thigh_imu_<leg>_{k}_{a}' for k in ('accel', 'gyro')
                                                                       for a in 'xyz'])
NATURE_TRAIN_SPLIT = REPO / 'data' / 'processed_nature' / 'moment' / 'windows.parquet'


def get_frame(acc: npt.NDArray, gyr: npt.NDArray, hip_velocity: npt.NDArray) -> npt.NDArray[np.float64]:
    """Anatomical axes (rows: up, mediolateral flexion-positive, anterior) in IMU coordinates from one walking bout.
    (n, 3), (n, 3), (n,) -> (3, 3)."""
    ok = np.isfinite(acc).all(axis = 1) & np.isfinite(gyr).all(axis = 1) & np.isfinite(hip_velocity)
    return sg.solve_imu_frame(acc, gyr, hip_velocity, ok, ok)


def get_mean_rotation(frames: list[npt.NDArray]) -> npt.NDArray[np.float64]:
    """Chordal mean of rotation matrices: the nearest rotation to their element-wise mean."""
    u, _, vt = np.linalg.svd(np.mean(frames, axis = 0))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        raise ValueError('mean frame is a reflection: the frames disagree in handedness')
    return rotation


def get_nature_frames() -> dict[str, npt.NDArray[np.float64]]:
    """Per leg, the average Nature thigh-IMU anatomical frame over the training participants' first walking recording."""
    windows = pd.read_parquet(NATURE_TRAIN_SPLIT, columns = ['subject', 'split'])
    train = set(windows.loc[windows['split'] == 'train', 'subject'].astype(str))
    inv = pd.read_parquet(nt.CACHE / 'samples_inventory.parquet')
    recs = inv[(inv['task'] == 'normal_walk') & inv['subject'].isin(train)].groupby('subject').head(1)['rec']
    df = nt.load_samples(nt.NATURE_16, recs)
    frames = {s: [] for s in nt.SIDES}
    for _, d in df.groupby('rec'):
        for s in nt.SIDES:
            frames[s].append(get_frame(d[[f'thigh_imu_{s}_accel_{a}' for a in 'xyz']].to_numpy(float),
                                       d[[f'thigh_imu_{s}_gyro_{a}' for a in 'xyz']].to_numpy(float),
                                       ft.get_causal_velocity(d[f'hip_angle_{s}'].to_numpy(float))))
    return {s: get_mean_rotation(f) for s, f in frames.items()}


def get_leg(d: pd.DataFrame, side: str) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """One leg's Sci-Rob encoder angle (rad), accel (m/s^2) and gyro (rad/s), IMU axes as recorded."""
    return (d[f'enc_angle_{side}'].to_numpy(float), d[[f'thigh_accel_{a}_{side}' for a in 'xyz']].to_numpy(float),
            d[[f'thigh_gyro_{a}_{side}' for a in 'xyz']].to_numpy(float))


def convert_leg(angle_rad: npt.NDArray, acc: npt.NDArray, gyr: npt.NDArray, rotation: npt.NDArray) -> npt.NDArray[np.float32]:
    """B1's 8 channels for one leg: hip angle (deg), its causal derivative (deg/s), thigh accel (m/s^2) and gyro (deg/s)
    in the Nature frame. ``rotation`` maps Sci-Rob IMU vectors to Nature ones. (n,), (n, 3), (n, 3) -> (n, 8)."""
    angle = np.rad2deg(angle_rad)
    return np.column_stack([angle, ft.get_causal_velocity(angle), acc @ rotation.T, np.rad2deg(gyr) @ rotation.T]).astype(np.float32)


def check_converted(x: npt.NDArray, label: str, nature_up: npt.NDArray | None) -> list[str]:
    """The converted walking bout must look like Nature's; raises if not, returns warnings.

    Fails unless the sagittal gyro (z) moves against the extension-positive encoder velocity, gravity is mostly on +y
    (nominal: within ~50 deg, since mounting roll is what calibration removes), and, calibrated, the mean gravity
    direction equals the average Nature "up" axis (cosine > 0.999). Warns when the gravity magnitude is off by more than
    5%: an accelerometer scale error the replay cannot fix.
    """
    ok = np.isfinite(x).all(axis = 1)
    g = x[ok, 2:5].mean(axis = 0)                                               # mean accel, Nature (x, y, z)
    r = sg.get_corr(x[ok, 7], x[ok, 1])
    problems = [f'sagittal gyro vs encoder velocity r = {r:.2f}'] if r > -0.5 else []
    problems += [f'gravity only {g[1]:.2f} m/s^2 on +y'] if g[1] < 0.6 * np.linalg.norm(g) else []
    if nature_up is not None and g @ nature_up / np.linalg.norm(g) < 0.999:
        problems.append(f'gravity direction {np.round(g / np.linalg.norm(g), 3)} vs Nature up {np.round(nature_up, 3)}')
    if problems:
        raise ValueError(f'{label}: converted signals do not match the Nature conventions: ' + '; '.join(problems))
    norm = float(np.linalg.norm(g))
    return [f'{label}: mean gravity {norm:.2f} m/s^2 (accelerometer scale?)'] if abs(norm - 9.81) > 0.05 * 9.81 else []


def main() -> None:
    inventory = pd.read_parquet(nt.CACHE / 'scirob_samples_inventory.parquet')
    nature = get_nature_frames()
    columns = [*nt.SCIROB_16, 'trq_est_l', 'trq_est_r', 'moment_l', 'moment_r']
    OUT.mkdir(parents = True, exist_ok = True)
    warnings = []
    for subject, trials in inventory.groupby('subject'):
        trials = trials.sort_values('rec')
        df = nt.load_samples(columns, trials['rec'], name = 'scirob_samples')
        lg = trials[trials['mode'] == 'LG']
        calib_rec = int(lg.sort_values(['speed', 'rec']).iloc[0]['rec'])
        calib = df[df['rec'] == calib_rec]
        rotations = {'nominal': {s: SCIROB_TO_NATURE for s in nt.SIDES}, 'calibrated': {}}
        for s in nt.SIDES:
            angle, acc, gyr = get_leg(calib, s)
            frame = get_frame(acc, gyr, ft.get_causal_velocity(angle))                       # person's anatomical frame
            rotations['calibrated'][s] = nature[s].T @ frame                                 # Sci-Rob IMU -> average Nature IMU
            warnings += check_converted(convert_leg(angle, acc, gyr, rotations['nominal'][s]), f'{subject} {s} nominal', None)
            check_converted(convert_leg(angle, acc, gyr, rotations['calibrated'][s]), f'{subject} {s} calibrated', nature[s][0])
        x = {v: [] for v in VARIANTS}
        moment, trq_est, starts, lengths = [], [], [], []
        offset = 0
        for rec, d in df.groupby('rec', sort = True):
            for v in VARIANTS:
                x[v].append(np.stack([convert_leg(*get_leg(d, s), rotations[v][s]) for s in nt.SIDES], axis = 1))  # (n, 2, 8)
            moment.append(d[['moment_l', 'moment_r']].to_numpy(np.float32))
            trq_est.append(d[['trq_est_l', 'trq_est_r']].to_numpy(np.float32))
            starts.append(offset)
            lengths.append(len(d))
            offset += len(d)
        meta = trials.set_index('rec').loc[sorted(df['rec'].unique())]
        np.savez(OUT / f'{subject}.npz', x_nominal = np.concatenate(x['nominal']), x_calibrated = np.concatenate(x['calibrated']),
                 moment = np.concatenate(moment), trq_est = np.concatenate(trq_est), rec = meta.index.to_numpy(np.int32),
                 start = np.array(starts, np.int64), length = np.array(lengths, np.int64),
                 mode = meta['mode'].to_numpy().astype(str), trial = meta['trial'].to_numpy().astype(str),
                 speed = meta['speed'].to_numpy(float), calibration_rec = np.int32(calib_rec),
                 rotation_calibrated = np.stack([rotations['calibrated'][s] for s in nt.SIDES]),
                 channels = np.array(CHANNELS), dev = np.bool_(subject in DEV))
        print(f'{subject}: {len(lengths)} trials, {offset / nt.RATE_HZ / 60:.1f} min, calibration trial {calib_rec}'
              f'{" (dev)" if subject in DEV else ""}')
    print('\n'.join(['warnings:', *warnings]) if warnings else 'no warnings')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
