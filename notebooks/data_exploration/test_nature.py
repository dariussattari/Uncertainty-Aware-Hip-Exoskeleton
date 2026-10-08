"""The one check for _nature.py: bout splitting, stair direction, and the shared-time-axis guard on a toy recording."""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import _nature as nt


def check():
    np.testing.assert_array_equal(nt.get_bouts([0, 1, 1, 0, 1, 1, 1, 0, 1], min_len = 2), [[1, 3], [4, 7]])
    assert nt.get_bouts([], min_len = 1).shape == (0, 2)
    assert [nt.get_stair_direction(r) for r in ('stairs_2_9', 'stairs_1_1_10', 'jump_1_1')] == ['up', 'down', None]
    t = np.arange(20 * nt.RATE_HZ) / nt.RATE_HZ
    assert abs(nt.get_stride_period(np.sin(2 * np.pi * t / 1.1)) - 1.1) < 0.01
    # an IMU whose x is up and z the flexion axis, walking about z: the solver must return rows (+x, +z, x-cross)
    rng = np.random.default_rng(0)
    flex = np.sin(2 * np.pi * t)
    acc = np.tile([9.81, 0., 0.], (len(t), 1)) + rng.normal(0, .01, (len(t), 3))
    gyr = np.column_stack([rng.normal(0, .05, len(t)), rng.normal(0, .05, len(t)), flex])
    frame = nt.solve_imu_frame(acc, gyr, -flex, np.ones(len(t), bool), np.ones(len(t), bool))
    np.testing.assert_allclose(np.abs(frame), np.eye(3)[[0, 2, 1]], atol = .05)
    assert frame[0, 0] > 0 and frame[1, 2] > 0 and frame[2, 1] < 0      # anterior = up x ML = x x z = -y
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        t = np.arange(5) / nt.RATE_HZ
        pd.DataFrame({'time': t, 'a': np.arange(5.)}).to_csv(folder / 'A.csv', index = False)
        pd.DataFrame({'time': t, 'b': -np.arange(5.)}).to_csv(folder / 'B.csv', index = False)
        df = nt.read_columns(folder, {'A': {'a': 'x'}, 'B': {'b': 'y'}})
        assert list(df) == ['x', 'y'] and df['y'].iloc[-1] == -4
        pd.DataFrame({'time': t + 1, 'b': np.zeros(5)}).to_csv(folder / 'B.csv', index = False)
        failed = False
        try:
            nt.read_columns(folder, {'A': {'a': 'x'}, 'B': {'b': 'y'}})
        except ValueError:
            failed = True
        assert failed, 'misaligned time axes must raise'
    print('ok')


if __name__ == '__main__':
    check()
