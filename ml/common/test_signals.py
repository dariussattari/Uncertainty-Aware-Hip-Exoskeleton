"""The one check for signals.py: bout splitting, stride period, and the IMU frame solver on known signals."""
import numpy as np

import signals as sg


def check():
    np.testing.assert_array_equal(sg.get_bouts([0, 1, 1, 0, 1, 1, 1, 0, 1], min_len = 2), [[1, 3], [4, 7]])
    assert sg.get_bouts([], min_len = 1).shape == (0, 2)
    t = np.arange(20 * sg.RATE_HZ) / sg.RATE_HZ
    assert abs(sg.get_stride_period(np.sin(2 * np.pi * t / 1.1)) - 1.1) < 0.01
    lag, r = sg.get_lag_corr(np.sin(2 * np.pi * t), np.sin(2 * np.pi * (t - 0.05)), max_lag = 20)
    assert lag == 10 and r > 0.99
    # an IMU whose x is up and z the flexion axis, walking about z: the solver must return rows (+x, +z, x-cross)
    rng = np.random.default_rng(0)
    flex = np.sin(2 * np.pi * t)
    acc = np.tile([9.81, 0., 0.], (len(t), 1)) + rng.normal(0, .01, (len(t), 3))
    gyr = np.column_stack([rng.normal(0, .05, len(t)), rng.normal(0, .05, len(t)), flex])
    frame = sg.solve_imu_frame(acc, gyr, -flex, np.ones(len(t), bool), np.ones(len(t), bool))
    np.testing.assert_allclose(np.abs(frame), np.eye(3)[[0, 2, 1]], atol = .05)
    assert frame[0, 0] > 0 and frame[1, 2] > 0 and frame[2, 1] < 0      # anterior = up x ML = x x z = -y
    print('ok')


if __name__ == '__main__':
    check()
