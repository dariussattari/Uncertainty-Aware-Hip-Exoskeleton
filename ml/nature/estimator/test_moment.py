"""The one check for the moment estimator: the TCN's receptive field and causality, the left-leg mirroring, the hip
delay, the window rules, the task selection, the target gather and the thigh-IMU fit perturbation (B1)."""
import math

import numpy as np
import pandas as pd
import torch

import imu_fit
import moment_data as md
import train_moment as tm
from models.moment_tcn import MomentTCN, get_receptive_field


def check() -> None:
    # receptive field: 249 samples, the arrays' window. An input change older than that leaves the output alone, one
    # inside it does not, and the future never reaches the present (the output is at the last sample)
    assert get_receptive_field() == md.WINDOW == 249
    torch.manual_seed(0)
    model = MomentTCN(3).eval()
    x = torch.randn(1, 3, 300)
    base = model(x)
    early, late = x.clone(), x.clone()
    early[..., 300 - 250] += 10.0                                              # one sample beyond the receptive field
    late[..., 300 - 249] += 10.0                                               # the oldest sample inside it
    torch.testing.assert_close(model(early), base)
    assert not torch.allclose(model(late), base)
    # mirroring: the left leg's lateral accel, gyro x / y and COP z change sign, nothing else; velocity is recomputed
    n = 400
    exo = pd.DataFrame({c.format(s = s): np.full(n, 1.0) for s in 'lr' for c in md.CHANNELS if '(' not in c})
    for s in 'lr':
        exo[f'hip_angle_{s}'] = exo[f'knee_angle_{s}'] = np.arange(n) * 0.05     # 0.05 deg per sample = 10 deg/s
    left, right = md.get_leg_inputs(exo, 'l'), md.get_leg_inputs(exo, 'r')
    flipped = [md.CHANNELS.index(c) for c in md.MIRRORED]
    np.testing.assert_array_equal(left[:, flipped], -right[:, flipped])
    kept = [i for i in range(len(md.CHANNELS)) if i not in flipped]
    np.testing.assert_array_equal(left[:, kept], right[:, kept])
    np.testing.assert_allclose(left[-1, 1], 10.0, rtol = 1e-3)               # deg/s once the 10 Hz filter settles
    # hip delay: the target at t is the moment at t - 10
    shifted = md.shift_back(np.arange(20.0), md.HIP_DELAY)
    assert np.isnan(shifted[:10]).all() and shifted[15] == 5.0
    # window ends: every stride from window - 1, all inputs finite, flag set, a target defined
    x = np.ones((30, 2), np.float32)
    x[12, 1] = np.nan
    flag = np.ones(30, bool)
    flag[25] = False
    y = np.ones((30, 2), np.float32)
    y[27] = np.nan
    assert md.get_valid_ends(x, flag, y, window = 5, stride = 1).tolist() == [4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22,
                                                                               23, 24, 26, 28, 29]
    # task selection: whole folders, or one direction of a two-direction folder
    task = pd.Series(['jump', 'incline_walk', 'incline_walk', 'stairs', 'stairs', 'lunges'])
    direction = pd.Series(['none', 'up', 'down', 'up', 'down', 'none'])
    assert md.is_paper_train(task, direction).tolist() == [True, True, False, False, True, False]
    # target gather: the leg's target at the window's last sample, masked where undefined
    targets = torch.full((6, 2, 2), float('nan'))
    targets[3, 1] = torch.tensor([0.5, float('nan')])
    data = tm.MomentData(torch.zeros(6, 2, 1), targets, torch.zeros(6, 2, 2), torch.tensor([1]), torch.tensor([1]), pd.DataFrame(),
                         torch.tensor([0]), torch.tensor([0]), np.zeros(1, np.float32), np.ones(1, np.float32), ['c'], 3, 200.0, 'test')
    xb, yb, mask = tm.get_batch(data, torch.tensor([0]))
    assert xb.shape == (1, 1, 3) and yb.tolist() == [[0.5, 0.0]] and mask.tolist() == [[True, False]]
    # IMU fit: rotations are proper; a still fit changes nothing; spinning at a constant 2 rad/s about z (lateral), an IMU
    # 0.1 m further up the thigh (y) feels the centripetal -omega^2 r = -0.4 m/s^2 along y and no tangential term
    gen = torch.Generator().manual_seed(0)
    rot, lever = imu_fit.sample_fits(64, 15.0, 0.1, 0.05, gen, torch.device('cpu'))
    torch.testing.assert_close(rot @ rot.transpose(1, 2), torch.eye(3).expand(64, 3, 3), atol = 1e-5, rtol = 0)
    torch.testing.assert_close(torch.linalg.det(rot), torch.ones(64), atol = 1e-5, rtol = 0)
    angles = torch.rad2deg(torch.arccos(((rot.diagonal(dim1 = 1, dim2 = 2).sum(1) - 1) / 2).clamp(-1, 1)))
    assert angles.max() <= 15.0 + 1e-3 and (lever[:, 1].abs() <= 0.1).all() and (lever[:, [0, 2]].abs() <= 0.05).all()
    accel, gyro = torch.randn(1, 50, 3), torch.randn(1, 50, 3)
    a2, g2 = imu_fit.perturb_imu(accel, gyro, torch.eye(3)[None], torch.zeros(1, 3), 200.0)
    torch.testing.assert_close(a2, accel)
    torch.testing.assert_close(g2, gyro)
    spin = torch.zeros(1, 50, 3)
    spin[..., 2] = math.degrees(2.0)
    a2, _ = imu_fit.perturb_imu(torch.zeros(1, 50, 3), spin, torch.eye(3)[None], torch.tensor([[0.0, 0.1, 0.0]]), 200.0)
    torch.testing.assert_close(a2[0, 25], torch.tensor([0.0, -0.4, 0.0]), atol = 1e-5, rtol = 0)
    # a 90 deg turn about y maps x to -z: the same gyro reading is seen on another axis
    turn = imu_fit.get_rotation(torch.tensor([[0.0, math.pi / 2, 0.0]]))
    torch.testing.assert_close(turn[0] @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, -1.0]), atol = 1e-6, rtol = 0)
    print('ok')


if __name__ == '__main__':
    check()
