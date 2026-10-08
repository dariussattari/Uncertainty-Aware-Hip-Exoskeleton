"""The one check for features.py: each family on signals whose answer is known, plus causality."""
import numpy as np

import features as ft


def check():
    rate = ft.RATE_HZ
    t = np.arange(12 * rate) / rate
    ends = np.arange(ft.WINDOW['spectral'] - 1, len(t), 10)
    # a clean gait-like cycle at 0.9 Hz: hip angle and its exact derivative
    angle = 20 * np.sin(2 * np.pi * 0.9 * t)
    polar = ft.get_polar_features(angle, np.gradient(angle) * rate, ends)
    assert np.allclose(polar['phase_rate_hz'], 0.9, atol = 0.03), polar['phase_rate_hz'][:3]
    assert polar['phase_monotonicity'].min() > 0.99 and polar['ring_cv'].max() < 0.02
    spec = ft.get_spectral_features(angle, ends)
    assert np.allclose(spec['dominant_hz'], 0.9, atol = 0.05) and spec['stride_regularity'].min() > 0.95
    assert abs(np.median(spec['stride_lag']) / rate - 1 / 0.9) < 0.02
    noise_entropy = ft.get_spectral_features(np.random.default_rng(0).normal(size = len(t)), ends)['spectral_entropy']
    assert np.median(noise_entropy) > 0.8 > spec['spectral_entropy'].max()
    # step regularity: a summed signal repeating every half stride scores ~1
    step = np.sin(2 * np.pi * 1.8 * t)
    assert ft.get_step_regularity(step, step, spec['stride_lag'], ends).min() > 0.9
    # high-frequency share: 30 Hz is all above the 10 Hz cutoff, 1 Hz none of it
    w = ft.get_windows(np.sin(2 * np.pi * 30 * t), ends, 200)
    assert ft.get_band_power_share(w, 10).min() > 0.95
    assert ft.get_band_power_share(ft.get_windows(np.sin(2 * np.pi * t), ends, 200), 10).max() < 0.05
    # jerk: constant acceleration has none; a linear ramp in one axis has exactly its slope
    still = np.tile([9.81, 0.0, 0.0], (len(t), 1))
    assert ft.get_jerk_features(still, np.zeros_like(still), ends)['jerk_rms'].max() == 0
    ramp = still + np.outer(t, [0.0, 0.0, 3.0])
    assert np.allclose(ft.get_jerk_features(ramp, np.zeros_like(still), ends)['jerk_peak'], 3.0)
    # pitch: gravity seen 30 degrees toward the anterior axis
    up, ant = np.full(len(t), 9.81 * np.cos(np.radians(30))), np.full(len(t), 9.81 * np.sin(np.radians(30)))
    assert np.allclose(ft.get_thigh_pitch(up, ant, ends[5:]), 30, atol = 0.1)
    assert np.allclose(ft.get_thigh_pitch(up, ant, ends[5:], gyro_ml = np.zeros(len(t))), 30, atol = 0.1)
    # fused pitch follows a swinging thigh whose accelerometer also feels a large swing acceleration
    psi = 25 * np.sin(2 * np.pi * t)
    swing = 6 * np.cos(2 * np.pi * t)                      # inertial acceleration on the anterior axis
    up_s, ant_s = 9.81 * np.cos(np.radians(psi)), 9.81 * np.sin(np.radians(psi)) + swing
    late = ends[ends > 6 * rate]                           # past the filter's start-up transient
    fused = np.abs(ft.get_thigh_pitch(up_s, ant_s, late, gyro_ml = np.gradient(psi) * rate) - psi[late]).max()
    gravity_only = np.abs(ft.get_thigh_pitch(up_s, ant_s, late) - psi[late]).max()
    assert fused < 0.5 * gravity_only, (fused, gravity_only)
    # causality: changing the future leaves every earlier window's features unchanged
    later = angle.copy()
    later[ends[10] + 1:] = 0
    a, b = ft.get_spectral_features(angle, ends[:11]), ft.get_spectral_features(later, ends[:11])
    assert all(np.allclose(a[k], b[k], equal_nan = True) for k in a)
    v1, v2 = ft.get_causal_velocity(angle), ft.get_causal_velocity(later)
    assert np.allclose(v1[: ends[10] + 1], v2[: ends[10] + 1])
    print('ok')


if __name__ == '__main__':
    check()
