"""Hand-engineered window features for hip-exoskeleton sensor data, beside the raw channels.

Every feature is evaluated at a window end ``e`` from samples ``<= e`` only, so each one can run causally on
the device. Four families, after the thesis plan:

  jerk and high-frequency content  RMS and peak jerk of the thigh acceleration vector; share of acceleration
                                   and angular-rate power above 10 Hz; RMS angular acceleration.   1 s history
  polar phase space                hip angle against hip velocity, each normalized over the window: phase at
                                   the window end (sin, cos), phase rate (cadence), share of samples where the
                                   phase advances (monotonicity), variation of the radius (ring CV). 2 s history
                                   (both signals scaled to their midrange and half-range)
  spectral                         dominant frequency and spectral entropy of the hip angle; stride regularity
                                   (hip-angle autocorrelation at one stride); step regularity (autocorrelation
                                   of both thighs' summed long-axis acceleration at half a stride).  3 s history
  orientation                      thigh pitch: gravity direction fused with the sagittal gyro
                                   (complementary filter, 1 s), so limb acceleration does not corrupt it. instant

Inputs are 1-D (or n x 3) arrays at ``RATE_HZ``: angles in degrees, angular rates in deg/s, accelerations in
m/s^2, IMU axes already named anatomically (up = along the thigh, mediolateral, anterior). A window with any
missing sample gives NaN.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, lfilter

RATE_HZ = 200
# Causal window lengths: Would be interesting to look into changing this
WINDOW = {'short': 200, 'polar': 400, 'spectral': 600}       # samples of history per family
HF_CUTOFF_HZ = 10.0                                         # human movement sits almost entirely below this

# Frequency band (Need Sources to backup how we came to this)
STRIDE_BAND_HZ = (0.3, 3.0)                                 # where a stride or step frequency can be
STRIDE_LAG_S = (0.5, 1.6)                                   # stride periods searched by the autocorrelation
ENTROPY_BAND_HZ = (0.25, 20.0)

PITCH_CUTOFF_HZ = 2.0                                       # gravity-only pitch: keeps gravity, drops some limb acceleration
PITCH_TAU_S = 1.0                                           # complementary filter: gyro below ~1 s, gravity above

N_FFT = 4096                                                # zero padding for the dominant-frequency peak


# ----------------------------------------------------------------------------- building blocks

def get_windows(x, ends, length):
    """Rows ``x[e - length + 1 : e + 1]`` for every end ``e``; rows reaching before the signal start are NaN.

    A 1-D ``x`` gives shape (len(ends), length); an (n, d) ``x`` gives (len(ends), d, length).
    """
    x = np.asarray(x, dtype = float)
    padded = np.concatenate([np.full((length - 1,) + x.shape[1:], np.nan), x])
    return sliding_window_view(padded, length, axis = 0)[np.asarray(ends, dtype = int)]


def fill_gaps(x):
    """Carry the last valid value forward (backward at the very start) so causal filters can run through gaps."""
    return pd.DataFrame(np.asarray(x, dtype = float)).ffill().bfill().to_numpy().reshape(np.shape(x))


def get_causal_velocity(angle, rate = RATE_HZ, cutoff_hz = 10.0):
    """Backward difference through a causal 2nd-order Butterworth: the previous dataset's ``enc_velo``.

    Recomputed rather than read from the Nature files, whose velocity channel is half scale in most sessions.
    """
    angle = np.asarray(angle, dtype = float)
    b, a = butter(2, cutoff_hz, fs = rate)
    velocity = lfilter(b, a, np.r_[0.0, np.diff(fill_gaps(angle))] * rate)
    velocity[~np.isfinite(angle)] = np.nan
    return velocity


def get_overlap_autocorrelation(rows):
    """Per-row autocorrelation at every lag, each lag normalized by the energy of its two overlapping segments.

    r(L) = sum x_t x_(t+L) / sqrt(sum_(t < n-L) x_t^2 * sum_(t >= L) x_t^2), mean removed over the row: bounded in
    [-1, 1] and exactly 1 at the period of a periodic signal, where the classic unbiased estimate is not.
    """
    n = rows.shape[-1]
    x = rows - rows.mean(axis = -1, keepdims = True)
    lagged = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * n, axis = -1)) ** 2, axis = -1)[..., :n]
    energy = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x ** 2, axis = -1)], axis = -1)
    lags = np.arange(n)
    head = energy[..., n - lags]
    tail = energy[..., n:n + 1] - energy[..., lags]
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        result = lagged / np.sqrt(head * tail)
    return result


def get_band_power_share(rows, cutoff_hz, rate = RATE_HZ):
    """Share of each row's power above ``cutoff_hz`` (mean removed, Hann taper, DC excluded)."""
    length = rows.shape[-1]
    tapered = (rows - rows.mean(axis = -1, keepdims = True)) * np.hanning(length)
    power = np.abs(np.fft.rfft(tapered, axis = -1)) ** 2
    freq = np.fft.rfftfreq(length, 1 / rate)
    total = power[:, 1:].sum(axis = 1)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        result = np.where(total > 0, power[:, freq > cutoff_hz].sum(axis = 1) / total, np.nan)
    return result


# ----------------------------------------------------------------------------- 1. jerk and high-frequency content

def get_jerk_features(acc, gyr, ends, length = WINDOW['short'], rate = RATE_HZ, cutoff_hz = HF_CUTOFF_HZ):
    """RMS and peak jerk (m/s^3), acceleration and gyro high-frequency power shares, RMS angular acceleration (deg/s^2)."""
    acc, gyr = np.asarray(acc, dtype = float), np.asarray(gyr, dtype = float)
    jerk = get_windows(np.r_[np.nan, np.linalg.norm(np.diff(acc, axis = 0), axis = 1) * rate], ends, length)
    alpha = get_windows(np.r_[np.nan, np.linalg.norm(np.diff(gyr, axis = 0), axis = 1) * rate], ends, length)
    return {'jerk_rms': np.sqrt(np.mean(jerk ** 2, axis = 1)), 'jerk_peak': np.max(jerk, axis = 1),
            'accel_hf_ratio': get_band_power_share(get_windows(np.linalg.norm(acc, axis = 1), ends, length), cutoff_hz, rate),
            'angular_accel_rms': np.sqrt(np.mean(alpha ** 2, axis = 1)),
            'gyro_hf_ratio': get_band_power_share(get_windows(np.linalg.norm(gyr, axis = 1), ends, length), cutoff_hz, rate)}


# ----------------------------------------------------------------------------- 2. polar phase space

def get_polar_features(angle, velocity, ends, length = WINDOW['polar'], rate = RATE_HZ):
    """Phase portrait of (angle, velocity), each centred on its midrange and scaled by its half-range over the window.

    Midrange scaling makes a sinusoid an exact unit circle whenever the window holds a full cycle (2 s covers
    any gait above 0.5 Hz); mean / standard-deviation scaling does not over a non-integer number of cycles.
    phi = atan2(-velocity~, angle~) advances by 2 pi per cycle whatever the angle's sign convention, because
    velocity is the angle's derivative. Returns the phase at the window end as sin and cos, the phase rate in
    cycles/s (cadence), the share of samples where the phase advances (1 for a clean cycle, about 0.5 for
    noise), and the coefficient of variation of the radius (0 for a perfect circle).
    """
    th, om = get_windows(angle, ends, length), get_windows(velocity, ends, length)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        th = (th - (th.max(axis = 1, keepdims = True) + th.min(axis = 1, keepdims = True)) / 2) \
            / ((th.max(axis = 1, keepdims = True) - th.min(axis = 1, keepdims = True)) / 2)
        om = (om - (om.max(axis = 1, keepdims = True) + om.min(axis = 1, keepdims = True)) / 2) \
            / ((om.max(axis = 1, keepdims = True) - om.min(axis = 1, keepdims = True)) / 2)
        phi = np.arctan2(-om, th)
        step = (np.diff(phi, axis = 1) + np.pi) % (2 * np.pi) - np.pi
        radius = np.hypot(th, om)
        ring_cv = radius.std(axis = 1) / radius.mean(axis = 1)
    valid = np.isfinite(step).all(axis = 1)
    return {'phase_sin': np.sin(phi[:, -1]), 'phase_cos': np.cos(phi[:, -1]),
            'phase_rate_hz': np.where(valid, step.sum(axis = 1) / (2 * np.pi * (length - 1) / rate), np.nan),
            'phase_monotonicity': np.where(valid, np.mean(step > 0, axis = 1), np.nan),
            'ring_cv': np.where(valid, ring_cv, np.nan)}


# ----------------------------------------------------------------------------- 3. spectral

def get_spectral_features(angle, ends, length = WINDOW['spectral'], rate = RATE_HZ):
    """Dominant frequency (Hz) in the stride band, normalized spectral entropy, stride regularity and its lag.

    Stride regularity is the largest autocorrelation of the hip angle over lags of 0.5-1.6 s (after Moe-Nilssen
    and Helbostad, on the hip angle instead of trunk acceleration, overlap-normalized so it stays in [-1, 1]);
    its lag is the stride period, returned in samples for the step-regularity search. The dominant frequency's
    peak is located between FFT bins by a parabolic fit.
    """
    rows = get_windows(angle, ends, length)
    valid = np.isfinite(rows).all(axis = 1)
    rows = np.where(valid[:, None], rows, 0.0)
    tapered = (rows - rows.mean(axis = 1, keepdims = True)) * np.hanning(length)
    freq = np.fft.rfftfreq(N_FFT, 1 / rate)
    band = np.flatnonzero((freq >= STRIDE_BAND_HZ[0]) & (freq <= STRIDE_BAND_HZ[1]))
    magnitude = np.abs(np.fft.rfft(tapered, N_FFT, axis = 1))
    k = band[np.argmax(magnitude[:, band], axis = 1)]
    rows_idx = np.arange(len(k))
    left, mid, right = magnitude[rows_idx, k - 1], magnitude[rows_idx, k], magnitude[rows_idx, k + 1]
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        shift = np.clip(np.nan_to_num(0.5 * (left - right) / (left - 2 * mid + right)), -0.5, 0.5)   # parabolic peak
    dominant = (k + shift) * rate / N_FFT
    power = np.abs(np.fft.rfft(tapered, axis = 1)) ** 2
    f0 = np.fft.rfftfreq(length, 1 / rate)
    eb = (f0 >= ENTROPY_BAND_HZ[0]) & (f0 <= ENTROPY_BAND_HZ[1])
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        p = power[:, eb] / power[:, eb].sum(axis = 1, keepdims = True)
        entropy = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis = 1) / np.log(eb.sum())
    ac = get_overlap_autocorrelation(rows)
    lags = np.arange(int(STRIDE_LAG_S[0] * rate), int(STRIDE_LAG_S[1] * rate) + 1)
    best = np.argmax(np.nan_to_num(ac[:, lags], nan = -np.inf), axis = 1)
    regularity = ac[np.arange(len(ac)), lags[best]]
    usable = valid & np.isfinite(regularity) & np.isfinite(entropy)
    return {'dominant_hz': np.where(usable, dominant, np.nan), 'spectral_entropy': np.where(usable, entropy, np.nan),
            'stride_regularity': np.where(usable, regularity, np.nan), 'stride_lag': np.where(usable, lags[best], np.nan)}


def get_step_regularity(long_acc_left, long_acc_right, stride_lag, ends, length = WINDOW['spectral']):
    """Autocorrelation of the two thighs' summed long-axis acceleration at half a stride (+-10%).

    The sum stands in for whole-body vertical acceleration, which repeats every step; high values mean the
    left and right steps look alike.
    """
    signal = np.asarray(long_acc_left, dtype = float) + np.asarray(long_acc_right, dtype = float)
    rows = get_windows(signal, ends, length)
    valid = np.isfinite(rows).all(axis = 1) & np.isfinite(stride_lag)
    ac = get_overlap_autocorrelation(np.where(valid[:, None], rows, 0.0))
    half = np.where(valid, stride_lag, 2 * STRIDE_LAG_S[0] * RATE_HZ) / 2
    lo, hi = np.floor(0.9 * half).astype(int), np.ceil(1.1 * half).astype(int)
    idx = lo[:, None] + np.arange(int((hi - lo).max()) + 1 if len(lo) else 1)[None, :]
    vals = np.take_along_axis(ac, np.clip(idx, 0, length - 1), axis = 1)
    vals = np.where(idx <= hi[:, None], np.nan_to_num(vals, nan = -np.inf), -np.inf)
    return np.where(valid, vals.max(axis = 1), np.nan)


# ----------------------------------------------------------------------------- 4. orientation

def get_thigh_pitch(acc_up, acc_anterior, ends, gyro_ml = None, rate = RATE_HZ, cutoff_hz = PITCH_CUTOFF_HZ, tau_s = PITCH_TAU_S):
    """Thigh pitch (deg) at each window end: the thigh's angle from vertical in the sagittal plane, flexion positive.

    At rest an accelerometer reads the upward reaction to gravity; a thigh flexed by psi sees it as
    (up: g cos psi, anterior: g sin psi), so psi_acc = atan2(anterior, up): 0 standing, about 90 seated. During
    gait the limb's own acceleration corrupts that, so with ``gyro_ml`` (deg/s about the mediolateral axis,
    flexion positive) a complementary filter carries the angle through movement and lets gravity correct its drift
    on a ``tau_s`` time scale: psi_k = a (psi_(k-1) + w_k dt) + (1 - a) psi_acc,k, a = tau / (tau + dt).
    Without a gyro it returns the gravity angle low-passed at ``cutoff_hz``.
    """
    up, anterior = fill_gaps(acc_up), fill_gaps(acc_anterior)
    if gyro_ml is None:
        b, a = butter(2, cutoff_hz, fs = rate)
        pitch = np.degrees(np.arctan2(lfilter(b, a, anterior), lfilter(b, a, up)))
    else:
        gravity = np.degrees(np.arctan2(anterior, up))
        alpha = tau_s / (tau_s + 1 / rate)
        drive = alpha * fill_gaps(gyro_ml) / rate + (1 - alpha) * gravity
        pitch = lfilter([1.0], [1.0, -alpha], drive, zi = [alpha * gravity[0]])[0]
    missing = ~(np.isfinite(acc_up) & np.isfinite(acc_anterior))
    if gyro_ml is not None:
        missing |= ~np.isfinite(gyro_ml)
    pitch[missing] = np.nan
    return pitch[np.asarray(ends, dtype = int)]


# ----------------------------------------------------------------------------- all of them, one leg or both

FEATURE_NAMES = ['jerk_rms', 'jerk_peak', 'accel_hf_ratio', 'angular_accel_rms', 'gyro_hf_ratio',
                 'phase_sin', 'phase_cos', 'phase_rate_hz', 'phase_monotonicity', 'ring_cv',
                 'dominant_hz', 'spectral_entropy', 'stride_regularity', 'thigh_pitch']


def get_leg_features(angle, acc, gyr, ends, rate = RATE_HZ):
    """Every per-leg feature for one leg; ``acc`` and ``gyr`` are (n, 3) in (up, mediolateral, anterior) order."""
    acc, gyr = np.asarray(acc, dtype = float), np.asarray(gyr, dtype = float)
    out = get_jerk_features(acc, gyr, ends, rate = rate)
    out.update(get_polar_features(angle, get_causal_velocity(angle, rate), ends, rate = rate))
    out.update(get_spectral_features(angle, ends, rate = rate))
    out['thigh_pitch'] = get_thigh_pitch(acc[:, 0], acc[:, 2], ends, gyro_ml = gyr[:, 1], rate = rate)
    return out


def get_window_features(left, right, ends, rate = RATE_HZ):
    """Both legs' features plus the bilateral step regularity, one row per window end.

    ``left`` and ``right`` are dicts with ``angle`` (n,), ``acc`` and ``gyr`` (n, 3, anatomical order).
    """
    legs = {side: get_leg_features(leg['angle'], leg['acc'], leg['gyr'], ends, rate) for side, leg in (('l', left), ('r', right))}
    table = pd.DataFrame({f'{name}_{side}': legs[side][name] for side in ('l', 'r') for name in FEATURE_NAMES})
    table['step_regularity'] = get_step_regularity(np.asarray(left['acc'])[:, 0], np.asarray(right['acc'])[:, 0],
                                                   legs['l']['stride_lag'], ends)
    return table.astype(np.float32)
