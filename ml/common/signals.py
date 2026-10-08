"""Dataset-agnostic signal helpers: bouts of a flag, lagged correlation, stride period, IMU anatomical frame.

Plain numpy on arrays at any sample rate (``RATE_HZ`` is only the default). Used by the Nature notebooks and
meant for any dataset's pipeline.
"""
import numpy as np

RATE_HZ = 200


def get_bouts(flag, min_len = 1):
    """[start, end) sample ranges of every run of true values at least ``min_len`` long, shape (k, 2)."""
    f = np.r_[False, np.asarray(flag, dtype = bool), False]
    edges = np.flatnonzero(np.diff(f.astype(np.int8)))
    runs = edges.reshape(-1, 2)
    return runs[(runs[:, 1] - runs[:, 0]) >= min_len]


def get_corr(a, b):
    """Pearson r over the samples where both are finite; NaN when too few or constant."""
    a, b = np.asarray(a, dtype = float), np.asarray(b, dtype = float)
    ok = np.isfinite(a) & np.isfinite(b)
    r = np.nan
    if ok.sum() > 100 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return r


def get_lag_corr(a, b, max_lag = 20):
    """(lag, r) maximising |corr(a[t], b[t + lag])| over |lag| <= max_lag samples; positive lag: b trails a."""
    a, b = np.asarray(a, dtype = float), np.asarray(b, dtype = float)
    best = (0, np.nan)
    for lag in range(-max_lag, max_lag + 1):
        r = get_corr(a[max(0, -lag): len(a) - max(0, lag)], b[max(0, lag): len(b) - max(0, -lag)])
        if np.isfinite(r) and (not np.isfinite(best[1]) or abs(r) > abs(best[1])):
            best = (lag, r)
    return best


def get_stride_period(x, lo = 0.6, hi = 2.2, rate = RATE_HZ):
    """Dominant period (s) of a gait signal from its autocorrelation peak in [lo, hi]; NaN if weak or too short."""
    x = np.asarray(x, dtype = float)
    x = x[np.isfinite(x)]
    result = np.nan
    if len(x) > int(2.2 * hi * rate):
        x = x - x.mean()
        spec = np.fft.rfft(x, 2 * len(x))
        ac = np.fft.irfft(spec * np.conj(spec))[:len(x)]
        ac = ac / ac[0]
        k = int(lo * rate) + int(np.argmax(ac[int(lo * rate): int(hi * rate)]))
        result = k / rate if ac[k] > 0.3 else np.nan
    return result


def solve_imu_frame(acc, gyr, hip_velocity, quiet, walking):
    """Anatomical axes in IMU coordinates, rows (up, mediolateral flexion-positive, anterior).

    up: the mean accelerometer reading while quiet (gravity reaction). Mediolateral: the dominant gyro
    axis while walking, signed so that rotation about it tracks hip flexion (the encoders are extension
    positive), made orthogonal to up. Anterior = up x mediolateral: a positive (flexion) rotation about the
    mediolateral axis tips the thigh's long axis backward, so up x ML points forward. Checked on the Nature
    data: seated (thigh horizontal, knee forward) the accelerometer reads +g on this axis.
    """
    up = acc[quiet].mean(0)
    up = up / np.linalg.norm(up)
    g = gyr[walking]
    w = np.linalg.svd(g - g.mean(0), full_matrices = False)[2][0]
    w = -w if get_corr(g @ w, -hip_velocity[walking]) < 0 else w
    ml = w - (w @ up) * up
    ml = ml / np.linalg.norm(ml)
    return np.vstack([up, ml, np.cross(up, ml)])
