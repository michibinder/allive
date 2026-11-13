import numpy as np
from scipy import signal

VERT_CUTOFF_KM = 20.0
BW_ORDER = 5
MIN_FILTER_POINTS = 10


def vertical_primes(data, dz_m):
    """T' of a (time, z) field: vertical Butterworth high-pass, cutoff VERT_CUTOFF_KM, rounded to the
    precision of data; profiles with fewer than MIN_FILTER_POINTS values are blanked."""
    fs = 1.0 / (dz_m / 1000.0)
    b, a = signal.butter(BW_ORDER, (1.0 / VERT_CUTOFF_KM) / (0.5 * fs), btype="high")
    primes = np.full(data.shape, np.nan)
    for row, profile in zip(primes, data):
        valid = ~np.isnan(profile)
        if valid.sum() >= MIN_FILTER_POINTS:
            mirrored = np.append(np.flip(profile[valid]), profile[valid])
            row[valid] = signal.filtfilt(b, a, mirrored)[valid.sum():].astype(data.dtype)
    return primes
