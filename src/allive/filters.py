import numpy as np
from scipy import signal

VERT_CUTOFF_KM = 15.0
BW_ORDER = 5
MIN_FILTER_POINTS = 10
TP_FILTERS = {
    "butterworth": {"primes": lambda curtain, mark_s: vertical_primes(curtain["data"], curtain["dz"]),
                    "label": rf"$\mathbf{{T'}}$ ($\boldsymbol{{\lambda_z}} <$ {VERT_CUTOFF_KM:.0f} km)"},
    "leg-mean": {"primes": lambda curtain, mark_s: leg_mean_primes(curtain["data"], curtain["time"], mark_s),
                 "label": r"$\mathbf{T'}$ (leg mean removed)"},
}
DEFAULT_TP_FILTER = next(iter(TP_FILTERS))


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


def leg_mean_primes(data, time_s, mark_s):
    """T' of a (time, z) field: each value minus the mean profile of its leg, the curtain split at the
    marks (a profile at a mark starts the next leg); heights without values stay NaN."""
    leg = np.searchsorted(mark_s, time_s, side="right")
    primes = np.empty_like(data)
    for number in np.unique(leg):
        rows = leg == number
        primes[rows] = data[rows] - np.ma.masked_invalid(data[rows]).mean(axis=0).filled(np.nan)
    return primes
