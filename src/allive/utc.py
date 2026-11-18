import datetime as dt

import numpy as np


def utc_now():
    """Current UTC time as a naive datetime."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def log_utc(message, out=None):
    """UTC-stamped line to out (stdout if None) as one flushed write, so threads never splice
    lines; also the ecmwfapi client logger."""
    print(f"{utc_now():%Y-%m-%d %H:%M:%S}Z {message}\n", end="", file=out, flush=True)


def utc_datetime(seconds):
    """Naive UTC datetime of a unix time."""
    return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).replace(tzinfo=None)


def utc_seconds(when):
    """Unix seconds of a datetime; a naive one is read as UTC."""
    return when.replace(tzinfo=when.tzinfo or dt.timezone.utc).timestamp()


def iso_seconds(text):
    """Unix seconds of an ISO time stamp; one without a zone is read as UTC."""
    return utc_seconds(dt.datetime.fromisoformat(text))


def time_clusters(times, gap):
    """Index arrays of the runs of sorted times (or grid rows) whose neighbours lie at most gap apart."""
    parts = np.split(np.arange(times.size), np.flatnonzero(np.diff(times) > gap) + 1)
    return [part for part in parts if part.size]
