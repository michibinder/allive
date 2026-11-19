import datetime as dt
import http.client
import urllib.request
from pathlib import Path

import numpy as np

from allive import paths, track
from allive.utc import log_utc, time_clusters, utc_datetime

SOURCE_URL = "https://wavecloud.pa.dlr.de/lidar/"
FETCH_TIMEOUT_S = 20.0
HTML_CONTENT_TYPE = "text/html"
N_META = 5
DZ_M = 100.0
MAX_STRIDE = 100
UNIX_MS_THRESHOLD = 1.0e11
UNIX_S_RANGE = (1.0e9, 4.0e9)
FLIGHT_WINDOW_PAD_S = 10800.0
FLIGHT_GAP_S = 7200.0
MIN_LEVEL_FRACTION = 0.02
VALID_RANGE = {"T.txt": (100.0, 400.0), "w.txt": (-100.0, 100.0), "u.txt": (-300.0, 300.0),
               "v.txt": (-300.0, 300.0)}


def is_url(source):
    """True when the source is an http(s) URL rather than a local directory."""
    return source.startswith(("http://", "https://"))


def fetch(source, name):
    """Bytes of one file from a base URL or a local directory (b'' when a local file is missing)."""
    if is_url(source):
        with urllib.request.urlopen(source.rstrip("/") + "/" + name, timeout=FETCH_TIMEOUT_S) as r:
            if r.headers.get_content_type() == HTML_CONTENT_TYPE:
                raise OSError("HTML page instead of the data file")
            return r.read()
    path = Path(source) / name
    return path.read_bytes() if path.exists() else b""


def is_restart(old, new):
    """True when fetched bytes neither extend nor are a prefix of the previous bytes."""
    return bool(new) and not old.startswith(new) and not new.startswith(old)


def fetch_all(source, previous):
    """Bytes of every file: after a restart exactly what was fetched, else failed or shrunken
    reads keep the previous bytes."""
    old = {name: (previous or {}).get(name, b"") for name in VALID_RANGE}
    fetched = {}
    for name in VALID_RANGE:
        try:
            fetched[name] = fetch(source, name)
        except (OSError, http.client.HTTPException) as err:
            log_utc(f"{name}: fetch failed ({err})")
            fetched[name] = None
    if any(is_restart(old[name], new) for name, new in fetched.items()):
        log_utc("files restarted: previous data dropped")
        return {name: new or b"" for name, new in fetched.items()}
    return {name: old[name] if new is None or old[name].startswith(new) else new
            for name, new in fetched.items()}


def split_blocks(numbers, stride):
    """(heights in m, values) of a line holding a km height axis, values and uncertainties in three
    equal blocks; (None, numbers) for a line on the implicit grid."""
    m, rest = divmod(numbers.size, 3)
    if m > 1 and rest == 0 and np.allclose(np.diff(numbers[:m]), stride * DZ_M / 1000.0):
        return numbers[:m] * 1000.0, numbers[m:2 * m]
    return None, numbers


def parse_line(line):
    """(metadata, explicit heights in m or None, values) of one profile line of bytes, or None
    when the line is unusable."""
    tokens = line.split()
    if len(tokens) < N_META + 1:
        return None
    try:
        numbers = np.array([float(t) for t in tokens])
    except ValueError:
        return None
    meta = numbers[:N_META]
    if meta[0] > UNIX_MS_THRESHOLD:
        meta[0] /= 1000.0
    when, lat, lon, _, stride = meta
    plausible = (UNIX_S_RANGE[0] <= when <= UNIX_S_RANGE[1] and abs(lat) <= 90.0
                 and abs(lon) <= 360.0 and stride == np.rint(stride) and 1 <= stride <= MAX_STRIDE)
    if not (np.all(np.isfinite(meta)) and plausible):
        return None
    return (meta, *split_blocks(numbers[N_META:], stride))


def parse_profiles(payload):
    """Profiles of one file sorted by time, the last written one per timestamp, or None; a last
    line without its newline is still being written and left out."""
    lines = payload.splitlines()
    if payload and not payload.endswith(b"\n"):
        lines.pop()
    lines = list(dict.fromkeys(reversed(lines)))[::-1]
    parsed = [p for p in map(parse_line, lines) if p is not None]
    if not parsed:
        return None
    meta = np.array([p[0] for p in parsed])
    order = np.argsort(meta[:, 0], kind="stable")
    order = order[np.append(np.diff(meta[order, 0]) != 0.0, True)]
    return {"time": meta[order, 0], "lat": meta[order, 1], "lon": meta[order, 2],
            "alt": meta[order, 3], "stride": np.rint(meta[order, 4]).astype(int),
            "heights": [parsed[i][1] for i in order], "values": [parsed[i][2] for i in order]}


def located(raw):
    """Mask of the profiles of a file that carry a position (latitude and longitude not both 0)."""
    return (raw["lat"] != 0.0) | (raw["lon"] != 0.0)


def regrid(z_src, values, z_dst):
    """NaN-aware linear interpolation that never bridges gaps and never extrapolates."""
    good = np.isfinite(values)
    if not good.any():
        return np.full(z_dst.shape, np.nan)
    out = np.interp(z_dst, z_src[good], values[good], left=np.nan, right=np.nan)
    gap = np.interp(z_dst, z_src, (~good).astype(float), left=1.0, right=1.0)
    return np.where(gap > 0, np.nan, out)


def build_curtain(raw, valid, above_aircraft):
    """The (time, z) field of one file on a common uniform height grid."""
    if raw is None:
        return None
    dz = DZ_M * raw["stride"].min()
    heights = []
    for explicit, values, stride, alt in zip(raw["heights"], raw["values"], raw["stride"],
                                             raw["alt"]):
        base = alt if above_aircraft else 0.0
        heights.append(base + np.arange(len(values)) * stride * DZ_M if explicit is None else explicit)
    z = np.arange(0.0, max(h[-1] for h in heights) + 0.5 * dz, dz)
    data = np.full((len(heights), z.size), np.nan)
    for i, (values, z_src) in enumerate(zip(raw["values"], heights)):
        bad = ~np.isfinite(values) | (values == 0.0) | (values < valid[0]) | (values > valid[1])
        data[i] = regrid(z_src, np.where(bad, np.nan, values), z)
    return {"time": raw["time"], "z": z, "dz": dz, "data": data}


def height_range_km(curtain):
    """(lowest, highest) level in km valid in at least MIN_LEVEL_FRACTION of the profiles."""
    counts = np.isfinite(curtain["data"]).sum(axis=0)
    levels = counts >= max(1.0, MIN_LEVEL_FRACTION * curtain["time"].size)
    if not levels.any():
        return None
    z = curtain["z"][levels] / 1000.0
    return z.min(), z.max()


def in_window(raw, start_s, end_s):
    """The profiles of a file between two times, or None when there are none."""
    if raw is None:
        return None
    keep = np.flatnonzero((raw["time"] >= start_s) & (raw["time"] <= end_s))
    if keep.size == 0:
        return None
    return {key: [value[i] for i in keep] if isinstance(value, list) else value[keep]
            for key, value in raw.items()}


def profile_count(parsed):
    """Number of profiles over all files."""
    return sum(raw["time"].size for raw in parsed.values() if raw is not None)


def window_end(parsed, start_s, end_s):
    """end_s moved on to the newest profile of every run (no gap over FLIGHT_GAP_S) overlapping the window."""
    times = np.unique(np.concatenate([raw["time"] for raw in parsed.values() if raw is not None] or [[]]))
    for run in time_clusters(times, FLIGHT_GAP_S):
        if times[run[0]] <= end_s and times[run[-1]] >= start_s:
            end_s = max(end_s, times[run[-1]])
    return end_s


def flight_profiles(parsed, planned):
    """The profiles within FLIGHT_WINDOW_PAD_S of the planned flight, the window end moved on by
    window_end."""
    start_s = planned["time"][0] - FLIGHT_WINDOW_PAD_S
    end_s = window_end(parsed, start_s, planned["time"][-1] + FLIGHT_WINDOW_PAD_S)
    kept = {name: in_window(raw, start_s, end_s) for name, raw in parsed.items()}
    dropped = profile_count(parsed) - profile_count(kept)
    if dropped:
        log_utc(f"{dropped} profiles outside the flight window ignored")
    return kept


def measured_fixes(parsed):
    """Time-sorted positions of the located profiles pooled over the files, one per timestamp,
    or None without any."""
    profiles = [p for p in parsed.values() if p is not None]
    if not profiles:
        return None
    masks = [located(p) for p in profiles]
    pooled = {key: np.concatenate([p[key][mask] for p, mask in zip(profiles, masks)])
              for key in track.TRACK_KEYS}
    if pooled["time"].size == 0:
        return None
    _, index = np.unique(pooled["time"], return_index=True)
    return {key: values[index] for key, values in pooled.items()}


def newest_profile(parsed):
    """Time of the newest profile over all files, or None without profiles."""
    times = [p["time"][-1] for p in parsed.values() if p is not None]
    return max(times) if times else None


def flight_stamp(parsed):
    """UTC stamp of the earliest profile over all files, or None."""
    starts = [p["time"][0] for p in parsed.values() if p is not None]
    return f"{utc_datetime(min(starts)):%Y%m%d-%H%M}" if starts else None


def archive(payloads, stamp):
    """Mirror the non-empty raw files byte-exact to archive_dir()/<stamp>; an archived file the
    new bytes do not extend is kept under a time-stamped name."""
    target = paths.archive_dir() / stamp
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        if not payload:
            continue
        path = target / name
        if path.exists() and not payload.startswith(path.read_bytes()):
            path.rename(path.with_name(f"{name}.{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S%f}"))
        path.write_bytes(payload)


def describe(name, raw, curtain):
    """One-line summary of a file."""
    if raw is None:
        return f"{name}: no data"
    strides = sorted(set(raw["stride"].tolist()))
    span = height_range_km(curtain)
    heights = f"valid {span[0]:.1f}-{span[1]:.1f} km" if span else "no valid level"
    alt = raw["alt"][located(raw)]
    aircraft = f", aircraft {alt[-1] / 1000.0:.1f} km" if alt.size else ""
    return (f"{name}: {raw['time'].size} profiles, {utc_datetime(raw['time'][0]):%H:%M}-"
            f"{utc_datetime(raw['time'][-1]):%H:%M} UTC, stride {'/'.join(map(str, strides))}, "
            f"{heights}{aircraft}")
