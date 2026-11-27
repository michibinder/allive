import datetime as dt
import importlib
import os
import signal
import socket
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import xarray as xr

from allive.utc import log_utc, utc_now

LEVELS_ML = "1/to/137"
OPTIONAL_PACKAGES = ("eccodes", "cfgrib", "ecmwfapi")
FATAL_MARS_CODES = ("MARS_EXPECTED_FIELDS", "MARS_RESTRICTED_ACCESS_TO_VALID_DATA")
COUNTED_KEYS = ("date", "time", "step", "levelist", "param")
EMBARGO_HOURS = None
RUN_LAG_HOURS = 7
BASE_CYCLE_HOURS = 12
STEP_SUMMARY_MAX_CHARS = 30
MARS_REQUESTS_PATH = "services/mars/requests"
DOWNLOAD_SOCKET_TIMEOUT_S = 300.0


def floor_cycle(t, hours):
    """t rounded down to the last cycle of a grid spaced by hours from 00 UTC."""
    return t.replace(hour=t.hour // hours * hours, minute=0, second=0, microsecond=0)


def base_time(cfg):
    """Configured forecast base time as a datetime."""
    return dt.datetime.fromisoformat(cfg["base"])


def file_stem(cfg):
    """Path prefix of every file of this download."""
    base = base_time(cfg)
    return f"{cfg['outdir']}/{cfg['name']}_{base:%y%m%d%H}"


def netcdf_files(cfg):
    """Paths of the converted model-level and level-1 netcdf files of a download."""
    stem = file_stem(cfg)
    return Path(f"{stem}-ml.nc"), Path(f"{stem}-ml-p.nc")


def step_list(steps):
    """Explicit slash-joined MARS step list."""
    return "/".join(str(s) for s in steps)


def build_requests(cfg):
    """One forecast base time: z from the analysis, lnsp, model levels per step chunk."""
    stem = file_stem(cfg)
    base = base_time(cfg)
    common = {
        "class": "od",
        "stream": "oper",
        "expver": "1",
        "date": f"{base:%Y-%m-%d}",
        "time": f"{base:%H}",
        "levtype": "ml",
        "grid": cfg["grid"],
        "area": cfg["area"],
    }
    steps = cfg["steps"]
    chunks = [[s for s in steps if a <= s <= b] for a, b in cfg["step_chunks"]]
    if not all(chunks) or sorted(s for chunk in chunks for s in chunk) != sorted(steps):
        raise ValueError("step_chunks must be non-empty and cover every configured step exactly once")
    fc = {**common, "type": "fc"}
    requests = [
        ({**common, "type": "an", "levelist": "1", "param": "129"}, f"{stem}-an-p.grb"),
        ({**fc, "levelist": "1", "param": "152", "step": step_list(steps)}, f"{stem}-fc-p.grb"),
    ]
    for (a, b), chunk in zip(cfg["step_chunks"], chunks):
        requests.append((
            {**fc, "levelist": LEVELS_ML, "param": cfg["params_ml"], "step": step_list(chunk)},
            f"{stem}-fc-s{a:03d}-{b:03d}.grb",
        ))
    return requests


def newest_base(last_step_h, now):
    """Newest 00/12 UTC base that is in MARS at now and whose last step is released: it waits
    RUN_LAG_HOURS without an embargo, else the longer of that and the last step plus the embargo."""
    wait = RUN_LAG_HOURS if EMBARGO_HOURS is None else max(RUN_LAG_HOURS, last_step_h + EMBARGO_HOURS)
    return floor_cycle(now - dt.timedelta(hours=wait), BASE_CYCLE_HOURS)


def allowed_base(cfg):
    """Newest base time that is available now for the configured steps."""
    return newest_base(max(cfg["steps"]), utc_now())


def check_base(cfg):
    """Exit if a forecast base time is not available yet (not in MARS, or inside the embargo)."""
    allowed = allowed_base(cfg)
    if base_time(cfg) > allowed:
        reason = "not in MARS yet" if EMBARGO_HOURS is None else f"inside the {EMBARGO_HOURS} h embargo"
        raise SystemExit(f"base {cfg['base']} is not available: {reason}; "
                         f"newest allowed base is {allowed:%Y-%m-%dT%H:%M}")


def count_values(value):
    """Number of values in a MARS list such as a/b/c or a/to/b[/by/c]."""
    parts = str(value).split("/")
    if "to" not in parts:
        return len(parts)
    i = parts.index("to")
    has_by = len(parts) > i + 3 and parts[i + 2] == "by"
    by = int(parts[i + 3]) if has_by else 1
    span = (int(parts[i + 1]) - int(parts[i - 1])) // by + 1
    return len(parts) - (4 if has_by else 2) - 1 + span


def expected_fields(req):
    """Field count a request returns: product of its list lengths."""
    total = 1
    for key in COUNTED_KEYS:
        if key in req:
            total *= count_values(req[key])
    return total


def step_summary(req):
    """Short description of the request's step list."""
    if "step" not in req:
        return "-"
    steps = req["step"]
    if len(steps) <= STEP_SUMMARY_MAX_CHARS:
        return steps
    return f"{steps.split('/')[0]}../{steps.split('/')[-1]} ({count_values(steps)} steps)"


def plan(cfg):
    """Print every request with its stream, target and expected field count; return the total."""
    requests = build_requests(cfg)
    total = 0
    for req, target in requests:
        fields = expected_fields(req)
        total += fields
        print(f"[plan] {os.path.basename(target)}  stream={req['stream']} type={req['type']}"
              f" date={req['date']} time={req['time']} step={step_summary(req)}"
              f" lev={req['levelist']} param={req['param']} fields={fields}")
    print(f"[plan] total {len(requests)} requests, {total} fields into {cfg['outdir']}")
    allowed = allowed_base(cfg)
    print(f"[plan] configured base {cfg['base']}, newest allowed base {allowed:%Y-%m-%dT%H:%M}")
    return total


def warn_pending_requests():
    """Warn about pending MARS requests, which hold the single execution slot."""
    try:
        from ecmwfapi import api

        key, url, email = api.get_apikey_values()
        connection = api.Connection(url, email, key, quiet=True)
        pending = connection.call(f"{url}/{MARS_REQUESTS_PATH}", method="GET")["mars"]
        listing = ", ".join(f"{p.get('name')}:{p.get('status')}" for p in pending)
    except Exception as exc:
        print(f"[warn] could not list pending MARS requests: {str(exc)[:200]}", flush=True)
        return
    if pending:
        print(f"[warn] {len(pending)} pending MARS request(s) may block the queue: {listing}",
              flush=True)


def count_messages(path):
    """Number of GRIB messages in a file."""
    import eccodes

    with open(path, "rb") as handle:
        return eccodes.codes_count_in_file(handle)


def retrieve(server, req, target, out=None, tries=3):
    """Run one MARS request into target via .tmp, check the field count, skip if present."""
    if os.path.exists(target):
        print(f"[skip] {target}", file=out, flush=True)
        return
    expected = expected_fields(req)
    print(f"[mars] {target}  ({expected} fields, {req['type']} {req['date']} t={req['time']}"
          f" s={step_summary(req)} lev={req['levelist']})", file=out, flush=True)
    tmp = target + ".tmp"
    for attempt in range(1, tries + 1):
        try:
            server.execute(req, tmp)
        except Exception as exc:
            print(f"[warn] attempt {attempt}/{tries} failed: {str(exc)[:300]}", file=out, flush=True)
            if any(code in str(exc) for code in FATAL_MARS_CODES) or attempt == tries:
                raise
            continue
        found = count_messages(tmp)
        if found != expected:
            raise RuntimeError(f"{tmp}: expected {expected} fields, found {found}")
        os.replace(tmp, target)
        return


def normalize_longitude(ds):
    """ds with longitudes -180..180, ascending."""
    return ds.assign_coords(longitude=(ds["longitude"] + 180) % 360 - 180).sortby("longitude")


def hourly_records(path, window):
    """Open one GRIB and split it into single-valid-time datasets within a (start, end)
    valid-time window, before any data is read."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    for name in ("time", "step", "hybrid"):
        if name in ds.coords and name not in ds.dims:
            ds = ds.expand_dims(name)
    if "step" not in ds.dims:
        ds = ds.expand_dims(step=[np.timedelta64(0, "h")])
    records = []
    for i in range(ds.sizes["time"]):
        for j in range(ds.sizes["step"]):
            part = ds.isel(time=i, step=j)
            vt = (part["valid_time"].values if "valid_time" in part.coords
                  else part["time"].values + part["step"].values)
            if not np.datetime64(window[0]) <= vt <= np.datetime64(window[1]):
                continue
            drop = [c for c in ("time", "step", "valid_time", "number") if c in part.coords]
            records.append(part.drop_vars(drop).expand_dims(time=[vt]))
    return records


def valid_time_records(gribs, window):
    """Records of all GRIB parts merged per valid time, in time order (window as in hourly_records)."""
    by_time = {}
    for path in gribs:
        for record in hourly_records(path, window):
            by_time.setdefault(record["time"].item(), []).append(record)
    return [xr.merge(by_time[t], combine_attrs="override") for t in sorted(by_time)]


def build_netcdf(gribs, target, start, end):
    """Concatenate the GRIB parts by valid time (start..end) into one ERA5-layout netcdf."""
    if os.path.exists(target):
        print(f"[skip] {target}")
        return
    records = valid_time_records(gribs, (start, end))
    if not records:
        raise ValueError(f"no valid times in {start}..{end} in {gribs}")
    ds = xr.concat(records, dim="time", combine_attrs="override").rename({"hybrid": "level"})
    ds = normalize_longitude(ds)
    if "z" in ds and ds["z"].isnull().any():
        ds["z"] = ds["z"].fillna(ds["z"].mean("time"))
    ds = ds.transpose("time", "level", "latitude", "longitude")
    tmp = f"{target}.tmp"
    ds.to_netcdf(tmp)
    os.replace(tmp, target)
    print(f"[out ] {target}  ({ds.sizes['time']} times)")


def grib_groups(requests):
    """GRIB targets split into model-level and level-1 groups."""
    gribs_ml = [t for r, t in requests if r["levelist"] != "1"]
    gribs_p = [t for r, t in requests if r["levelist"] == "1"]
    return gribs_ml, gribs_p


def conversion_window(cfg):
    """Valid-time window (start, end) kept in the converted netcdfs: the whole download window."""
    base = base_time(cfg)
    return cfg["base"], (base + dt.timedelta(hours=max(cfg["steps"]))).isoformat()


def convert(cfg, gribs_ml, gribs_p):
    """Offline GRIB -> netcdf conversion over the conversion window."""
    missing = [g for g in gribs_ml + gribs_p if not os.path.exists(g)]
    if missing:
        raise SystemExit(f"[fail] missing GRIB files: {missing}")
    start, end = conversion_window(cfg)
    print(f"[conv] {start}..{end}, area {cfg['area']}", flush=True)
    file_ml, file_ml_p = netcdf_files(cfg)
    build_netcdf(gribs_ml, file_ml, start, end)
    build_netcdf(gribs_p, file_ml_p, start, end)


def fetch(client, req, target):
    """Run one request with its own client, logging to target's .log (with the traceback of a
    failure); True on success."""
    log_utc(f"[run ] {target}")
    try:
        with open(os.path.splitext(target)[0] + ".log", "a", buffering=1) as out:
            try:
                retrieve(client("mars", log=partial(log_utc, out=out)), req, target, out)
            except Exception:
                traceback.print_exc(file=out)
                raise
    except Exception as exc:
        error = str(exc).partition("\n")[0]
        log_utc(f"[fail] {target}: {type(exc).__name__}: {error}")
        return False
    log_utc(f"[ok  ] {target}")
    return True


def download(cfg):
    """Run the MARS requests of the configured download, then convert the GRIB files."""
    socket.setdefaulttimeout(DOWNLOAD_SOCKET_TIMEOUT_S)
    try:
        for name in OPTIONAL_PACKAGES:
            importlib.import_module(name)
    except ImportError as error:
        raise SystemExit(f"{error}; the IFS download needs: pip install 'allive[download]'")
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(cfg["outdir"], exist_ok=True)
    requests = build_requests(cfg)
    gribs_ml, gribs_p = grib_groups(requests)
    check_base(cfg)
    warn_pending_requests()
    from ecmwfapi import ECMWFService

    with ThreadPoolExecutor(max_workers=cfg["parallel"]) as pool:
        try:
            succeeded = list(pool.map(lambda job: fetch(ECMWFService, *job), requests))
        except KeyboardInterrupt:
            log_utc("[intr] interrupted; submitted requests keep running at ECMWF,"
                    " their request ids are in the per-request .log files")
            os._exit(128 + signal.SIGINT)
    failed = [target for (_, target), ok in zip(requests, succeeded) if not ok]
    if failed:
        print(f"[fail] {len(failed)} of {len(requests)} requests failed, conversion skipped:")
        for target in failed:
            print(f"[fail] {target}")
        raise SystemExit(1)
    convert(cfg, gribs_ml, gribs_p)
    print(f"[done] download stage complete ({len(requests)} requests)")
