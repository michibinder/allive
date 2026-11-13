import datetime as dt
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from allive import levels, mars, paths
from allive.track import TRACK_KEYS
from allive.utc import utc_datetime, utc_now

GRID = "0.1/0.1"
PARAMS_ML = "130/131/132/135"
AREA_MARGIN_DEG = 0.5
AREA_ROUND_DEG = 1.0
PAD_HOURS = 1
CHUNK_STEPS = 24
Z_TOP_M = 80000.0
DZ_M = 200.0
VARIABLES = ("t", "p", "u", "v", "w")
CURTAIN_VARIABLES = ("t", "w", "u", "v")
SECONDS_PER_HOUR = 3600.0
STEP_GRID = ((0, 90, 1), (90, 144, 3), (144, 240, 6))
BASE_STAMP_FORMAT = "%y%m%d%H"


def mars_area(track):
    """N/W/S/E MARS area around the track, widened by AREA_MARGIN_DEG and rounded outward to
    AREA_ROUND_DEG."""
    north = np.ceil((track["lat"].max() + AREA_MARGIN_DEG) / AREA_ROUND_DEG) * AREA_ROUND_DEG
    south = np.floor((track["lat"].min() - AREA_MARGIN_DEG) / AREA_ROUND_DEG) * AREA_ROUND_DEG
    east = np.ceil((track["lon"].max() + AREA_MARGIN_DEG) / AREA_ROUND_DEG) * AREA_ROUND_DEG
    west = np.floor((track["lon"].min() - AREA_MARGIN_DEG) / AREA_ROUND_DEG) * AREA_ROUND_DEG
    return "/".join(f"{v:g}" for v in (north, west, south, east))


def forecast_steps(first, last):
    """Steps of STEP_GRID from the largest one <= first to the smallest one >= last."""
    grid = sorted({step for start, end, spacing in STEP_GRID for step in range(start, end + 1, spacing)})
    if first < grid[0] or last > grid[-1]:
        raise ValueError(f"forecast steps {first}..{last} h are outside the grid {grid[0]}..{grid[-1]} h")
    lower = max(step for step in grid if step <= first)
    upper = min(step for step in grid if step >= last)
    return [step for step in grid if lower <= step <= upper]


def default_base(first_valid):
    """Newest 00/12 UTC run at or before first_valid, and in MARS now if there is no embargo."""
    base = mars.floor_cycle(first_valid, mars.BASE_CYCLE_HOURS)
    if mars.EMBARGO_HOURS is None:
        base = min(base, mars.newest_base(0, utc_now()))
    return base


def flight_config(plan_path, track, base=None):
    """mars forecast config for the box and hours the track needs, from the run base
    (naive UTC datetime; default: default_base)."""
    first_hour = np.floor(track["time"][0] / SECONDS_PER_HOUR) * SECONDS_PER_HOUR
    last_hour = (np.ceil(track["time"][-1] / SECONDS_PER_HOUR) + PAD_HOURS) * SECONDS_PER_HOUR
    first_valid = utc_datetime(first_hour)
    base = default_base(first_valid) if base is None else base
    if base > first_valid:
        raise ValueError(f"base {base:%Y-%m-%dT%H:%M} is later than the first valid hour "
                         f"{first_valid:%Y-%m-%dT%H:%M} of the flight")
    first_step = round((first_valid - base).total_seconds() / SECONDS_PER_HOUR)
    last_step = first_step + round((last_hour - first_hour) / SECONDS_PER_HOUR)
    steps = forecast_steps(first_step, last_step)
    area = mars_area(track)
    chunks = [steps[i:i + CHUNK_STEPS] for i in range(0, len(steps), CHUNK_STEPS)]
    cfg = {
        "name": f"ifs_{Path(plan_path).stem}_{area.replace('/', '_')}_s{steps[0]:03d}-{steps[-1]:03d}",
        "outdir": str(paths.ifs_dir()),
        "base": base.isoformat(timespec="minutes"),
        "steps": steps,
        "step_chunks": [(chunk[0], chunk[-1]) for chunk in chunks],
        "area": area,
        "grid": GRID,
        "params_ml": PARAMS_ML,
    }
    cfg["parallel"] = len(mars.build_requests(cfg))
    return cfg


def downloaded_config(plan_path, track):
    """Config of the newest downloaded run that covers the flight, or None."""
    area = mars_area(track).replace("/", "_")
    pattern = f"ifs_{glob.escape(Path(plan_path).stem)}_{area}_s*_*-ml.nc"
    bases = set()
    for path in paths.ifs_dir().glob(pattern):
        try:
            bases.add(dt.datetime.strptime(path.name[:-len("-ml.nc")].rsplit("_", 1)[1], BASE_STAMP_FORMAT))
        except ValueError:
            continue
    for base in sorted(bases, reverse=True):
        try:
            cfg = flight_config(plan_path, track, base)
        except ValueError:
            continue
        if all(f.exists() for f in mars.netcdf_files(cfg)):
            return cfg
    return None


def open_ifs(file_ml, file_ml_p):
    """Both IFS files merged on the times of the model-level file; latitude ascending,
    longitude -180..180."""
    with xr.open_dataset(file_ml) as ds_ml, xr.open_dataset(file_ml_p) as ds_p:
        merged = ds_ml.merge(ds_p.sel(time=ds_ml["time"]), join="outer")
        ds = levels.normalize(merged).load()
    return mars.normalize_longitude(ds).sortby("latitude")


def track_columns(ds, track):
    """ds interpolated linearly in time, latitude and longitude to the track points."""
    points = {
        "time": xr.DataArray(pd.to_datetime(track["time"], unit="s").values, dims="point"),
        "latitude": xr.DataArray(track["lat"], dims="point"),
        "longitude": xr.DataArray(track["lon"], dims="point"),
    }
    return ds.interp(**points)


def column_dataset(columns, valid):
    """Valid columns as a (time, level, 1, 1) dataset for levels.on_heights."""
    index = np.flatnonzero(valid)
    data = {
        name: (("time", "level", "latitude", "longitude"),
               columns[name].transpose("point", "level").values[index][:, :, None, None])
        for name in columns.data_vars
    }
    coords = {
        "time": columns["time"].values[index],
        "level": columns["level"].values,
        "latitude": [0.0],
        "longitude": [0.0],
    }
    return xr.Dataset(data, coords)


def vertical_velocity(omega, temperature, pressure):
    """Vertical wind in m/s from omega in Pa/s, hydrostatic."""
    return -omega * levels.R_DRY * temperature / (pressure * levels.GRAVITY)


def sample_curtains(ds, track, dz_m=DZ_M):
    """t, w, u, v curtains along the track on a uniform height grid."""
    columns = track_columns(ds, track)
    surface = np.isfinite(columns["lnsp"].isel(level=0).values)
    valid = surface & np.isfinite(columns["t"].values).all(axis=1)
    z = np.arange(0.0, Z_TOP_M + dz_m / 2, dz_m)
    coefficients = levels.coefficients()
    shape = (len(track["time"]), len(z))
    data = {name: np.full(shape, np.nan, dtype=np.float32) for name in VARIABLES}
    if valid.any():
        heights = levels.on_heights(
            column_dataset(columns, valid), coefficients, z, VARIABLES, "geom_height")
        for name in heights.data_vars:
            data[name][valid] = heights[name].isel(latitude=0, longitude=0).values
    data["w"] = vertical_velocity(data["w"], data["t"], data["p"])
    axes = {**{key: track[key] for key in TRACK_KEYS}, "z": z, "dz": dz_m}
    return {name: {**axes, "data": data[name]} for name in CURTAIN_VARIABLES}


def virtual_curtains(cfg, track):
    """IFS curtains t, w, u, v along the track, or None while the download is missing."""
    files = mars.netcdf_files(cfg)
    if not all(f.exists() for f in files):
        return None
    return sample_curtains(open_ifs(*files), track)


def release_notice(cfg):
    """Release sentence for the download and whether its base is available now."""
    base = mars.base_time(cfg)
    if mars.EMBARGO_HOURS is None:
        in_mars = base + dt.timedelta(hours=mars.RUN_LAG_HOURS)
        sentence = f"IFS run {base:%Y-%m-%d %H} UTC is in MARS from {in_mars:%Y-%m-%d %H:%M} UTC"
    else:
        last_valid = base + dt.timedelta(hours=max(cfg["steps"]))
        release = last_valid + dt.timedelta(hours=mars.EMBARGO_HOURS)
        sentence = (f"IFS valid times up to {last_valid:%Y-%m-%d %H:%M} UTC are released on "
                    f"{release:%Y-%m-%d %H:%M} UTC ({mars.EMBARGO_HOURS} h embargo)")
    return sentence, base <= mars.allowed_base(cfg)
