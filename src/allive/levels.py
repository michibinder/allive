from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

GRAVITY = 9.80665
R_DRY = 287.06
EARTH_RADIUS = 6371229
Q_VIRTUAL = 0.609133
LAPSE_RATE = 0.0065
FIELD_DIMS = ["time", "level", "latitude", "longitude"]
P_ATTRS = {"long_name": "Pressure", "units": "Pa", "standard_name": "air_pressure"}
COEFFICIENT_FILE = Path(__file__).with_name("l137_levels.csv")


def normalize(ds):
    """Rename new-style CDS dims/coords to 'time'/'level' and order the dims."""
    ren = {}
    if "model_level" in ds.dims:
        ren["model_level"] = "level"
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ren["valid_time"] = "time"
    ds = ds.rename(ren)
    ds = ds.drop_vars([v for v in ("expver", "number") if v in ds.variables])
    return ds.transpose("time", "level", "latitude", "longitude")


def virtual_temperature(ds):
    """Virtual temperature from t and q, the dry t when q is absent."""
    return ds["t"] * (1.0 + Q_VIRTUAL * ds["q"]) if "q" in ds else ds["t"].copy()


def compute_z_level(ds, p_half):
    """Geopotential and geopotential height at all full levels from t/q/lnsp (ECMWF recipe)."""
    z_h = ds["z"][:, 0, :, :].copy()
    ds["t_moist"] = virtual_temperature(ds)

    for i in range(136, 0, -1):
        ph_levplusone = p_half[:, i + 1, :, :].values
        ph_lev = p_half[:, i, :, :].values

        dlog_p = np.log(ph_levplusone / ph_lev)
        alpha = 1.0 - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)

        t_level = ds["t_moist"][:, i, :, :] * R_DRY
        ds["z"][:, i, :, :] = z_h + (t_level * alpha)
        z_h = z_h + (t_level * dlog_p)

    i = 0
    alpha = np.log(2)
    t_level = ds["t_moist"][:, i, :, :] * R_DRY
    ds["z"][:, i, :, :] = z_h + (t_level * alpha)

    ds["geop_height"] = ds["z"] / GRAVITY
    return ds


def upward_columns(da, it):
    """Time index it of a model-level field as float64 (level, lat, lon), lowest model level first."""
    return da.values[it, ::-1].astype(np.float64)


def interp_columns(z_new, zp, fps):
    """np.interp of every column of each fp(zp) onto z_new along axis 0 (zp ascending), vectorised;
    bit-identical to np.interp, held end values included."""
    z = z_new.reshape((-1,) + (1,) * (zp.ndim - 1))
    j = np.zeros(z.shape[:1] + zp.shape[1:], dtype=np.intp)
    for zk in zp:
        j += zk <= z
    j = np.clip(j - 1, 0, len(zp) - 2)
    z0 = np.take_along_axis(zp, j, axis=0)
    dz = np.take_along_axis(zp, j + 1, axis=0) - z0
    offset = z - z0
    below, above = z < zp[0], z >= zp[-1]
    out = []
    for fp in fps:
        f0 = np.take_along_axis(fp, j, axis=0)
        inner = (np.take_along_axis(fp, j + 1, axis=0) - f0) / dz * offset + f0
        out.append(np.where(below, fp[0], np.where(above, fp[-1], inner)))
    return out


def extrapolated_t_p(z, zp, t, tv, p):
    """t and p below the lowest full level (LAPSE_RATE) and p above the top one (isothermal), both
    hydrostatic with the end level's virtual temperature tv and q held."""
    t_below = t[0] + LAPSE_RATE * np.maximum(zp[0] - z, 0.0)
    p_below = p[0] * (t_below / t[0]) ** (GRAVITY * t[0] / (R_DRY * LAPSE_RATE * tv[0]))
    p_above = p[-1] * np.exp(-GRAVITY * np.maximum(z - zp[-1], 0.0) / (R_DRY * tv[-1]))
    return t_below, p_below, p_above


def interp_ds_vertically(ds, z_new, alt_var, vars):
    """Interpolate model-level fields column-wise to the height grid z_new; outside the source column
    t and p are extrapolated by extrapolated_t_p, the other fields keep their end values."""
    tv = virtual_temperature(ds)
    nt = ds.sizes["time"]
    data = {var: np.empty((nt, len(z_new), ds.sizes["latitude"], ds.sizes["longitude"])) for var in vars}
    z = z_new[:, None, None]
    for it in range(nt):
        zp = upward_columns(ds[alt_var], it)
        fields = dict(zip(vars, interp_columns(z_new, zp, [upward_columns(ds[var], it) for var in vars])))
        t_below, p_below, p_above = extrapolated_t_p(
            z, zp, upward_columns(ds["t"], it), upward_columns(tv, it), upward_columns(ds["p"], it))
        below, above = z < zp[0], z > zp[-1]
        if "t" in fields:
            fields["t"] = np.where(below, t_below, fields["t"])
        if "p" in fields:
            fields["p"] = np.where(below, p_below, np.where(above, p_above, fields["p"]))
        for var in vars:
            data[var][it] = fields[var]
    coords = {"time": ds["time"], "level": z_new, "latitude": ds["latitude"], "longitude": ds["longitude"]}
    return xr.Dataset({var: (FIELD_DIMS, data[var], ds[var].attrs) for var in vars}, coords=coords,
                      attrs=ds.attrs)


def on_heights(ds, ml_coeff, z_new, vars, alt_var):
    """Pressure and geometric height of one in-memory time block, interpolated to the heights z_new."""
    lnsp = ds["lnsp"][:, 0, :, :].drop_vars("level").expand_dims(dim={"level": 138}, axis=1)
    dims = np.shape(lnsp)

    a = xr.DataArray(ml_coeff["a [Pa]"]).rename({"dim_0": "level"})
    a = a.expand_dims(dim={"time": dims[0], "latitude": dims[2], "longitude": dims[3]}, axis=[0, 2, 3])
    b = xr.DataArray(ml_coeff["b"]).rename({"dim_0": "level"})
    b = b.expand_dims(dim={"time": dims[0], "latitude": dims[2], "longitude": dims[3]}, axis=[0, 2, 3])

    p_half = a + b * np.exp(lnsp.values)

    ds = compute_z_level(ds, p_half)
    ds["geom_height"] = EARTH_RADIUS * ds["geop_height"] / (EARTH_RADIUS - ds["geop_height"])

    i = np.arange(0, 137)
    ds["p"] = ds["t"].copy()
    ds["p"].attrs = dict(P_ATTRS)
    ds["p"][:, i, :, :] = (p_half[:, i + 1, :, :].values + p_half[:, i, :, :].values) / 2

    vars = [v for v in vars if v in ds]
    ds = interp_ds_vertically(ds, z_new, alt_var, vars)

    for var_name in vars:
        ds[var_name] = ds[var_name].astype("float32")
    return ds


def coefficients():
    """The L137 model-level coefficient table."""
    return pd.read_csv(COEFFICIENT_FILE)
