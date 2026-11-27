import datetime as dt

import numpy as np
import pytest
import xarray as xr

from allive import cli, filters, ifs, levels, mars, stream, track
from allive.figure import spread_centres, wind_groups
from allive.utc import time_clusters

DEPARTURE = "2026-11-10T05:30"
DEPARTURE_S = dt.datetime.fromisoformat(DEPARTURE).replace(tzinfo=dt.timezone.utc).timestamp()
BASE = dt.datetime(2026, 11, 10, 0)
WAYPOINTS = ((10.0, 50.0, 0.0), (10.0, 51.0, 340.0), (10.5, 51.5, 360.0), (11.0, 51.5, 360.0),
             (11.0, 52.0, 360.0), (10.5, 52.0, 360.0), (10.5, 51.5, 360.0), (10.0, 50.5, 0.0))
NAMES = ("EDMO", "", "", "", "", "", "", " EDMO ")
FTML = """<?xml version="1.0" encoding="utf-8"?>
<FlightTrack><ListOfWaypoints>
{}
</ListOfWaypoints></FlightTrack>
"""
WAYPOINT = '<Waypoint flightlevel="{2}" lat="{1}" location="{3}" lon="{0}"><Comments></Comments></Waypoint>'
KML = """<?xml version="1.0" encoding="utf-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
{}
<Placemark><name>route</name><LineString><coordinates>9,48,0 10,50,0</coordinates></LineString></Placemark>
</Document></kml>
"""
KML_POINT = "<Placemark><name>{0}</name><Point><coordinates>{1},{2},{3}</coordinates></Point></Placemark>"
KML_POINTS = (("MAH - Way point", 9.5, 48.5, 10000), ("MAH again - Way point", 9.5, 48.5, 10000),
              ("Germany - Country intersection", 10.0, 49.0, 0), ("ALOSO - Way point", 11.0, 50.0, 11500))
SLOW_SPEED_MS = 40.0
FIX_STEP_S = 30.0
STRIDE = 1
LEVEL_STEP_M = STRIDE * stream.DZ_M
TEMPERATURE_K = 250.0
HOURLY_WARMING_K = 10.0
SURFACE_PRESSURE_PA = 1.0e5


def write_plan(path):
    """Write the test flight plan as FTML and return its path."""
    path.write_text(FTML.format("\n".join(WAYPOINT.format(*w, name) for w, name in zip(WAYPOINTS, NAMES))))
    return path


def line(when, lat, lon, alt, numbers):
    """One stream line of bytes: unix time, position, aircraft altitude, stride and numbers."""
    return (f"{when} {lat} {lon} {alt} {STRIDE} " + " ".join(str(n) for n in numbers) + "\n").encode()


def three_blocks(first_km, values):
    """Heights in km, values and uncertainties of the three-block layout."""
    heights = first_km + np.arange(len(values)) * LEVEL_STEP_M / 1000.0
    return [*heights, *values, *np.zeros(len(values))]


def test_flight_plan_to_track(tmp_path):
    """Track times increase, end at the landing point and keep the delay of the cruise fixes after the
    newest one; areas round outward."""
    plan = write_plan(tmp_path / "plan.ftml")
    planned = track.planned_flight(plan, DEPARTURE_S, track.DEFAULT_SPEED_MS)
    assert np.all(np.diff(planned["time"]) > 0)
    assert (planned["lon"][-1], planned["lat"][-1]) == pytest.approx(WAYPOINTS[-1][:2])
    plan_data = track.read_flight_plan(plan)
    kml = tmp_path / "plan.kml"
    kml.write_text(KML.format("\n".join(KML_POINT.format(*point) for point in KML_POINTS)))
    route = track.read_flight_plan(kml)
    assert route["name"] == ["MAH", "ALOSO"] and list(route["lon"]) == [9.5, 11.0]
    assert list(route["lat"]) == [48.5, 50.0] and list(route["alt"]) == [10000.0, 11500.0]
    with pytest.raises(ValueError, match="supported"):
        track.read_flight_plan(tmp_path / "plan.gpx")
    dense = track.planned_track(plan_data, DEPARTURE_S, SLOW_SPEED_MS, FIX_STEP_S)
    delay_s, kept = 420.0, slice(dense["time"].size // 5, -(dense["time"].size // 5))
    fixes = {key: dense[key][kept] for key in track.TRACK_KEYS}
    fixes["time"] = fixes["time"] + delay_s
    expected = DEPARTURE_S + track.leg_geometry(plan_data)[2] / SLOW_SPEED_MS + delay_s
    assert fixes["alt"].max() > track.MIN_TURN_ALTITUDE_M
    parked = {key: np.repeat(dense[key][:1], 3) for key in track.TRACK_KEYS}
    assert track.route_progress(plan_data, parked, DEPARTURE_S, SLOW_SPEED_MS) is None
    progress = track.route_progress(plan_data, fixes, DEPARTURE_S, SLOW_SPEED_MS)
    retimed = track.planned_track(plan_data, DEPARTURE_S, SLOW_SPEED_MS, FIX_STEP_S, progress)
    merged = track.merge_fixes(retimed, fixes, dense["time"])
    later = merged["time"] > fixes["time"][-1]
    assert later.any()
    for key in ("lon", "lat"):
        shifted = np.interp(merged["time"][later] - delay_s, dense["time"], dense[key])
        assert merged[key][later] == pytest.approx(shifted, abs=0.02)
    marks = track.waypoint_marks(plan_data, fixes, progress, DEPARTURE_S, SLOW_SPEED_MS)
    flown = marks["measured"]
    assert marks["label"][:flown.sum()] == ["1", "2", "3", "4", "5", "6"]
    assert marks["time"][flown] == pytest.approx(expected[1:7], abs=120.0)
    ahead = expected > fixes["time"][-1]
    assert marks["label"][flown.sum():] == [str(k) for k in np.flatnonzero(ahead)]
    assert marks["time"][~flown] == pytest.approx(expected[ahead], abs=60.0)
    places = np.c_[plan_data["lon"], plan_data["lat"]][np.r_[1:7, np.flatnonzero(ahead)]]
    assert np.c_[marks["lon"], marks["lat"]] == pytest.approx(places, abs=0.05)
    assert track.waypoint_marks(None, fixes, None, None, SLOW_SPEED_MS)["label"] == ["1", "2", "3", "4"]
    north, west, south, east = (float(v) for v in ifs.mars_area(planned).split("/"))
    assert all(v == round(v) for v in (north, west, south, east))
    assert 0 <= north - (planned["lat"].max() + ifs.AREA_MARGIN_DEG) < ifs.AREA_ROUND_DEG
    assert 0 <= planned["lon"].min() - ifs.AREA_MARGIN_DEG - west < ifs.AREA_ROUND_DEG
    assert south <= planned["lat"].min() - ifs.AREA_MARGIN_DEG and east >= planned["lon"].max()


def test_forecast_request(monkeypatch):
    """Steps bracket the grid, the config of a fixed track is as expected, the base follows the embargo."""
    assert ifs.forecast_steps(5, 7) == [5, 6, 7]
    assert ifs.forecast_steps(89, 91) == [89, 90, 93]
    assert ifs.forecast_steps(145, 146) == [144, 150]
    for first, last in ((-1, 5), (0, 241)):
        with pytest.raises(ValueError):
            ifs.forecast_steps(first, last)
    fixed = {"time": np.array([DEPARTURE_S, DEPARTURE_S + 3600.0]), "lat": np.array([48.2, 49.1]),
             "lon": np.array([10.3, 11.7]), "alt": np.array([0.0, 0.0])}
    cfg = ifs.flight_config("plans/plan.ftml", fixed)
    assert cfg["name"] == "ifs_plan_50_9_47_13_s005-008"
    assert (cfg["base"], cfg["steps"], cfg["area"]) == ("2026-11-10T00:00", [5, 6, 7, 8], "50/9/47/13")
    assert cfg["step_chunks"] == [(5, 8)] and cfg["parallel"] == 3
    first_valid = dt.datetime(2026, 11, 10, 5)
    monkeypatch.setattr(ifs, "utc_now", lambda: dt.datetime(2026, 11, 10, 3))
    assert ifs.default_base(first_valid) == dt.datetime(2026, 11, 9, 12)
    monkeypatch.setattr(mars, "EMBARGO_HOURS", 25)
    assert ifs.default_base(first_valid) == BASE


def test_stream_formats():
    """Line layouts, repeated timestamps, unterminated lines, invalid values and the flight window."""
    when = 1794289322
    meta, heights, values = stream.parse_line(line(when, 49.0, 10.2, 10000, [200, 201, 202]))
    assert heights is None and list(values) == [200, 201, 202] and meta[4] == STRIDE
    meta, heights, values = stream.parse_line(line(when, 0, 0, 0, three_blocks(10.0, [3.0, 4.0, 5.0])))
    assert list(heights) == pytest.approx([10000.0, 10000.0 + LEVEL_STEP_M, 10000.0 + 2 * LEVEL_STEP_M])
    assert list(values) == [3.0, 4.0, 5.0]
    payload = (line(when, 49.0, 10.2, 10000, [1, 2, 3]) + line(when, 49.0, 10.2, 10000, [4, 5, 6])
               + line(when + 1, 0, 0, 0, [7, 8, 9]) + line(when + 2, 49.0, 10.2, 10000, [1, 2, 3])[:-1])
    raw = stream.parse_profiles(payload)
    assert list(raw["time"]) == [when, when + 1] and list(raw["values"][0]) == [4, 5, 6]
    assert list(stream.located(raw)) == [True, False]
    bad = stream.parse_profiles(line(when, 49.0, 10.2, 10000, [250, 0.0, 251, -999, 252, 500, 253, 254]))
    data = stream.build_curtain(bad, stream.VALID_RANGE["T.txt"], False)["data"][0]
    assert list(np.isnan(data)) == [False, True, False, True, False, True, False, False]
    flight = b"".join(line(when + s, 49.0, 10.2, 10000, [1]) for s in (0, 7000, 14000, 21201))
    parsed = {**dict.fromkeys(stream.VALID_RANGE), "T.txt": stream.parse_profiles(flight)}
    kept = stream.flight_profiles(parsed, {"time": np.array([when, when + 1000.0])})
    assert list(kept["T.txt"]["time"] - when) == [0, 7000, 14000]


def test_wind_layout():
    """Wind sub-axes keep the order and spacing inside the panel; profiles cluster by time gap; planned
    times fit the panel, 15 min from the measured and one sub-axes width from earlier planned ones."""
    rng = np.random.default_rng(1)
    xlim, width = (0.0, 10000.0), 1000.0
    for _ in range(20):
        times = np.sort(rng.uniform(*xlim, size=int(rng.integers(1, 6))))
        centres = spread_centres(times, width, xlim)
        assert np.all(np.diff(centres) >= width - 1e-6)
        assert centres[0] >= xlim[0] + width / 2 - 1e-6 and centres[-1] <= xlim[1] - width / 2 + 1e-6
    clusters = time_clusters(np.array([0.0, 800.0, 1600.0, 3000.0, 3500.0]), 1000.0)
    assert [list(c) for c in clusters] == [[0, 1, 2], [3, 4]]
    panel = {"IFS": {"time": np.arange(*xlim, 60.0)},
             "ALIMA": {"time": np.array([5000.0]), "data": np.ones((1, 1))}}
    planned = np.array([0.0, 100.0, 1500.0, 1600.0, 3000.0, 5500.0, 6000.0, 9800.0])
    shown = [group["time"] for group in wind_groups(panel, planned, xlim, width)]
    assert shown == [1500.0, 3000.0, 5000.0, 6000.0]


def write_ifs(cfg):
    """Write a tiny IFS file pair under the names the config expects: isothermal columns that warm by
    HOURLY_WARMING_K per step, so that the temperature row has a contour line to draw."""
    north, west, south, east = (float(v) for v in cfg["area"].split("/"))
    times = [BASE + dt.timedelta(hours=h) for h in cfg["steps"]]
    coords = {"level": np.arange(1.0, 138.0), "latitude": np.linspace(south, north, 3),
              "longitude": np.linspace(west, east, 3)}
    shape = (len(times), 137, 3, 3)
    ml = xr.Dataset({name: (("time", "level", "latitude", "longitude"), np.full(shape, value, "float32"))
                     for name, value in (("t", TEMPERATURE_K), ("w", 0.0), ("u", 0.0), ("v", 0.0))},
                    {"time": times, **coords})
    ml["t"] = ml["t"] + HOURLY_WARMING_K * xr.DataArray(np.arange(len(times), dtype="float32"), dims="time")
    p_times = [BASE, *times]
    p_coords = {**coords, "level": [1.0], "time": p_times}
    p_shape = (len(p_times), 1, 3, 3)
    p = xr.Dataset({"z": (("time", "level", "latitude", "longitude"), np.zeros(p_shape, "float32")),
                    "lnsp": (("time", "level", "latitude", "longitude"),
                             np.full(p_shape, np.log(SURFACE_PRESSURE_PA), "float32"))}, p_coords)
    file_ml, file_ml_p = mars.netcdf_files(cfg)
    file_ml.parent.mkdir(parents=True, exist_ok=True)
    ml.to_netcdf(file_ml)
    p.to_netcdf(file_ml_p)
    return file_ml, file_ml_p


def test_pipeline_and_levels(tmp_path, monkeypatch, capsys):
    """The plot command writes one figure per T' filter with the IFS layer, leg means split at the marks;
    model levels of an isothermal column are hydrostatic."""
    monkeypatch.setenv("ALLIVE_DATA", str(tmp_path))
    plan = write_plan(tmp_path / "plan.ftml")
    planned = track.planned_flight(plan, DEPARTURE_S, track.DEFAULT_SPEED_MS)
    files = write_ifs(ifs.flight_config(plan, planned, BASE))
    source = tmp_path / "source"
    source.mkdir()
    levels_m = 500.0 * np.arange(60) + 5000.0
    for name, base_value in (("T.txt", 220.0), ("w.txt", 0.5), ("u.txt", 20.0), ("v.txt", -10.0)):
        values = base_value + np.sin(levels_m / 3000.0)
        lines = b"".join(
            line(int(DEPARTURE_S) + 600 + 30 * k, planned["lat"][5], planned["lon"][5], 10000,
                 values if name in ("T.txt", "w.txt") else three_blocks(5.0, values))
            for k in range(3))
        (source / name).write_bytes(lines)
    cli.main(["plot", str(source), "--flight", str(plan), "--departure", DEPARTURE])
    cli.main(["plot", str(source), "--flight", str(plan), "--departure", DEPARTURE,
              "--tp-filter", "leg-mean", "--panels", "tp"])
    assert len(list((tmp_path / "figures").glob("*.png"))) == 2
    assert "IFS" not in capsys.readouterr().out
    primes = filters.leg_mean_primes(np.array([[1.0, np.nan], [3.0, np.nan], [5.0, 2.0]]),
                                     np.array([0.0, 1.0, 2.0]), np.array([2.0]))
    assert np.allclose(primes, [[-1.0, np.nan], [1.0, np.nan], [0.0, 0.0]], equal_nan=True)
    ds = ifs.open_ifs(*files).isel(time=[0], latitude=[0], longitude=[0]).copy(deep=True)
    z = np.arange(0.0, 60001.0, 1000.0)
    result = levels.on_heights(ds, levels.coefficients(), z, ("t", "p"), "geom_height")
    column = result.isel(time=0, latitude=0, longitude=0)
    geopotential = levels.EARTH_RADIUS * z / (levels.EARTH_RADIUS + z)
    expected = SURFACE_PRESSURE_PA * np.exp(-levels.GRAVITY * geopotential / (levels.R_DRY * TEMPERATURE_K))
    assert np.allclose(column["t"].values[1:], TEMPERATURE_K, atol=1e-3)
    assert np.allclose(column["p"].values, expected, rtol=0.05)
