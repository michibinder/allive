import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pyproj import Geod

FLIGHT_LEVEL_M = 30.48
GEOD = Geod(ellps="WGS84")
TRACK_KEYS = ("time", "lon", "lat", "alt")
TRACK_STEP_S = 60.0
DEFAULT_SPEED_MS = 230.0
MIN_LOOP_SPAN = 4


def read_flight_plan(path):
    """Waypoints (lon, lat, alt in m) of an FTML flight plan."""
    path = Path(path)
    if path.suffix.lower() != ".ftml":
        raise ValueError(f"unsupported flight plan format {path.suffix!r}; supported: .ftml")
    waypoints = [w.attrib for w in ET.parse(path).getroot().iter("Waypoint")]
    return {
        "lon": np.array([float(w["lon"]) for w in waypoints]),
        "lat": np.array([float(w["lat"]) for w in waypoints]),
        "alt": np.array([float(w["flightlevel"]) for w in waypoints]) * FLIGHT_LEVEL_M,
    }


def leg_geometry(plan):
    """(azimuth of every leg, length of every leg, cumulative distance at every waypoint)."""
    lon, lat = plan["lon"], plan["lat"]
    azimuth, _, length = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return azimuth, length, np.concatenate([[0.0], np.cumsum(length)])


def closed_loops(plan):
    """(start, end) waypoint index pairs of closed loops without a loop inside."""
    lon, lat = plan["lon"], plan["lat"]
    n = lon.size
    loops = [(i, j) for i in range(n) for j in range(i + MIN_LOOP_SPAN, n)
             if np.isclose(lon[i], lon[j]) and np.isclose(lat[i], lat[j])]
    return [(i, j) for i, j in loops
            if not any(a >= i and b <= j and (a, b) != (i, j) for a, b in loops)]


def circle_times(plan, departure_s, speed_ms):
    """Unix time of the middle of every circle (innermost closed loop) of the flight plan."""
    along = leg_geometry(plan)[2]
    return np.array([departure_s + 0.5 * (along[i] + along[j]) / speed_ms
                     for i, j in closed_loops(plan)])


def planned_track(plan, departure_s, speed_ms, step_s):
    """Great-circle track at constant ground speed, sampled every step_s, ending at the landing point."""
    lon, lat = plan["lon"], plan["lat"]
    azimuth, length, along = leg_geometry(plan)
    distance = np.arange(0.0, along[-1], speed_ms * step_s)
    distance = np.append(distance, along[-1])
    leg = np.clip(np.searchsorted(along, distance, side="right") - 1, 0, len(length) - 1)
    lon_t, lat_t, _ = GEOD.fwd(lon[leg], lat[leg], azimuth[leg], distance - along[leg])
    return {
        "time": departure_s + distance / speed_ms,
        "lon": lon_t,
        "lat": lat_t,
        "alt": np.interp(distance, along, plan["alt"]),
    }


def planned_flight(plan_path, departure_s, speed_ms):
    """Planned track of an FTML flight plan sampled every TRACK_STEP_S."""
    return planned_track(read_flight_plan(plan_path), departure_s, speed_ms, TRACK_STEP_S)


def extended_grid(time, first, last):
    """The time grid, continued at its first step spacing backwards to first and forwards to last."""
    step = time[1] - time[0]
    before = time[0] - step * np.arange(int(np.ceil(max(time[0] - first, 0.0) / step)), 0, -1)
    after = time[-1] + step * np.arange(1, int(np.ceil(max(last - time[-1], 0.0) / step)) + 1)
    return np.concatenate([before, time, after])


def merge_fixes(track, fixes):
    """Planned track with the measured fixes interpolated in between their first and last time."""
    grid = extended_grid(track["time"], fixes["time"][0], fixes["time"][-1])
    first = min(fixes["time"][0], track["time"][0])
    last = max(fixes["time"][-1], track["time"][-1])
    time = np.union1d(grid[(grid >= first) & (grid <= last)], fixes["time"][[0, -1]])
    measured = (time >= fixes["time"][0]) & (time <= fixes["time"][-1])
    merged = {"time": time}
    for key in TRACK_KEYS[1:]:
        merged[key] = np.interp(time, track["time"], track[key])
        merged[key][measured] = np.interp(time[measured], fixes["time"], fixes[key])
    return merged

