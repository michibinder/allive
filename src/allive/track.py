import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pyproj import Geod

FLIGHT_LEVEL_M = 30.48
GEOD = Geod(ellps="WGS84")
TRACK_KEYS = ("time", "lon", "lat", "alt")
MARK_KEYS = TRACK_KEYS[:3]
TRACK_STEP_S = 60.0
DEFAULT_SPEED_MS = 230.0
ROUTE_STEP_M = 2000.0
MAX_OFF_ROUTE_M = 30000.0
MAX_SPEED_FACTOR = 1.5
REACH_SLACK_M = 20000.0
KML_NS = {"k": "http://www.opengis.net/kml/2.2"}
KML_WAYPOINT_SUFFIX = " - Way point"
KML_SKIPPED_SUFFIX = "Country intersection"
MIN_LEG_M = 2000.0
MIN_TURN_DEG = 35.0
MIN_TURN_ALTITUDE_M = 7000.0
TURN_MERGE_S = 600.0
UNPLANNED_LABEL = "•"
FLIGHT_PLAN_SUFFIXES = (".ftml", ".kml")


def read_ftml(path):
    """Waypoints (lon, lat, alt in m, name) of an FTML flight plan."""
    waypoints = [w.attrib for w in ET.parse(path).getroot().iter("Waypoint")]
    return {
        "lon": np.array([float(w["lon"]) for w in waypoints]),
        "lat": np.array([float(w["lat"]) for w in waypoints]),
        "alt": np.array([float(w["flightlevel"]) for w in waypoints]) * FLIGHT_LEVEL_M,
        "name": [w.get("location", "").strip() for w in waypoints],
    }


def read_kml(path):
    """Waypoints (lon, lat, alt in m, name) of the point placemarks of a KML route, border crossings and
    repeated points left out."""
    rows = []
    for mark in ET.parse(path).getroot().iterfind(".//k:Placemark", KML_NS):
        name = (mark.findtext("k:name", "", KML_NS) or "").strip()
        point = mark.findtext(".//k:Point/k:coordinates", None, KML_NS)
        if point is None or name.endswith(KML_SKIPPED_SUFFIX):
            continue
        lon, lat, alt = (float(v) for v in point.strip().split(",")[:3])
        if not rows or rows[-1][:2] != (lon, lat):
            rows.append((lon, lat, alt, name.removesuffix(KML_WAYPOINT_SUFFIX)))
    if not rows:
        raise ValueError(f"no way points in the KML flight plan {Path(path).name}")
    lon, lat, alt, name = zip(*rows)
    return {"lon": np.array(lon), "lat": np.array(lat), "alt": np.array(alt), "name": list(name)}


def read_flight_plan(path):
    """Waypoints (lon, lat, alt in m, name) of a flight plan, the reader chosen by the file suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in FLIGHT_PLAN_SUFFIXES:
        raise ValueError(f"unsupported flight plan format {path.suffix!r}; supported: "
                         f"{', '.join(FLIGHT_PLAN_SUFFIXES)}")
    return read_kml(path) if suffix == ".kml" else read_ftml(path)


def waypoint_labels(plan):
    """Label of every waypoint as the Mission Support System numbers it: its index, counted from 0."""
    return [str(index) for index in range(len(plan["name"]))]


def leg_geometry(plan):
    """(azimuth of every leg, length of every leg, cumulative distance at every waypoint)."""
    lon, lat = plan["lon"], plan["lat"]
    azimuth, _, length = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return azimuth, length, np.concatenate([[0.0], np.cumsum(length)])


def route_times(distance, progress, departure_s, speed_ms):
    """Unix time at distances along the planned route: interpolated between the on-route fixes of progress, at
    the ground speed before and after them, from the departure without progress."""
    if progress is None:
        return departure_s + distance / speed_ms
    times, along = progress
    first = np.r_[True, np.diff(along) > 0]
    passed = np.interp(distance, along[first], times[first])
    before = times[0] - (along[0] - distance) / speed_ms
    after = times[-1] + (distance - along[-1]) / speed_ms
    return np.where(distance < along[0], before, np.where(distance > along[-1], after, passed))


def planned_track(plan, departure_s, speed_ms, step_s, progress=None):
    """Great-circle track sampled every step_s at the ground speed, ending at the landing point, timed by the
    progress along the route (route_times)."""
    lon, lat = plan["lon"], plan["lat"]
    azimuth, length, along = leg_geometry(plan)
    distance = np.arange(0.0, along[-1], speed_ms * step_s)
    distance = np.append(distance, along[-1])
    leg = np.clip(np.searchsorted(along, distance, side="right") - 1, 0, len(length) - 1)
    lon_t, lat_t, _ = GEOD.fwd(lon[leg], lat[leg], azimuth[leg], distance - along[leg])
    return {
        "time": route_times(distance, progress, departure_s, speed_ms),
        "lon": lon_t,
        "lat": lat_t,
        "alt": np.interp(distance, along, plan["alt"]),
    }


def planned_flight(plan_path, departure_s, speed_ms):
    """Planned track of a flight plan file sampled every TRACK_STEP_S."""
    return planned_track(read_flight_plan(plan_path), departure_s, speed_ms, TRACK_STEP_S)


def extended_grid(time, first, last):
    """The time grid, continued at its first step spacing backwards to first and forwards to last."""
    step = time[1] - time[0]
    before = time[0] - step * np.arange(int(np.ceil(max(time[0] - first, 0.0) / step)), 0, -1)
    after = time[-1] + step * np.arange(1, int(np.ceil(max(last - time[-1], 0.0) / step)) + 1)
    return np.concatenate([before, time, after])


def merge_fixes(track, fixes, grid):
    """Track with the measured fixes interpolated in between their first and last time, there and in a gap
    to the track at the times of the grid continued at its first step."""
    first, last = fixes["time"][0], fixes["time"][-1]
    grid = extended_grid(grid, first, last)
    outside = (track["time"] < first) | (track["time"] > last)
    filled = (grid >= min(first, track["time"][-1])) & (grid <= max(last, track["time"][0]))
    time = np.union1d(np.r_[track["time"][outside], grid[filled]], [first, last])
    measured = (time >= first) & (time <= last)
    merged = {"time": time}
    for key in TRACK_KEYS[1:]:
        merged[key] = np.interp(time, track["time"], track[key])
        merged[key][measured] = np.interp(time[measured], fixes["time"], fixes[key])
    return merged


def route_progress(plan, fixes, departure_s, speed_ms):
    """(time, distance along the planned route) of the fixes at cruise altitude that lie on the route, never
    decreasing, or None without any."""
    route = planned_track(plan, 0.0, 1.0, ROUTE_STEP_M)
    along = route["time"]
    cruise = fixes["alt"] >= MIN_TURN_ALTITUDE_M
    times, found = [], []
    for when, lon, lat in zip(fixes["time"][cruise], fixes["lon"][cruise], fixes["lat"][cruise]):
        _, _, off = GEOD.inv(np.full(along.size, lon), np.full(along.size, lat), route["lon"], route["lat"])
        near = off <= MAX_OFF_ROUTE_M
        if found:
            reach = found[-1] + (when - times[-1]) * speed_ms * MAX_SPEED_FACTOR + REACH_SLACK_M
            near &= (along >= found[-1]) & (along <= reach)
            cost = off
        else:
            near &= off <= off.min() + ROUTE_STEP_M
            cost = np.abs(route_times(along, None, departure_s, speed_ms) - when)
        if near.any():
            times.append(when)
            found.append(along[np.flatnonzero(near)[np.argmin(cost[near])]])
    return (np.array(times), np.array(found)) if times else None


def course_change(before, after):
    """Absolute change between two courses in degrees, at most 180."""
    return np.abs((after - before + 180.0) % 360.0 - 180.0)


def measured_turns(fixes):
    """Fixes (time, lon, lat) where the aircraft started to turn: course changes above MIN_TURN_DEG at
    cruise altitude, those within TURN_MERGE_S of the previous one merged into it."""
    keep = [0]
    for i in range(1, fixes["time"].size):
        last = keep[-1]
        leg = GEOD.inv(fixes["lon"][last], fixes["lat"][last], fixes["lon"][i], fixes["lat"][i])[2]
        if leg >= MIN_LEG_M:
            keep.append(i)
    keep = np.array(keep)
    lon, lat = fixes["lon"][keep], fixes["lat"][keep]
    course = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[0]
    change = course_change(course[:-1], course[1:])
    change[1:] = np.maximum(change[1:], course_change(course[:-2], course[2:]))
    turning = keep[1:-1][(change > MIN_TURN_DEG) & (fixes["alt"][keep[1:-1]] >= MIN_TURN_ALTITUDE_M)]
    first = turning[np.r_[True, np.diff(fixes["time"][turning]) > TURN_MERGE_S]] if turning.size else turning
    return {key: fixes[key][first] for key in MARK_KEYS}


def match_plan(turns, plan):
    """Index of the planned waypoint every turn belongs to (-1 for none): the nearest one within
    MAX_OFF_ROUTE_M that comes after the previous match."""
    matched, last = [], -1
    for lon, lat in zip(turns["lon"], turns["lat"]):
        off = GEOD.inv(np.full(plan["lon"].size, lon), np.full(plan["lon"].size, lat),
                       plan["lon"], plan["lat"])[2]
        off[:last + 1] = np.inf
        best = int(np.argmin(off))
        matched.append(best if off[best] <= MAX_OFF_ROUTE_M else -1)
        last = max(last, matched[-1])
    return np.array(matched, int)


def passed_waypoints(plan, fixes, times):
    """True for every planned waypoint that a fix at cruise altitude passed within MAX_OFF_ROUTE_M, within
    TURN_MERGE_S of its time."""
    cruise = fixes["alt"] >= MIN_TURN_ALTITUDE_M
    passed = np.zeros(times.size, bool)
    for k, (lon, lat, when) in enumerate(zip(plan["lon"], plan["lat"], times)):
        close = cruise & (np.abs(fixes["time"] - when) <= TURN_MERGE_S)
        if close.any():
            off = GEOD.inv(np.full(close.sum(), lon), np.full(close.sum(), lat),
                           fixes["lon"][close], fixes["lat"][close])[2]
            passed[k] = off.min() <= MAX_OFF_ROUTE_M
    return passed


def waypoint_marks(plan, fixes, progress, departure_s, speed_ms):
    """Marks {"time", "lon", "lat", "label", "measured"} in time order: the turns in the fixes and the planned
    waypoints the aircraft passed (measured), and the planned waypoints still ahead of the newest fix."""
    empty = dict.fromkeys(MARK_KEYS, np.array([]))
    turns = empty if fixes is None else measured_turns(fixes)
    if plan is None:
        labels = [str(k + 1) for k in range(turns["time"].size)]
        extra, measured = empty, np.array([], bool)
    else:
        numbers = waypoint_labels(plan)
        belongs = match_plan(turns, plan)
        labels = [numbers[k] if k >= 0 else UNPLANNED_LABEL for k in belongs]
        times = route_times(leg_geometry(plan)[2], progress, departure_s, speed_ms)
        index = np.arange(times.size)
        free = ~np.isin(index, belongs)
        passed = free & (False if fixes is None else passed_waypoints(plan, fixes, times))
        flown = max(belongs.max(initial=-1), index[passed].max(initial=-1))
        newest = -np.inf if fixes is None else fixes["time"][-1]
        ahead = free & ~passed & (index > flown) & (times > newest)
        shown = np.flatnonzero(passed | ahead)
        labels += [numbers[k] for k in shown]
        extra = {"time": times[shown], **{key: plan[key][shown] for key in MARK_KEYS[1:]}}
        measured = passed[shown]
    marks = {key: np.concatenate([turns[key], extra[key]]) for key in MARK_KEYS}
    measured = np.concatenate([np.ones(turns["time"].size, bool), measured])
    order = np.argsort(marks["time"], kind="stable")
    return {**{key: value[order] for key, value in marks.items()}, "label": [labels[k] for k in order],
            "measured": measured[order]}
