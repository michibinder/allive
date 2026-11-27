import datetime as dt
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from allive.figure import FIELDS, cell_edges, contour_key, finite_profiles, render
from allive.filters import DEFAULT_TP_FILTER, TP_FILTERS
from allive.ifs import downloaded_config, flight_config, virtual_curtains
from allive.mars import netcdf_files
from allive.paths import figure_dir
from allive.stream import (VALID_RANGE, archive, build_curtain, describe, fetch_all, flight_profiles,
                           flight_stamp, is_url, measured_fixes, newest_profile, parse_profiles)
from allive.track import (TRACK_KEYS, TRACK_STEP_S, merge_fixes, planned_track, read_flight_plan,
                          route_progress, waypoint_marks)
from allive.utc import log_utc, utc_datetime, utc_seconds

IFS_VARIABLES = {"T.txt": "t", "w.txt": "w", "u.txt": "u", "v.txt": "v"}
TIME_MARGIN = 0.02
MEASURED_LEAD_S = 1800.0
DEFAULT_ZLIM_KM = (20.0, 85.0)
DEFAULT_WATCH_S = 60.0
DEFAULT_PANELS = ("t", "tp", "w", "u", "v")


def to_primes(curtain, tp_filter, mark_s):
    """The curtain with its data replaced by T' of the chosen filter; mark_s are the waypoint times."""
    return {**curtain, "data": TP_FILTERS[tp_filter]["primes"](curtain, mark_s)}


def panel_layers(curtains, ifs, panels, tp_filter, mark_s):
    """{panel: {source: curtain}} of the IFS and ALIMA curtains with valid values each panel draws,
    T' of the chosen filter where asked."""
    layers = {}
    for key in panels:
        spec = FIELDS[key]
        sources = {"IFS": ifs[IFS_VARIABLES[spec["file"]]] if ifs else None,
                   "ALIMA": curtains[spec["file"]]}
        shown = {source: to_primes(curtain, tp_filter, mark_s) if spec["primes"] else curtain
                 for source, curtain in sources.items() if curtain is not None}
        layers[key] = {source: curtain for source, curtain in shown.items()
                       if np.isfinite(curtain["data"]).any()}
    return layers


def measured_span(curtains):
    """(left edge of the first, right edge of the last) cell of the profiles with valid values over all
    files, or None."""
    times = [c["time"][finite_profiles(c)] for c in curtains.values() if c is not None]
    edges = [cell_edges(t) for t in times if t.size]
    if not edges:
        return None
    return min(left[0] for left, _ in edges), max(right[-1] for _, right in edges)


def time_limits(span, track, hours):
    """(start, end) of the time axis in unix seconds: from MEASURED_LEAD_S before the measurements (else the
    track start) to the later end of both, else now; the last hours end at the newest measurement if any."""
    spans = [] if span is None else [span]
    if track is not None:
        spans.append((track["time"][0], track["time"][-1]))
    if not spans:
        now = time.time()
        return now - 3600.0, now
    start = track["time"][0] if span is None else span[0] - MEASURED_LEAD_S
    end = max(s[1] for s in spans)
    if hours:
        end = end if span is None else span[1]
        start = end - hours * 3600.0
    return start, end + TIME_MARGIN * (end - start)


def cursor_time(when, window_s, latest, track):
    """Cursor in unix seconds: given (the window centre, with a log line, for a clock time outside
    the window), else the newest measured profile, else MEASURED_LEAD_S after the track's start or the
    window start, whichever is later."""
    if when is None:
        return latest if latest is not None else max(track["time"][0] + MEASURED_LEAD_S, window_s[0])
    if isinstance(when, dt.datetime):
        return utc_seconds(when)
    window = tuple(utc_datetime(s) for s in window_s)
    for day in (window[0].date(), window[1].date()):
        candidate = dt.datetime.combine(day, when)
        if window[0] <= candidate <= window[1]:
            return utc_seconds(candidate)
    log_utc(f"--time {when:%H:%M} lies outside the shown time window: cursor at its centre")
    return utc_seconds(window[0] + (window[1] - window[0]) / 2)


def flight_note(args, cfg, ifs):
    """Lines of the note on the flight: take-off, flight plan and, with IFS curtains, their run."""
    lines = [f"take-off {utc_datetime(args.departure):%d %b %Y %H:%M} UTC", Path(args.flight).stem]
    if ifs is not None:
        lines.append(f"IFS run {dt.datetime.fromisoformat(cfg['base']):%d %b %Y %H} UTC")
    return lines


def download_command(args):
    """Command line that downloads the IFS data of the flight."""
    return (f"allive download {args.flight} "
            f"--departure {utc_datetime(args.departure):%Y-%m-%dT%H:%M} --speed {args.speed:g}")


def ifs_layer(args, cfg, track):
    """IFS curtains along the track, or None with a log line on how to get them."""
    files = netcdf_files(cfg)
    try:
        ifs = virtual_curtains(cfg, track)
    except (OSError, KeyError, ValueError) as err:
        reason = str(err).splitlines()[0] if str(err) else type(err).__name__
        log_utc(f"IFS files {files[0]} and {files[1]} unreadable ({reason}); delete them and run "
                f"the download again: {download_command(args)}")
        return None
    if ifs is None:
        log_utc(f"no IFS data in {files[0].parent}; download it with: {download_command(args)}")
    return ifs


def track_point(path, when_s):
    """Position {"lon", "lat", "alt"} at a time along a track or measured fixes, or None."""
    if path is None or when_s is None:
        return None
    return {key: float(np.interp(when_s, path["time"], path[key])) for key in TRACK_KEYS[1:]}


def flight_data(args, plan, planned, fixes, progress):
    """(track, IFS curtains, note lines) of the planned flight: the route timed by the progress along it with
    the fixes merged in, the IFS run chosen for the planned track; no track, curtains or lines without one."""
    if planned is None:
        return None, None, []
    track = planned_track(plan, args.departure, args.speed, TRACK_STEP_S, progress)
    if fixes is not None:
        track = merge_fixes(track, fixes, planned["time"])
    cfg = downloaded_config(args.flight, planned) or flight_config(args.flight, planned)
    ifs = ifs_layer(args, cfg, track)
    return track, ifs, flight_note(args, cfg, ifs)


def output_path(args, stamp):
    """Figure path: one per flight plan and departure, else one per measured flight; a T' filter other
    than the default adds its name."""
    if args.flight:
        stem = f"{Path(args.flight).stem}_{utc_datetime(args.departure):%Y%m%d-%H%M}"
    else:
        stem = stamp or "nodata"
    suffix = "" if args.tp_filter == DEFAULT_TP_FILTER else f"_{args.tp_filter}"
    return figure_dir() / f"allive_{stem}{suffix}.png"


def process(payloads, args):
    """Parse, archive, build and render one set of file payloads."""
    parsed = {name: parse_profiles(payload) for name, payload in payloads.items()}
    stamp = flight_stamp(parsed)
    if is_url(args.source) and stamp:
        archive(payloads, stamp)
    plan = read_flight_plan(args.flight) if args.flight else None
    planned = planned_track(plan, args.departure, args.speed, TRACK_STEP_S) if plan else None
    if planned is not None:
        parsed = flight_profiles(parsed, planned)
    curtains = {name: build_curtain(parsed[name], valid, args.above_aircraft)
                for name, valid in VALID_RANGE.items()}
    if not args.ifs_only:
        for name in VALID_RANGE:
            log_utc(describe(name, parsed[name], curtains[name]))
    fixes = measured_fixes(parsed)
    progress = route_progress(plan, fixes, args.departure, args.speed) if plan and fixes else None
    track, ifs, note = flight_data(args, plan, planned, fixes, progress)
    waypoints = waypoint_marks(plan, fixes, progress, args.departure, args.speed)
    layers = panel_layers(curtains, ifs, args.panels, args.tp_filter, waypoints["time"])
    latest = newest_profile(parsed)
    xlim_s = time_limits(measured_span(curtains), track, args.hours)
    has_data = any(layers.values()) or latest is not None
    cursor_s = cursor_time(args.time, xlim_s, latest, track) if has_data else None
    aircraft = track_point(fixes if track is None else track, cursor_s)
    view = {"xlim_s": xlim_s, "zlim_km": tuple(args.zlim), "cursor_s": cursor_s,
            "aircraft_km": aircraft and aircraft["alt"] / 1000.0, "waypoints": waypoints, "plan": plan,
            "fixes": fixes, "tp_filter": args.tp_filter, "position": track_point(fixes, cursor_s)}
    path = render(layers, view, note + contour_key(layers), output_path(args, stamp))
    log_utc(f"saved {path}")


def watch(args):
    """Fetch, draw and optionally keep watching the ALIMA files."""
    previous = None
    try:
        while True:
            if args.ifs_only:
                payloads = dict.fromkeys(VALID_RANGE, b"")
            else:
                payloads = fetch_all(args.source, previous)
            if payloads != previous:
                try:
                    process(payloads, args)
                    previous = payloads
                except Exception:
                    if args.watch is None:
                        raise
                    log_utc(f"redraw failed, retried next round\n{traceback.format_exc().rstrip()}",
                            sys.stderr)
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass
