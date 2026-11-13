import datetime as dt
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from allive.figure import FIELDS, cell_edges, render
from allive.filters import vertical_primes
from allive.ifs import downloaded_config, flight_config, virtual_curtains
from allive.mars import netcdf_files
from allive.paths import figure_dir
from allive.stream import (VALID_RANGE, archive, build_curtain, describe, fetch_all, flight_profiles,
                           flight_stamp, height_range_km, is_url, measured_fixes, newest_profile,
                           parse_profiles)
from allive.track import circle_times, merge_fixes, planned_flight, read_flight_plan
from allive.utc import log_utc, utc_datetime, utc_seconds

IFS_VARIABLES = {"T.txt": "t", "w.txt": "w", "u.txt": "u", "v.txt": "v"}
TIME_MARGIN = 0.01
ZLIM_FALLBACK_KM = (15.0, 90.0)
ZLIM_STEP_KM = 5.0
DEFAULT_WATCH_S = 60.0
DEFAULT_PANELS = ("t", "tp", "w", "u", "v")


def to_primes(curtain):
    """The curtain with its data replaced by the vertical high-pass T' (short profiles blanked)."""
    return {**curtain, "data": vertical_primes(curtain["data"], curtain["dz"])}


def panel_layers(curtains, ifs, panels):
    """{panel: {source: curtain}} of the IFS and ALIMA curtains with valid values each panel draws,
    T' where asked."""
    layers = {}
    for key in panels:
        spec = FIELDS[key]
        sources = {"IFS": ifs[IFS_VARIABLES[spec["file"]]] if ifs else None,
                   "ALIMA": curtains[spec["file"]]}
        shown = {source: to_primes(curtain) if spec["primes"] else curtain
                 for source, curtain in sources.items() if curtain is not None}
        layers[key] = {source: curtain for source, curtain in shown.items()
                       if np.isfinite(curtain["data"]).any()}
    return layers


def source_curtains(layers, source):
    """Curtains of one source over all panels."""
    return [panel[source] for panel in layers.values() if source in panel]


def measured_span(parsed):
    """(left edge of the first, right edge of the last) profile cell over all files, or None."""
    edges = [cell_edges(raw["time"]) for raw in parsed.values() if raw is not None]
    if not edges:
        return None
    return min(left[0] for left, _ in edges), max(right[-1] for _, right in edges)


def time_limits(span, track, hours):
    """(start, end) of the time axis in unix seconds: track and measurements, else now; the last
    hours end at the newest measurement, else at the end of the track."""
    spans = [] if span is None else [span]
    if track is not None:
        spans.append((track["time"][0], track["time"][-1]))
    if not spans:
        now = time.time()
        return now - 3600.0, now
    start, end = min(s[0] for s in spans), max(s[1] for s in spans)
    if hours:
        end = end if span is None else span[1]
        start = end - hours * 3600.0
    return start, end + TIME_MARGIN * (end - start)


def height_limits(layers, window_s, zlim):
    """Altitude limits in km: given, else the measured range widened to ZLIM_STEP_KM, else IFS."""
    if zlim:
        return tuple(zlim)
    measured = source_curtains(layers, "ALIMA")
    ranges = [r for r in (height_range_km(c, window_s) for c in measured) if r]
    if ranges:
        low = np.floor(min(r[0] for r in ranges) / ZLIM_STEP_KM) * ZLIM_STEP_KM
        high = np.ceil(max(r[1] for r in ranges) / ZLIM_STEP_KM) * ZLIM_STEP_KM
        return (low, high) if high > low else (low, low + ZLIM_STEP_KM)
    ifs = source_curtains(layers, "IFS")
    if ifs:
        return ifs[0]["z"][0] / 1000.0, ifs[0]["z"][-1] / 1000.0
    return ZLIM_FALLBACK_KM


def cursor_time(when, window_s, latest, track):
    """Cursor in unix seconds: given (the window centre, with a log line, for a clock time outside
    the window), else the newest measured profile, else the track's end."""
    if when is None:
        return latest[0] if latest is not None else track["time"][-1]
    if isinstance(when, dt.datetime):
        return utc_seconds(when)
    window = tuple(utc_datetime(s) for s in window_s)
    for day in (window[0].date(), window[1].date()):
        candidate = dt.datetime.combine(day, when)
        if window[0] <= candidate <= window[1]:
            return utc_seconds(candidate)
    log_utc(f"--time {when:%H:%M} lies outside the shown time window: cursor at its centre")
    return utc_seconds(window[0] + (window[1] - window[0]) / 2)


def measurement_note(latest):
    """Blocks of lines of the note on the newest measured profile and position, or None."""
    if latest is None:
        return None
    when, position = latest
    lines = [f"ALIMA latest profile {utc_datetime(when):%H:%M:%S} UTC"]
    if position is not None:
        lat, lon, alt = position
        north, east = "N" if lat >= 0 else "S", "E" if lon >= 0 else "W"
        lines += [f"{abs(lat):.2f}°{north} {abs(lon):.2f}°{east}", f"aircraft at {alt / 1000.0:.1f} km"]
    return [tuple(lines)]


def flight_note(args, ifs_lines):
    """Blocks of lines of the note on the flight plan, departure, ground speed and IFS layer."""
    flight = (f"flight plan {Path(args.flight).stem}",
              f"departure {utc_datetime(args.departure):%d %b %Y %H:%M} UTC",
              f"ground speed {args.speed:g} m/s")
    return [flight, ifs_lines]


def download_command(args):
    """Command line that downloads the IFS data of the flight."""
    return (f"allive download {args.flight} "
            f"--departure {utc_datetime(args.departure):%Y-%m-%dT%H:%M} --speed {args.speed:g}")


def ifs_layer(args, cfg, track):
    """(IFS curtains along the track or None, IFS lines of the flight note)."""
    files = netcdf_files(cfg)
    try:
        ifs = virtual_curtains(cfg, track)
    except (OSError, KeyError, ValueError) as err:
        reason = str(err).splitlines()[0] if str(err) else type(err).__name__
        log_utc(f"IFS files {files[0]} and {files[1]} unreadable ({reason}); delete them and run "
                f"the download again: {download_command(args)}")
        return None, ("IFS files unreadable",)
    if ifs is None:
        log_utc(f"no IFS data in {files[0].parent}; download it with: {download_command(args)}")
        return None, ("no IFS data",)
    return ifs, (f"IFS run {dt.datetime.fromisoformat(cfg['base']):%d %b %Y %H} UTC",
                 f"forecast +{cfg['steps'][0]} to +{cfg['steps'][-1]} h")


def aircraft_altitude_km(path, when_s):
    """Aircraft altitude in km at a time along a track or measured fixes, or None."""
    if path is None or when_s is None:
        return None
    return float(np.interp(when_s, path["time"], path["alt"])) / 1000.0


def flight_data(args, planned, fixes):
    """(track, IFS curtains, note, times of the planned circles after the newest fix) of the planned
    flight; no track, IFS or note and no circles without one."""
    if planned is None:
        return None, None, None, np.array([])
    track = planned if fixes is None else merge_fixes(planned, fixes)
    cfg = downloaded_config(args.flight, planned) or flight_config(args.flight, planned)
    ifs, ifs_lines = ifs_layer(args, cfg, track)
    circles = circle_times(read_flight_plan(args.flight), args.departure, args.speed)
    ahead = circles if fixes is None else circles[circles > fixes["time"][-1]]
    return track, ifs, flight_note(args, ifs_lines), ahead


def output_path(args, stamp):
    """Figure path: one per flight plan and departure, else one per measured flight."""
    if args.flight:
        stem = f"{Path(args.flight).stem}_{utc_datetime(args.departure):%Y%m%d-%H%M}"
        return figure_dir() / f"allive_{stem}.png"
    return figure_dir() / f"allive_{stamp or 'nodata'}.png"


def process(payloads, args):
    """Parse, archive, build and render one set of file payloads."""
    parsed = {name: parse_profiles(payload) for name, payload in payloads.items()}
    stamp = flight_stamp(parsed)
    if is_url(args.source) and stamp:
        archive(payloads, stamp)
    planned = planned_flight(args.flight, args.departure, args.speed) if args.flight else None
    if planned is not None:
        parsed = flight_profiles(parsed, planned)
    curtains = {name: build_curtain(parsed[name], valid, args.above_aircraft)
                for name, valid in VALID_RANGE.items()}
    if not args.ifs_only:
        for name in VALID_RANGE:
            log_utc(describe(name, parsed[name], curtains[name]))
    fixes = measured_fixes(parsed)
    track, ifs, note, circles_s = flight_data(args, planned, fixes)
    layers = panel_layers(curtains, ifs, args.panels)
    latest = newest_profile(parsed, fixes)
    xlim_s = time_limits(measured_span(parsed), track, args.hours)
    zlim_km = height_limits(layers, xlim_s, args.zlim)
    has_data = any(layers.values()) or latest is not None
    cursor_s = cursor_time(args.time, xlim_s, latest, track) if has_data else None
    view = {"xlim_s": xlim_s, "zlim_km": zlim_km, "cursor_s": cursor_s,
            "aircraft_km": aircraft_altitude_km(fixes if track is None else track, cursor_s)}
    path = render(layers, view, circles_s, (note, measurement_note(latest)), output_path(args, stamp))
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
