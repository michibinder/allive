import argparse
import datetime as dt
from functools import partial
from xml.etree.ElementTree import ParseError

from allive import ifs, mars, quicklook, track
from allive.figure import FIELDS
from allive.stream import SOURCE_URL
from allive.utc import iso_seconds

MAX_CLOCK_CHARS = 5


def run_base(text):
    """Forecast run base (00 or 12 UTC) from an ISO string, for argparse."""
    try:
        base = dt.datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO time: {text}")
    if base.hour % mars.BASE_CYCLE_HOURS or base.minute or base.second or base.tzinfo:
        raise argparse.ArgumentTypeError(f"{text} is not a naive 00 or 12 UTC run time")
    return base


def readable_plan(text):
    """Path of a flight plan that can be read, for argparse."""
    try:
        track.read_flight_plan(text)
    except (OSError, ValueError, KeyError, ParseError) as error:
        raise argparse.ArgumentTypeError(f"{type(error).__name__}: {error}")
    return text


def clock_or_iso(text):
    """HH or HH:MM as a time of day, anything else as an ISO datetime, for argparse."""
    try:
        if text.replace(":", "").isdigit() and len(text) <= MAX_CLOCK_CHARS:
            hour, _, minute = text.partition(":")
            return dt.time(int(hour), int(minute or 0))
        return dt.datetime.fromisoformat(text.upper())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{error}; use HH, HH:MM or an ISO time")


def add_flight_arguments(parser, required):
    """The arguments that define a flight: departure time and ground speed."""
    hint = "" if required else " (required with --flight)"
    parser.add_argument("--departure", type=iso_seconds, metavar="ISO", required=required,
                        help=f"departure time, UTC, e.g. 2026-11-10T05:30{hint}")
    parser.add_argument("--speed", type=float, default=track.DEFAULT_SPEED_MS, metavar="MS",
                        help="ground speed of the planned track (default: %(default)s m/s)")


def add_plot_parser(commands):
    """The plot subcommand: the ALIMA quicklook."""
    p = commands.add_parser("plot", description="Real-time quicklook of the ALIMA lidar profiles.")
    p.add_argument("source", nargs="?", default=SOURCE_URL,
                   help="base URL (http/https) or local directory with T.txt, w.txt, u.txt, v.txt")
    p.add_argument("--watch", nargs="?", type=float, const=quicklook.DEFAULT_WATCH_S, default=None,
                   metavar="SECONDS", help="keep running, re-fetch every SECONDS (60 if omitted) "
                   "and redraw when any file changed")
    p.add_argument("--hours", type=float,
                   help="show only the last H hours of measurements (of the track when there are none)")
    p.add_argument("--zlim", nargs=2, type=float, metavar=("ZMIN_KM", "ZMAX_KM"),
                   help="fixed altitude limits in km (default: automatic)")
    p.add_argument("--panels", nargs="+", choices=list(FIELDS), default=quicklook.DEFAULT_PANELS,
                   help="panel rows from top to bottom (t, tp, w, u, v)")
    p.add_argument("--above-aircraft", action="store_true",
                   help="height grid starts at each profile's aircraft altitude, not at sea level")
    p.add_argument("--flight", type=readable_plan, metavar="PLAN",
                   help="flight plan (.ftml) for the IFS virtual curtains")
    add_flight_arguments(p, required=False)
    p.add_argument("--time", type=clock_or_iso, metavar="WHEN", help="cursor and profile time: HH, "
                   "HH:MM or ISO UTC (default: newest measured profile, else the end of the flight)")
    p.add_argument("--ifs-only", action="store_true", help="do not read the measurements")
    p.set_defaults(handler=partial(run_plot, p))


def add_ifs_parser(commands, name, description):
    """A subcommand that plans or runs the IFS download of a flight plan."""
    p = commands.add_parser(name, description=description)
    p.add_argument("flight_plan", type=readable_plan, metavar="PLAN", help="flight plan file (.ftml)")
    add_flight_arguments(p, required=True)
    p.add_argument("--base", type=run_base, metavar="ISO",
                   help="forecast run to use, 00 or 12 UTC, e.g. 2026-11-13T00 (default: newest usable)")
    p.set_defaults(handler=run_ifs)


def run_plot(parser, args):
    """Check the combination of options of the plot subcommand and run the quicklook."""
    if args.flight and args.departure is None:
        parser.error("--departure is required with --flight")
    if args.ifs_only and args.watch is not None:
        parser.error("--ifs-only draws once and cannot be combined with --watch")
    quicklook.watch(args)


def run_ifs(args):
    """Build the planned track and the download config and run the plan or download stage."""
    planned = track.planned_flight(args.flight_plan, args.departure, args.speed)
    try:
        cfg = ifs.flight_config(args.flight_plan, planned, args.base)
    except ValueError as error:
        raise SystemExit(str(error))
    notice, released = ifs.release_notice(cfg)
    if args.command == "plan":
        mars.plan(cfg)
        if not released:
            print(notice)
    elif released:
        mars.download(cfg)
    else:
        raise SystemExit(notice)


def main(argv=None):
    """Command line entry point: plot, plan or download."""
    parser = argparse.ArgumentParser(prog="allive",
                                     description="ALIMA real-time quicklook with IFS curtains.")
    commands = parser.add_subparsers(dest="command", required=True)
    add_plot_parser(commands)
    add_ifs_parser(commands, "plan", "Print the MARS requests of the IFS download for a flight plan.")
    add_ifs_parser(commands, "download", "Run the IFS download for a flight plan.")
    args = parser.parse_args(argv)
    args.handler(args)
