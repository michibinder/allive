# allive

Real-time quicklook of the ALIMA lidar on HALO (alima-live).

allive reads the profiles the instrument sends to the ground during a flight and draws them as
time–height curtains (temperature, T′, vertical wind) and as wind profiles (zonal, meridional).
Given the flight plan, it adds the IFS forecast along the planned track as a virtual lidar, so the
forecast is on the figure ahead of the aircraft.

## Install

```bash
git clone <url> allive
cd allive
python -m venv .venv && source .venv/bin/activate
pip install -e .               # the quicklook
pip install -e ".[download]"   # plus the IFS download (ecmwf-api-client, eccodes, cfgrib)
```

Python 3.10 or newer. Developed and tested with Python 3.13, numpy 2.4, scipy 1.18, matplotlib 3.11
and xarray 2026.7.

## Data directory

allive keeps everything except the flight plans in one directory:

| folder | content |
|---|---|
| `figures/` | the quicklooks: `allive_<plan>_<departure>.png`, without a flight plan `allive_<first profile>.png` |
| `alima/<first profile>/` | byte-exact copy of the live files of each flight |
| `ifs/` | IFS downloads: `ifs_<plan>_<N_W_S_E>_s<steps>_<run>-ml.nc` and `-ml-p.nc`, the GRIB files, request logs |

The directory is `$ALLIVE_DATA` if that is set. Otherwise it is `data/` in this repository (for an
install with `pip install -e`), and `~/allive-data` for any other install. To use IFS data that was
downloaded on another machine, copy its two `.nc` files into `ifs/`.

## Use

```bash
allive plot --watch
allive plot --watch --flight plans/flight.ftml --departure 2026-11-15T15:30
allive plot data/alima/20261111-1846
allive plot --flight plans/flight.ftml --departure 2026-11-15T15:30 --ifs-only
allive plan plans/flight.ftml --departure 2026-11-15T15:30
allive download plans/flight.ftml --departure 2026-11-15T15:30
```

- `plot --watch` fetches the live files every 60 s (`--watch SECONDS` for another interval) and
  redraws when one of them has changed. The figure file is replaced in one step, so an image viewer
  can stay open on it. Every fetch is mirrored to `alima/`; the server empties its files between
  flights, so this mirror is the only copy.
- `plot DIRECTORY` draws from a directory with `T.txt`, `w.txt`, `u.txt` and `v.txt`, for example an
  archived flight.
- `--flight` with `--departure` (UTC) adds the planned track: the IFS curtains, the aircraft altitude
  and the IFS wind profiles at the planned waypoints. `--speed` sets the ground speed (default 230 m/s).
  Between the first and the newest profile the track is the measured one. Before and after that, the
  route is flown at this speed: to the first and from the last profile taken at cruise altitude (above
  7 km) within 30 km of the route, and from the departure time while there is no such profile, for
  example on the ground. A delay therefore shifts the IFS curtain ahead, the planned waypoints and the
  end of the time axis. The IFS download keeps to the departure time (from the planned take-off to one
  hour after the planned landing), so its file names do not change during the flight, and a track that
  runs past these hours has no IFS there. The flight plan is an FTML file (waypoints with flight
  levels, as written by the Mission Support System) or a KML file; from a KML the named points are read
  in file order, with border crossings left out. Only the profiles of that flight are drawn: from 3 h
  before the planned start to 3 h after the planned end, and on to the newest profile of any run of
  profiles (no gap over 2 h) that reaches into this window. A flight longer than planned is so drawn to
  its end; another flight in the same files stays out unless it reaches into this window or follows
  within 2 h.
- `plan` prints the MARS requests of a flight and the time their data are released. It sends nothing.
  `download` sends the requests and converts the result. Both take `--base 2026-11-13T00` to choose
  the forecast run.

Further options of `plot`:

| option | effect |
|---|---|
| `--panels t tp w u v` | rows from top to bottom (default: all five) |
| `--hours H` | only the last H hours |
| `--zlim A B` | altitude range in km (default: 20 85) |
| `--time HH:MM` or ISO | cursor and profile time (default: newest profile, else take-off + 30 min) |
| `--ifs-only` | do not read the measurements |
| `--above-aircraft` | height grids without explicit heights start at the aircraft, not at sea level |

## The figure

- The time axis starts 30 min before the first profile with valid values (without one: at the start of
  the track) and ends at the end of the track or at the newest such profile, whichever is later. All
  rows show 20–85 km altitude unless `--zlim` is given.
- The rows T, T′ and w are curtains: a time–height panel, a narrow panel with the profiles at the
  cursor (IFS green, ALIMA red), and a colorbar. IFS is drawn first and ALIMA on top of it. T′ is the temperature after a
  5th-order Butterworth high-pass in the vertical with a cutoff of 15 km, for ALIMA and IFS alike.
  Its colour scale is logarithmic: one colour per doubling from 1 to 32 K, white below 1 K.
- The colour scales are fixed, and the x ranges of the profile panels equal them: T 180–300 K, T′ ±32 K,
  w ±4 m/s. Figures of different flights therefore compare directly, and the axes do not change when
  the measurements come in. A profile that leaves the range runs off the panel edge, and one wholly
  beyond it does not show in the profile panel (as often for the measured w). The T range is the
  0.1–99.9 % range of the ALIMA temperatures of 18 Nov 2026 (184–292 K), rounded outward to 10 K.
- The T profile panel has two thin grey dashed lines of the dry-adiabatic lapse rate, −9.8 K/km, 10 km
  apart over its full width: from 60 and 50 km at its right edge (300 K) up to 72.2 and 62.2 km at its
  left edge (180 K). The upper line has a label that runs along it, above its lower-right end, or below
  its upper-left end when fewer profiles cross it there. The label lies under the profiles.
- Every row with an IFS curtain has faded grey contour lines of the IFS field, with or without
  measurements, so the IFS stays visible under the measurements and an IFS-only figure looks the same:
  every 10 K for T, 1 K for T′ and 0.2 m/s for w, solid for positive and dashed for negative values.
  A line of the notes names these steps. Left of the cursor, the times of the profiles with valid values
  show only the measurement colours under the contour lines: there the IFS colours are left out over the
  full height, elsewhere they stay.
- The dashed cursor marks the newest measured profile, with the aircraft symbol centred on it. Without
  measurements it sits 30 min after take-off, about where the first profile is expected, so an IFS-only
  figure reads as a forecast. The label of the first profile panel gives its time.
- The rows u and v show wind profiles in small sub-axes at the time of their measurement. Profiles at
  most 15 min apart share one sub-axes. A sub-axes with two or more profiles has a legend with their
  times at its lower centre, in the first wind row only. With IFS data, every sub-axes of measured
  profiles also shows the IFS profile, and every planned waypoint still ahead of the aircraft (in an
  IFS-only figure: every waypoint of the plan) gets a sub-axes with the IFS profile at its time, unless
  that sub-axes would stick out of the panel, the waypoint lies within 15 min of a measured profile, or
  it lies within one sub-axes width (a tenth of the time axis) of the previous waypoint that got one. The
  next waypoint ahead therefore keeps its sub-axes. Where sub-axes overlap, the layout moves them apart,
  so they can sit off their times (by up to about 13 min with the wind timing of the past flights); the
  legends give the measured times.
- Waypoints are marked in the first time-height panel by a tick above the time axis with a boxed label
  at the height of the aircraft symbol, and by a thin dotted line through the time-height panels of the
  rows T, T′ and w.
  - Measured waypoints are black. These are the planned waypoints the aircraft passed within 30 km, at
    the time it passed them, and every turn of the measured track: a course change of more than
    35 degrees above 7 km altitude, where turns within 10 minutes count as one.
  - Planned waypoints are grey. They are shown only ahead of the aircraft, at times estimated from the
    progress along the plan.
  - Labels: a planned waypoint carries its number in the flight plan, counted from 0 as in the
    Mission Support System, never its name. A turn within 30 km of a planned waypoint is that
    waypoint; any other turn is "•". Without a flight plan the turns are numbered 1, 2, ...
  - Labels that would overlap the previous one are hidden, the ticks and lines stay.
- The corner right of the wind rows gives the take-off time and the name of the flight plan and, while
  the IFS is drawn, its run. A line wider than the corner gets a smaller font, down to 5 pt. Below the
  lines is a sketch of the flight in longitude and latitude at equal distances, without a map: the
  route of the flight plan grey, the measured track black, every waypoint mark of the flight with its
  label and colour from the time axis, and the aircraft symbol at the cursor. As on the time axis, a
  label that would overlap one shown before it is left out, so not every label of the time axis appears
  in the sketch. The corner takes no room from the panels. Without wind rows the lines are joined into
  one line above the top panel, and there is no sketch.

## Live files

Format as observed on 11 November 2026 (server address: `SOURCE_URL` in `stream.py`):

- One profile per line: `unix_time latitude longitude aircraft_altitude_m stride values…`.
- `T.txt` has values only, on the grid `z_k = k · stride · 100 m`.
- `w.txt`, `u.txt` and `v.txt` have three equal blocks after the five leading numbers: heights in km,
  values, uncertainties.
- `u.txt` and `v.txt` carry no position (latitude, longitude and altitude are 0).
- Fill values are 0.00 and −999.00. Lines can repeat; the line written last for a time stamp counts.

## IFS forecast

- Request for a flight: the box around the track (0.5° margin, rounded outward to whole degrees) on a
  0.1° grid; temperature, wind and omega on the 137 model levels; hourly steps from take-off to one
  hour after landing (3-hourly beyond +90 h, 6-hourly beyond +144 h).
- Run: the newest 00 or 12 UTC run at or before take-off. The quicklook uses the newest downloaded
  run that covers the flight.
- Access: an ECMWF API key in `~/.ecmwfapirc`. The DLR key has had real-time access since
  19 Nov 2026, so `EMBARGO_HOURS = None` in `src/allive/mars.py`: the default run is no newer than
  the newest run in MARS (7 h after its base time), so a flight days ahead gets today's run. A key
  without real-time access receives a field 25 h after its valid time; for such a key set
  `EMBARGO_HOURS = 25`, and `allive plan` prints when the fields are released.
- Virtual lidar: the model-level fields are interpolated linearly in time, latitude and longitude to
  the track and then to geometric height (0–80 km in 200 m steps); `w = −ω R T / (p g)`.

## Code

| module | content |
|---|---|
| `track.py` | flight plan → track, progress along the route, waypoint marks |
| `mars.py` | MARS requests of one forecast run, download, GRIB → netCDF |
| `levels.py` | model levels → pressure and geometric height |
| `ifs.py` | request for a flight, IFS curtains along the track |
| `stream.py` | the live files: fetch, parse, archive, curtains |
| `filters.py` | T′ |
| `figure.py`, `style.py` | the figure |
| `quicklook.py` | from the files to the figure, watch loop |
| `cli.py` | command line |

Tests: `pip install -e ".[test]"`, then `pytest`.

## Origin of the code

`levels.py` is adapted from `era5_processor.py` of ma-lidar-visualizations (DLR, LGPL-3.0-or-later).
The colormaps, the label helpers, the Butterworth filter and the MARS download are copies from mapy
by the same author, so that allive runs without it.
