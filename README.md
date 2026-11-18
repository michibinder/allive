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
  and the planned wind circles. `--speed` sets the ground speed (default 230 m/s). The flight plan is
  an FTML file (waypoints with flight levels, as written by the Mission Support System) or a KML file;
  from a KML the named points are read in file order, with border crossings left out. Only the profiles
  of that flight are drawn: from 3 h before the planned start to 3 h after the planned end, and on to the
  newest profile of any run of profiles (no gap over 2 h) that reaches into this window. A flight longer
  than planned is so drawn to its end; another flight in the same files stays out unless it reaches into
  this window or follows within 2 h.
- `plan` prints the MARS requests of a flight and the time their data are released. It sends nothing.
  `download` sends the requests and converts the result. Both take `--base 2026-11-13T00` to choose
  the forecast run.

Further options of `plot`:

| option | effect |
|---|---|
| `--panels t tp w u v` | rows from top to bottom (default: all five) |
| `--hours H` | only the last H hours |
| `--zlim A B` | altitude range in km (default: the range with valid measurements) |
| `--time HH:MM` or an ISO time | time of the cursor and the profiles (default: the newest profile) |
| `--ifs-only` | do not read the measurements |
| `--above-aircraft` | height grids without explicit heights start at the aircraft, not at sea level |

## The figure

- The rows T, T′ and w are curtains: a time–height panel, a narrow panel with the profiles at the
  cursor (IFS green, ALIMA red), and a colorbar. IFS is drawn first and ALIMA on top of it. T′ is the temperature after a
  5th-order Butterworth high-pass in the vertical with a cutoff of 20 km, for ALIMA and IFS alike.
  Its colour scale is logarithmic: one colour per doubling from 1 to 32 K, white below 1 K.
- In a row that shows measurements over the IFS, faded grey contour lines keep the IFS field visible:
  every 10 K for T, 1 K for T′ and 0.2 m/s for w, solid for positive and dashed for negative values.
  A line of the notes names these steps.
- The dashed cursor marks the newest measured profile, with the aircraft symbol centred on it. The
  label of the first profile panel gives its time.
- The rows u and v show wind profiles in small sub-axes at the time of their measurement. Profiles at
  most 15 min apart share one sub-axes. A sub-axes with two or more profiles has a legend with their
  times at its lower centre, in the first wind row only. With a flight plan, every planned circle
  ahead of the aircraft shows the IFS profile.
- Waypoints are marked in the first time-height panel by a tick above the time axis with a boxed label
  at the height of the aircraft symbol, and by a thin dotted line through the time-height panels of the
  rows T, T′ and w.
  - Measured waypoints are black. These are the planned waypoints the aircraft passed within 30 km, at
    the time it passed them, and every turn of the measured track: a course change of more than
    35 degrees above 7 km altitude, where turns within 10 minutes count as one.
  - Planned waypoints are grey. They are shown only ahead of the aircraft, at times estimated from the
    progress along the plan.
  - Labels: a planned waypoint carries its name, or its number counted from 0 if it has none, as in the
    Mission Support System. A name that only spells out coordinates, such as 5000N01150E, counts as
    none. A turn within 30 km of a planned waypoint is that waypoint; any other turn
    is "•". Without a flight plan the turns are numbered 1, 2, ...
  - Labels that would overlap the previous one are hidden, the ticks and lines stay.
- The corner right of the wind rows gives the take-off time and the name of the flight plan and, while
  the IFS is drawn, its run. A line wider than the corner gets a smaller font, down to 5 pt. Below the
  lines is a sketch of the flight in longitude and latitude at equal distances, without a map: the
  route of the flight plan grey, the measured track black, every waypoint mark of the flight with its
  label and colour from the time axis, and the aircraft symbol at the cursor. It shows where each label
  of the time axis lies on the route. The corner takes no room from the panels. Without wind rows the
  lines are joined into one line above the top panel, and there is no sketch.

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
- Access: an ECMWF API key in `~/.ecmwfapirc`. A key without real-time access receives a field 25 h
  after its valid time, and `allive plan` prints that time. For a key with real-time access set
  `EMBARGO_HOURS = None` in `src/allive/mars.py`; the default run is then also no newer than the
  newest run in MARS (7 h after its base time), so a flight days ahead gets today's run.
- Virtual lidar: the model-level fields are interpolated linearly in time, latitude and longitude to
  the track and then to geometric height (0–80 km in 200 m steps); `w = −ω R T / (p g)`.

## Code

| module | content |
|---|---|
| `track.py` | flight plan → track, planned circles |
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
