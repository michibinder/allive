import functools
import os
import string

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.transforms import Bbox
from scipy.optimize import isotonic_regression

from allive import style
from allive.filters import VERT_CUTOFF_KM
from allive.utc import time_clusters, utc_datetime

TEMP_CLIM = (190.0, 280.0)
TEMP_CMAP = "turbo"
PERT_LEVELS = (-32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
PERT_CBAR_TICKS = (-16.0, -4.0, -1.0, 1.0, 4.0, 16.0)
PERT_CMAP = style.wave_cmap()
PROF_LW = 1.1
CURSOR_KW = {"color": "black", "lw": 1.8, "ls": "--", "zorder": 5}
MAX_PROFILE_WIDTH_S = 300.0
FIGURE_WIDTH_IN = 13.0
PANEL_HEIGHT_IN = 3.0
CURTAIN_WIDTH_RATIO = 5.5
COLORBAR_BOUNDS = (1.06, 0.1, 0.07, 0.8)
WIND_INSET_FRACTION = 0.10
WIND_GROUP_GAP_S = 900.0
LABEL_BAND_BOTTOM = 0.84
SEE_THROUGH_ALPHA = 0.5
WIND_TICK_MULTIPLES = (1, 2, 4, 5)
WIND_MAX_TICKS = 3
DPI = 150
NO_DATA_NOTE = "no valid data yet"
SOURCE_COLORS = {"IFS": "#009E73", "ALIMA": "#E31A1C"}
WIND_PROFILE_COLORS = (SOURCE_COLORS["ALIMA"], "#0072B2", "#E69F00", "#7B3FA0", "#56B4E9", "0.3")
NOTE_STYLE = {"fontsize": 7, "color": "0.25"}
CELL_NOTE_STYLE = {**NOTE_STYLE, "fontsize": 8}
MIN_NOTE_FONTSIZE = 5.0
NOTE_FONT_STEP = 0.25
NOTE_OFFSET_PT = 18.0
AIRCRAFT_ICON = "✈"
ICON_MARGIN_FRACTION = 0.07
ICON_SIDE_INSET_PT = 10.0
ICON_KW = {"fontsize": 22, "fontname": "DejaVu Sans", "ha": "center", "va": "center", "color": "black",
           "zorder": 6, "path_effects": [patheffects.withStroke(linewidth=2.5, foreground="white")]}
CURTAIN_LABEL_X = style.XLBL / 2.0
PANEL_LETTER_PAD_PT = (12.0, style.PANEL_LABEL_PAD_PT)
PROFILE_FONTSIZE = 7
WAYPOINT_ZORDER = 4
WAYPOINT_TICK_KW = {"colors": "0.4", "linewidths": 0.5, "zorder": WAYPOINT_ZORDER}
WAYPOINT_LINE_KW = {"linewidths": 0.6, "linestyles": ":", "zorder": WAYPOINT_ZORDER}
WAYPOINT_COLORS = {True: "black", False: "0.5"}
WAYPOINT_GAP_PT = 2.0
WAYPOINT_BOX = {"boxstyle": "round,pad=0.2", "facecolor": "white", "linewidth": 0.5}
SKETCH_FONTSIZE = 6
SKETCH_LW = 0.8
SKETCH_ICON_KW = {**ICON_KW, "fontsize": 11}
SKETCH_GAP_PT = 6.0
SKETCH_PAD_PT = 1.5
WIND_LEGEND_KW = {"fontsize": PROFILE_FONTSIZE, "loc": "lower left", "handlelength": 1.0,
                  "handletextpad": 0.5, "borderpad": 0.3, "borderaxespad": 1.0, "framealpha": 1.0}
PROFILE_TICK_PAD_PT = -14.0
PROFILE_TICK_BOX = {"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none",
                    "alpha": 0.8}
ZERO_LINE_KW = {"color": "0.6", "lw": 0.6, "zorder": 1}
CONTOUR_KW = {"colors": "0.55", "linewidths": 0.6, "alpha": 0.6, "negative_linestyles": "dashed",
              "zorder": 2}
WIND_CMAP = style.vik_white_cmap()
FIELDS = {
    "t": {"file": "T.txt", "label": "temperature", "cbar": "temperature / K",
          "cmap": TEMP_CMAP, "clim": TEMP_CLIM, "step": 20.0, "primes": False, "zero_line": False,
          "curtain": True, "contours": (10.0, "K (T)")},
    "tp": {"file": "T.txt",
           "label": rf"$\mathbf{{T'}}$ ($\boldsymbol{{\lambda_z}} <$ {VERT_CUTOFF_KM:.0f} km)",
           "cbar": r"$T'$ / K", "cmap": PERT_CMAP, "clim": (PERT_LEVELS[0], PERT_LEVELS[-1]),
           "levels": PERT_LEVELS, "cbar_ticks": PERT_CBAR_TICKS, "step": 16.0,
           "primes": True, "zero_line": False, "curtain": True, "contours": (1.0, "K (T′)")},
    "w": {"file": "w.txt", "label": r"$\mathbf{w}$", "cbar": r"$w$ / m$\,$s$^{-1}$",
          "cmap": WIND_CMAP, "clim": (-4.0, 4.0), "step": 2.0, "primes": False,
          "zero_line": True, "curtain": True, "contours": (0.2, "m/s (w)")},
    "u": {"file": "u.txt", "label": r"$\mathbf{u}$ / m$\,$s$^{\mathbf{-1}}$", "clim": (-100.0, 100.0),
          "step": 50.0, "primes": False, "curtain": False},
    "v": {"file": "v.txt", "label": r"$\mathbf{v}$ / m$\,$s$^{\mathbf{-1}}$", "clim": (-100.0, 100.0),
          "step": 50.0, "primes": False, "curtain": False},
}


def cell_edges(time_s):
    """Left and right time edge of every profile cell; wide gaps take the other side's width."""
    half = MAX_PROFILE_WIDTH_S / 2.0
    gap = np.diff(time_s) / 2.0
    before = np.concatenate([[np.inf], gap])
    after = np.concatenate([gap, [np.inf]])
    left = np.where(before <= half, before, np.where(after <= half, after, half))
    right = np.where(after <= half, after, np.where(before <= half, before, half))
    return time_s - left, time_s + right


def seconds_to_num(seconds):
    """Unix seconds (UTC) as matplotlib date numbers."""
    millis = np.rint(np.asarray(seconds, float) * 1000.0).astype("int64")
    return mdates.date2num(millis.astype("datetime64[ms]"))


def colour_norm(spec):
    """Colour norm of a curtain field: the bins between its levels when it has some, else linear between
    its colour limits."""
    if "levels" in spec:
        return BoundaryNorm(spec["levels"], plt.get_cmap(spec["cmap"]).N, clip=True)
    return Normalize(*spec["clim"])


def draw_curtain(ax, curtain, spec):
    """One pcolormesh of the curtain with transparent gaps between non-contiguous profiles."""
    left, right = cell_edges(curtain["time"])
    n = left.size
    x_edges = np.empty(2 * n)
    x_edges[0::2], x_edges[1::2] = left, right
    values = np.full((curtain["z"].size, 2 * n - 1), np.nan)
    values[:, 0::2] = curtain["data"].T
    z_km = curtain["z"] / 1000.0
    half = curtain["dz"] / 2000.0
    z_edges = np.append(z_km - half, z_km[-1] + half)
    ax.pcolormesh(seconds_to_num(x_edges), z_edges, values, shading="flat", rasterized=True,
                  cmap=spec["cmap"], norm=colour_norm(spec))


def has_contours(spec, panel):
    """True for a row that shows measurements over an IFS curtain and so draws the IFS contour lines."""
    return "contours" in spec and "IFS" in panel and "ALIMA" in panel


def contour_key(layers):
    """Note lines that explain the IFS contour lines of the rows that draw them."""
    steps = ["{:g} {}".format(*FIELDS[key]["contours"]) for key, panel in layers.items()
             if has_contours(FIELDS[key], panel)]
    return [f"grey lines: IFS in steps of {', '.join(steps)}"] if steps else []


def draw_contours(ax, curtain, spec):
    """Faded grey contour lines of a curtain at the multiples of the field's contour step except zero:
    solid for positive, dashed for negative values."""
    data = np.ma.masked_invalid(curtain["data"].T)
    levels = inner_ticks(data.min(), data.max(), spec["contours"][0])
    levels = levels[levels != 0.0]
    if levels.size:
        ax.contour(seconds_to_num(curtain["time"]), curtain["z"] / 1000.0, data, levels=levels, **CONTOUR_KW)


def step_ticks(low, high, step):
    """Multiples of step from low to high, both included."""
    return np.arange(np.ceil(low / step), np.floor(high / step) + 1) * step


def profile_limits(shown, spec):
    """x limits of a profile panel: the colour limits, widened in tick steps to hold the values."""
    values = np.concatenate([np.ravel(spec["clim"]), *(v[np.isfinite(v)] for v in shown)])
    return (min(spec["clim"][0], np.floor(values.min() / spec["step"]) * spec["step"]),
            max(spec["clim"][1], np.ceil(values.max() / spec["step"]) * spec["step"]))


def inner_ticks(low, high, step):
    """Multiples of step strictly between low and high."""
    ticks = step_ticks(low, high, step)
    return ticks[(ticks > low) & (ticks < high)]


def set_profile_ticks(ax, low, high, step):
    """x limits and small tick labels at the multiples of step strictly within them."""
    ax.set_xlim(low, high)
    ax.set_xticks(inner_ticks(low, high, step))
    ax.tick_params(axis="x", labelsize=PROFILE_FONTSIZE)


def wind_tick_step(half, step):
    """Smallest multiple of the field step in WIND_TICK_MULTIPLES with at most WIND_MAX_TICKS ticks
    inside plus/minus half."""
    steps = [k * step for k in WIND_TICK_MULTIPLES]
    return next((s for s in steps if inner_ticks(-half, half, s).size <= WIND_MAX_TICKS), steps[-1])


def style_profile_ticks(ax, low, high, step):
    """Profile x ticks with boxed tick labels inside the panel."""
    set_profile_ticks(ax, low, high, step)
    ax.tick_params(axis="x", pad=PROFILE_TICK_PAD_PT)
    for label in ax.get_xticklabels():
        label.set_bbox(PROFILE_TICK_BOX)
    ax.set_axisbelow(False)


def nearest_row(curtain, time_s):
    """Index of the profile of a curtain nearest to a time."""
    return int(np.argmin(np.abs(curtain["time"] - time_s)))


def shown_values(curtain, row, zlim_km):
    """Values of one profile within the altitude range."""
    z_km = curtain["z"] / 1000.0
    return curtain["data"][row][(z_km >= zlim_km[0]) & (z_km <= zlim_km[1])]


def group_profiles(panel, time_s, rows):
    """[(source, curtain, row)] of one group: the IFS profile nearest its time, then the ALIMA ones."""
    found = [("IFS", panel["IFS"], nearest_row(panel["IFS"], time_s))] if "IFS" in panel else []
    return found + [("ALIMA", panel["ALIMA"], int(i)) for i in rows]


def wind_groups(panel, circles_s, xlim_s):
    """Groups {"time", "profiles"} of a wind row sorted by time: clusters of the measured profiles,
    then of the planned circles farther than WIND_GROUP_GAP_S from all of them."""
    measured, found = np.array([]), []
    if "ALIMA" in panel:
        curtain = panel["ALIMA"]
        seen = np.isfinite(curtain["data"]).any(axis=1)
        rows = np.flatnonzero(seen & (curtain["time"] >= xlim_s[0]) & (curtain["time"] <= xlim_s[1]))
        measured = curtain["time"][rows]
        found += [(measured[part].mean(), rows[part]) for part in time_clusters(measured, WIND_GROUP_GAP_S)]
    if "IFS" in panel:
        planned = np.sort([t for t in circles_s if xlim_s[0] <= t <= xlim_s[1]
                           and np.all(np.abs(measured - t) > WIND_GROUP_GAP_S)])
        found += [(planned[part].mean(), []) for part in time_clusters(planned, WIND_GROUP_GAP_S)]
    return sorted(({"time": t, "profiles": group_profiles(panel, t, rows)} for t, rows in found),
                  key=lambda group: group["time"])


def spread_centres(times, width, xlim_s):
    """Sub-axes centres nearest to the sorted times (least squares) inside the panel, one width apart or
    closer only when they do not fit."""
    room = xlim_s[1] - xlim_s[0] - width
    steps = np.arange(times.size) * min(width, room / max(times.size - 1, 1))
    fitted = isotonic_regression(times - steps).x
    return np.clip(fitted, xlim_s[0] + width / 2.0, xlim_s[1] - width / 2.0 - steps[-1:]) + steps


def place_groups(groups, width, xlim_s):
    """The groups of a wind row with the centre of their sub-axes."""
    centres = spread_centres(np.array([group["time"] for group in groups]), width, xlim_s)
    return [{**group, "centre": centre} for group, centre in zip(groups, centres)]


def half_range(groups, spec, zlim_km):
    """Half of the symmetric x range the profiles of one wind row need."""
    shown = [shown_values(c, i, zlim_km) for group in groups for _, c, i in group["profiles"]]
    low, high = profile_limits(shown, spec)
    return max(-low, high)


def wind_layout(layers, circles_s, view):
    """Sub-axes width in s, the half x range common to all sub-axes, the placed groups of every wind
    row and the row that carries the legends (the first)."""
    xlim_s = view["xlim_s"]
    width = WIND_INSET_FRACTION * (xlim_s[1] - xlim_s[0])
    rows = {key: place_groups(wind_groups(panel, circles_s, xlim_s), width, xlim_s)
            for key, panel in layers.items() if not FIELDS[key]["curtain"]}
    half = max((half_range(groups, FIELDS[key], view["zlim_km"]) for key, groups in rows.items()),
               default=0.0)
    return {"width_s": width, "half": half, "rows": rows, "legend_row": next(iter(rows), None)}


def profile_label(source, times_s, fmt):
    """Legend label of profiles: the source and their UTC times joined by a dash."""
    return f"{source} " + "-".join(f"{utc_datetime(t):{fmt}}" for t in times_s)


def wind_lines(profiles):
    """(curtain, row, colour, legend label or None) of the profiles of one sub-axes; more measured
    profiles than colours are all drawn in the ALIMA colour under one label."""
    lines = [(c, i, SOURCE_COLORS["IFS"], profile_label("IFS", [c["time"][i]], "%H:%M"))
             for source, c, i in profiles if source == "IFS"]
    measured = [(c, i) for source, c, i in profiles if source == "ALIMA"]
    if len(measured) <= len(WIND_PROFILE_COLORS):
        return lines + [(c, i, colour, profile_label("ALIMA", [c["time"][i]], "%H:%M:%S"))
                        for (c, i), colour in zip(measured, WIND_PROFILE_COLORS)]
    label = profile_label("ALIMA", [c["time"][i] for c, i in (measured[0], measured[-1])], "%H:%M:%S")
    return lines + [(c, i, SOURCE_COLORS["ALIMA"], None if k else label)
                    for k, (c, i) in enumerate(measured)]


def wind_inset(ax, centre_s, width_s):
    """Sub-axes of the full panel height centred on a time, sharing the altitude axis."""
    left = seconds_to_num(centre_s - width_s / 2.0)
    right = seconds_to_num(centre_s + width_s / 2.0)
    return ax.inset_axes((left, 0.0, right - left, 1.0), transform=ax.get_xaxis_transform(), sharey=ax)


def style_wind_inset(inset, half, step):
    """Symmetric x range, wind ticks and labels on top, vertical grid lines, no frame."""
    set_profile_ticks(inset, -half, half, wind_tick_step(half, step))
    inset.tick_params(axis="x", which="both", bottom=False, labelbottom=False, top=True, labeltop=True)
    inset.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    inset.xaxis.minorticks_off()
    inset.xaxis.grid(True)
    inset.patch.set_visible(False)
    for spine in inset.spines.values():
        spine.set_visible(False)


def add_wind_legend(inset):
    """Compact legend of a sub-axes, outside the layout; arrange_legends settles its place."""
    inset.legend(**WIND_LEGEND_KW).set_in_layout(False)


def legend_start(inset, edges, renderer):
    """Display x where the legend of a sub-axes starts: centred on it and kept inside the panel edges."""
    pad = renderer.points_to_pixels(WIND_LEGEND_KW["borderaxespad"] * WIND_LEGEND_KW["fontsize"])
    width = inset.get_legend().get_window_extent(renderer).width + 2.0 * pad
    return float(np.clip((inset.bbox.x0 + inset.bbox.x1 - width) / 2.0, edges[0], edges[1] - width))


def place_legend(ax, legend, x, y, renderer):
    """Anchor the lower left corner of a legend at the display point (x, y) of a panel and return the
    box it then has."""
    legend.set_bbox_to_anchor(ax.transAxes.inverted().transform((x, y)), transform=ax.transAxes)
    return legend.get_window_extent(renderer)


def arrange_legends(ax, renderer):
    """Legend of every sub-axes of a wind panel at its lower centre: above a text of the panel it would
    cover (the aircraft symbol), see-through where a profile runs under it."""
    edges = (ax.bbox.x0, ax.bbox.x1)
    texts = [text.get_window_extent(renderer) for text in ax.texts if text.get_visible()]
    for inset in ax.child_axes:
        legend = inset.get_legend()
        if legend is None:
            continue
        x = legend_start(inset, edges, renderer)
        box = place_legend(ax, legend, x, ax.bbox.y0, renderer)
        covered = [text.y1 for text in texts if text.overlaps(box)]
        if covered:
            box = place_legend(ax, legend, x, max(covered), renderer)
        if any(crosses(box, line) for line in inset.get_legend_handles_labels()[0]):
            legend.get_frame().set_alpha(SEE_THROUGH_ALPHA)


def crosses(box, line):
    """True when a segment of a line passes a display box, judged by the box around the segment."""
    xy = line.get_transform().transform(line.get_xydata())
    low, high = np.minimum(xy[:-1], xy[1:]), np.maximum(xy[:-1], xy[1:])
    return bool(np.any((low[:, 0] <= box.x1) & (high[:, 0] >= box.x0)
                       & (low[:, 1] <= box.y1) & (high[:, 1] >= box.y0)))


def see_through_labels(ax, renderer):
    """See-through boxes for the boxed labels of a wind panel that lie on a profile, which so stays
    visible."""
    lines = [line for inset in ax.child_axes for line in inset.get_legend_handles_labels()[0]]
    for text in ax.texts:
        patch = text.get_bbox_patch()
        if patch is not None and any(crosses(patch.get_window_extent(renderer), line) for line in lines):
            patch.set_facecolor((1.0, 1.0, 1.0, SEE_THROUGH_ALPHA))


def freeze_layout(fig):
    """Lay the figure out, keep that layout and return the renderer."""
    fig.draw_without_rendering()
    fig.set_layout_engine("none")
    return fig.canvas.get_renderer()


def arrange_waypoint_labels(labels, renderer):
    """Move every waypoint label that sticks out of its panel inside it, then hide those within
    WAYPOINT_GAP_PT of the previous visible one."""
    gap = renderer.points_to_pixels(WAYPOINT_GAP_PT)
    previous = None
    for text in labels:
        frame, limits = text.axes.bbox, text.axes.get_xlim()
        box = text.get_bbox_patch().get_window_extent(renderer)
        shift = max(frame.x0 + gap - box.x0, 0.0) + min(frame.x1 - gap - box.x1, 0.0)
        text.set_x(text.get_position()[0] + shift * (limits[1] - limits[0]) / frame.width)
        box = box.translated(shift, 0.0)
        if previous is not None and box.overlaps(previous.padded(gap)):
            text.set_visible(False)
        else:
            previous = box


def finish_wind_panels(axes, renderer):
    """Place the legends of the wind panels and make their labels see-through where needed."""
    for ax in axes:
        arrange_legends(ax, renderer)
        see_through_labels(ax, renderer)


def draw_wind_row(ax, groups, wind, spec, legends):
    """One sub-axes per group of a wind row on the common x range, with a legend where asked for and the
    group has several profiles; altitude grid only."""
    for group in groups:
        inset = wind_inset(ax, group["centre"], wind["width_s"])
        for curtain, i, colour, label in wind_lines(group["profiles"]):
            inset.plot(curtain["data"][i], curtain["z"] / 1000.0, color=colour, lw=PROF_LW, label=label)
        inset.axvline(0.0, **ZERO_LINE_KW)
        style_wind_inset(inset, wind["half"], spec["step"])
        if legends and len(group["profiles"]) > 1:
            add_wind_legend(inset)
    if not groups:
        empty_panel(ax, NO_DATA_NOTE)
    ax.tick_params(axis="x", which="both", top=False)
    ax.grid(True, axis="y")


def draw_profiles(ax, layers, spec, cursor_s, zlim_km):
    """Profile of every layer nearest to the cursor; x range and ticks follow the colour scale."""
    shown = []
    for source, curtain in layers.items():
        i = nearest_row(curtain, cursor_s)
        ax.plot(curtain["data"][i], curtain["z"] / 1000.0, color=SOURCE_COLORS[source], lw=PROF_LW)
        shown.append(shown_values(curtain, i, zlim_km))
    if spec["zero_line"]:
        ax.axvline(0.0, **ZERO_LINE_KW)
    style_profile_ticks(ax, *profile_limits(shown, spec), spec["step"])


def draw_curtain_row(ax, profile_ax, panel, spec, view):
    """Curtain layers, IFS contour lines over measurements, colorbar and cursor profiles of one quantity."""
    for curtain in panel.values():
        draw_curtain(ax, curtain, spec)
    if has_contours(spec, panel):
        draw_contours(ax, panel["IFS"], spec)
    if not panel:
        empty_panel(ax, NO_DATA_NOTE)
    add_colorbar(profile_ax.inset_axes(COLORBAR_BOUNDS), spec)
    draw_profiles(profile_ax, panel, spec, view["cursor_s"], view["zlim_km"])
    profile_ax.tick_params(labelleft=False)
    profile_ax.grid(True)
    ax.grid(True)


def icon_height_km(aircraft_km, zlim_km):
    """Height of the aircraft icon: the aircraft altitude kept a margin above the panel bottom and below
    the label band, the lower end of that range when unknown."""
    span = zlim_km[1] - zlim_km[0]
    low = zlim_km[0] + ICON_MARGIN_FRACTION * span
    high = zlim_km[0] + (LABEL_BAND_BOTTOM - ICON_MARGIN_FRACTION) * span
    return low if aircraft_km is None else float(np.clip(aircraft_km, low, high))


def icon_position(ax, xy, renderer):
    """Display point of the icon: the data point xy, moved ICON_SIDE_INSET_PT inside the side spines."""
    inset = renderer.points_to_pixels(ICON_SIDE_INSET_PT)
    x, y = ax.transData.transform(xy)
    return Bbox.from_bounds(np.clip(x, ax.bbox.x0 + inset, ax.bbox.x1 - inset), y, 0.0, 0.0)


def draw_cursor(ax, view):
    """Dashed cursor line with the aircraft icon on it; nothing without a cursor in the time window."""
    if view["cursor_s"] is None or not view["xlim_s"][0] <= view["cursor_s"] <= view["xlim_s"][1]:
        return
    x = seconds_to_num(view["cursor_s"])
    ax.axvline(x, **CURSOR_KW)
    xy = (x, icon_height_km(view["aircraft_km"], view["zlim_km"]))
    icon = ax.annotate(AIRCRAFT_ICON, (0.0, 0.0), xycoords=functools.partial(icon_position, ax, xy),
                       **ICON_KW)
    icon.set_in_layout(False)


def waypoint_order(waypoints, view):
    """Indices of the waypoints inside the time window, in time order."""
    low, high = view["xlim_s"]
    return [k for k in np.argsort(waypoints["time"]) if low <= waypoints["time"][k] <= high]


def draw_waypoint_lines(ax, waypoints, view):
    """Thin dotted line over the whole panel height at every waypoint inside the time window."""
    order = waypoint_order(waypoints, view)
    lines = ax.vlines(seconds_to_num(waypoints["time"][order]), 0.0, 1.0, transform=ax.get_xaxis_transform(),
                      colors=[WAYPOINT_COLORS[bool(waypoints["measured"][k])] for k in order],
                      **WAYPOINT_LINE_KW)
    lines.set_in_layout(False)


def waypoint_label(ax, xy, label, measured, fontsize, **kwargs):
    """Boxed waypoint label centred on xy, black for a measured and grey for a planned waypoint, outside
    the layout."""
    colour = WAYPOINT_COLORS[bool(measured)]
    text = ax.text(*xy, label, ha="center", va="center", fontsize=fontsize, color=colour,
                   zorder=WAYPOINT_ZORDER + 0.1, bbox={**WAYPOINT_BOX, "edgecolor": colour}, **kwargs)
    text.set_in_layout(False)
    return text


def draw_waypoints(ax, waypoints, view):
    """Ticks and labels of the waypoints inside the time window just above the time axis; the label
    texts in time order."""
    order = waypoint_order(waypoints, view)
    transform = ax.get_xaxis_transform()
    ticks = ax.vlines(seconds_to_num(waypoints["time"][order]), 0.0, ICON_MARGIN_FRACTION,
                      transform=transform, **WAYPOINT_TICK_KW)
    ticks.set_in_layout(False)
    return [waypoint_label(ax, (seconds_to_num(waypoints["time"][k]), ICON_MARGIN_FRACTION),
                           waypoints["label"][k], waypoints["measured"][k], PROFILE_FONTSIZE,
                           transform=transform)
            for k in order]


def add_colorbar(cax, spec):
    """Vertical colorbar of a curtain row in the axes beside its profile panel, also without data."""
    cb = cax.figure.colorbar(ScalarMappable(colour_norm(spec), spec["cmap"]), cax=cax,
                             extend="both")
    cb.set_label(spec["cbar"])
    cb.set_ticks(spec.get("cbar_ticks", step_ticks(*spec["clim"], spec["step"])))


def date_label(start_s, end_s):
    """x label of the time axis with the date(s) of the shown window."""
    a, b = utc_datetime(start_s), utc_datetime(end_s)
    if a.date() == b.date():
        return f"time (UTC), {b:%d %b %Y}"
    if (a.year, a.month) == (b.year, b.month):
        return f"time (UTC), {a:%d}-{b:%d %b %Y}"
    return f"time (UTC), {a:%d %b}-{b:%d %b %Y}"


def corner_axes(fig, cells):
    """Empty axes with the axis off in the free cells; laid out, their boxes make up the corner."""
    axes = [fig.add_subplot(cell) for cell in cells]
    for ax in axes:
        ax.axis("off")
    return axes


def add_lines(fig, box, lines, renderer):
    """Grey lines of one line each from the top left of a figure box down, the font of a line wider than the
    box shrunk in NOTE_FONT_STEP steps until it fits, not below MIN_NOTE_FONTSIZE; the figure y below."""
    width = box.width * fig.bbox.width
    y = box.y1
    for line in lines:
        text = fig.text(box.x0, y, line, ha="left", va="top", **CELL_NOTE_STYLE)
        while text.get_window_extent(renderer).width > width and text.get_fontsize() > MIN_NOTE_FONTSIZE:
            text.set_fontsize(max(MIN_NOTE_FONTSIZE, text.get_fontsize() - NOTE_FONT_STEP))
        y = text.get_window_extent(renderer).y0 / fig.bbox.height
    return y


def text_box(text, renderer):
    """Display box of a text, with its box patch when it has one."""
    patch = text.get_bbox_patch()
    if patch is None:
        return text.get_window_extent(renderer)
    text.update_bbox_position_size(renderer)
    return patch.get_window_extent(renderer)


def draw_sketch(fig, box, view, renderer):
    """Planned route (grey), measured track (black), all waypoint marks and the aircraft at the cursor in
    longitude and latitude at equal distances, at the top of a figure box and inset so no text sticks out."""
    paths = [(path, measured) for path, measured in ((view["plan"], False), (view["fixes"], True))
             if path is not None]
    if not paths:
        return
    ax = fig.add_axes(box.bounds)
    ax.axis("off")
    for path, measured in paths:
        ax.plot(path["lon"], path["lat"], color=WAYPOINT_COLORS[measured], lw=SKETCH_LW)
    marks, texts = view["waypoints"], []
    for lon, lat, label, measured in zip(marks["lon"], marks["lat"], marks["label"], marks["measured"]):
        texts.append(waypoint_label(ax, (lon, lat), label, measured, SKETCH_FONTSIZE))
    if view["position"] is not None:
        texts.append(ax.text(view["position"]["lon"], view["position"]["lat"], AIRCRAFT_ICON,
                             **SKETCH_ICON_KW))
    ax.margins(0.0)
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(ax.get_ylim()))), adjustable="box", anchor="N")
    sizes = [text_box(text, renderer).size for text in texts] or [(0.0, 0.0)]
    half = (np.max(sizes, axis=0) / 2.0 + renderer.points_to_pixels(SKETCH_PAD_PT)) / fig.bbox.size
    ax.set_position(box.padded(-half[0], -half[1]))


def fill_corner(fig, axes, notes, view, renderer):
    """The notes as lines at the top left of the corner, the first free cells in adjacent grid rows, and the
    track sketch below them."""
    rows = np.array([ax.get_subplotspec().rowspan.start for ax in axes])
    box = Bbox.union([axes[k].get_position() for k in time_clusters(rows, 1)[0]])
    bottom = add_lines(fig, box, notes, renderer)
    gap = renderer.points_to_pixels(SKETCH_GAP_PT) / fig.bbox.height if notes else 0.0
    draw_sketch(fig, Bbox.from_extents(box.x0, box.y0, box.x1, bottom - gap), view, renderer)


def add_note(ax, text):
    """Small note just above the panel, aligned with its left edge."""
    ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, NOTE_OFFSET_PT),
                textcoords="offset points", ha="left", va="bottom", **NOTE_STYLE)


def row_axes(fig, cells, top):
    """Axes in the grid cells of one row: the first shares time and altitude with top, the others
    its altitude."""
    ax = fig.add_subplot(cells[0], sharex=top, sharey=top)
    return (ax, *(fig.add_subplot(cell, sharey=ax) for cell in cells[1:]))


def draw_row(panels, key, panel, view, wind):
    """A curtain row with its profile panel or a wind row, with cursor and row label."""
    spec = FIELDS[key]
    if spec["curtain"]:
        draw_curtain_row(*panels, panel, spec, view)
    else:
        draw_wind_row(panels[0], wind["rows"][key], wind, spec, key == wind["legend_row"])
    draw_cursor(panels[0], view)
    style.add_corner_label(panels[0], spec["label"], x=CURTAIN_LABEL_X)
    panels[0].set_ylabel("altitude / km")


def draw_rows(fig, layers, view, wind):
    """All rows from top to bottom; (top time-height axes, [(time-height axes, free right-column cell)]
    of the wind rows, waypoint label texts of the first row)."""
    grid = fig.add_gridspec(len(layers), 2, width_ratios=(CURTAIN_WIDTH_RATIO, 1.0))
    letters = iter(string.ascii_lowercase)
    first_curtain = next((key for key in layers if FIELDS[key]["curtain"]), None)
    top, wind_rows, waypoint_texts = None, [], []
    for row, (key, panel) in enumerate(layers.items()):
        curtain = FIELDS[key]["curtain"]
        panels = row_axes(fig, (grid[row, 0], grid[row, 1]) if curtain else (grid[row, 0],), top)
        top = panels[0] if top is None else top
        draw_row(panels, key, panel, view, wind)
        if curtain and view["waypoints"]["time"].size:
            draw_waypoint_lines(panels[0], view["waypoints"], view)
        if row == 0 and view["waypoints"]["time"].size:
            waypoint_texts = draw_waypoints(panels[0], view["waypoints"], view)
        if not curtain:
            wind_rows.append((panels[0], grid[row, 1]))
        marks = [style.add_panel_label(ax, next(letters), PANEL_LETTER_PAD_PT) for ax in panels]
        if key == first_curtain and view["cursor_s"] is not None:
            style.add_mirrored_label(panels[1], f"{utc_datetime(view['cursor_s']):%H:%M} UTC", marks[1])
        panels[0].tick_params(labelbottom=row == len(layers) - 1, labeltop=row == 0 and curtain)
    panels[0].set_xlabel(date_label(*view["xlim_s"]))
    return top, wind_rows, waypoint_texts


def render(layers, view, circles_s, notes, out):
    """Draw one row per quantity, the notes and the track sketch and save the figure atomically to out."""
    with plt.style.context(style.STYLE_SHEET):
        fig = plt.figure(layout="constrained", figsize=(FIGURE_WIDTH_IN, PANEL_HEIGHT_IN * len(layers)),
                         dpi=DPI)
        tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp.png")
        try:
            wind = wind_layout(layers, circles_s, view)
            top, wind_rows, waypoint_texts = draw_rows(fig, layers, view, wind)
            top.set_xlim(seconds_to_num(view["xlim_s"][0]), seconds_to_num(view["xlim_s"][1]))
            top.set_ylim(*view["zlim_km"])
            top.xaxis_date()
            top.xaxis.set_major_locator(mdates.AutoDateLocator())
            top.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            cells = [cell for _, cell in wind_rows] if notes or view["cursor_s"] is not None else []
            corner = corner_axes(fig, cells)
            if notes and not corner:
                add_note(top, " | ".join(notes))
            if wind_rows or waypoint_texts:
                renderer = freeze_layout(fig)
                arrange_waypoint_labels(waypoint_texts, renderer)
                finish_wind_panels([ax for ax, _ in wind_rows], renderer)
                if corner:
                    fill_corner(fig, corner, notes, view, renderer)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(tmp, dpi=DPI, facecolor="w")
            os.replace(tmp, out)
        finally:
            plt.close(fig)
            tmp.unlink(missing_ok=True)
        return out


def empty_panel(ax, text):
    """Grey centred note in an axes without data."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="grey")
