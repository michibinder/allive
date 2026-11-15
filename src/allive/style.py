import functools
from pathlib import Path

import numpy as np
from cmcrameri import cm as crameri_cm
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox

STYLE_SHEET = Path(__file__).with_name("allive.mplstyle")
PANEL_LABEL_PAD_PT = 8.0
XLBL = 0.04
YPP = 0.93
LABEL_BOX_CIRCLE = {"boxstyle": "circle", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}
LABEL_BOX_PAD = 0.3
LABEL_BOX_ROUND = {"boxstyle": f"round,pad={LABEL_BOX_PAD}", "lw": 0.67, "facecolor": "white",
                   "edgecolor": "black"}
VIK_WHITE_WIDTH = 0.14
VIK_WHITE_PLATEAU = 0.04
VIK_WHITE_N = 256


def add_panel_label(ax, label, pad_pt=PANEL_LABEL_PAD_PT):
    """Circled panel letter pad_pt points (one value, or x and y) in from the top-right corner of
    an axes; returns the letter."""
    pad_x, pad_y = np.broadcast_to(pad_pt, 2)
    return ax.annotate(label, xy=(1.0, 1.0), xycoords="axes fraction", xytext=(-pad_x, -pad_y),
                       textcoords="offset points", ha="right", va="top", fontsize=12, fontweight="bold",
                       bbox=LABEL_BOX_CIRCLE, zorder=6)


def add_corner_label(ax, text, x=XLBL, y=YPP, weight="bold"):
    """Round-box descriptive label anchored top-left at axes fraction (x, y) of an axes."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            weight=weight, bbox=LABEL_BOX_ROUND, zorder=6)


def mirrored_corner(letter, renderer):
    """Display point at the height of a panel letter, as far inside the left spine of its axes as the
    circle of the letter is inside the right one."""
    letter.update_positions(renderer)
    letter.update_bbox_position_size(renderer)
    circle = letter.get_bbox_patch().get_window_extent(renderer)
    frame = letter.axes.bbox
    return Bbox.from_bounds(frame.x0 + frame.x1 - circle.x1, (circle.y0 + circle.y1) / 2.0, 0.0, 0.0)


def add_mirrored_label(ax, text, letter):
    """Round-box label at the top left of an axes that mirrors its panel letter: same height, same
    distance from the spine."""
    ax.annotate(text, (0.0, 0.0), xycoords=functools.partial(mirrored_corner, letter),
                xytext=(LABEL_BOX_PAD * rcParams["font.size"], 0.0), textcoords="offset points",
                ha="left", va="center", weight="bold", bbox=LABEL_BOX_ROUND, zorder=6)


def wave_cmap():
    """Diverging wave colormap with smooth transitions and a white centre."""
    colors = [(0.0, "darkslateblue"), (0.1, "royalblue"), (0.2, "cornflowerblue"), (0.4, "lavender"),
              (0.48, "white"), (0.52, "white"), (0.6, "palegoldenrod"), (0.75, "goldenrod"),
              (0.85, "indianred"), (1.0, "darkred")]
    return LinearSegmentedColormap.from_list("wave", colors=colors, N=256)


def vik_white_cmap():
    """Crameri's 'vik' with a truly white centre: a white plateau, smoothstep blended back into vik."""
    x = np.linspace(0, 1, VIK_WHITE_N)
    colors = crameri_cm.vik(x)

    distance = np.abs(x - 0.5)
    t = np.clip((distance - VIK_WHITE_PLATEAU / 2) / (VIK_WHITE_WIDTH - VIK_WHITE_PLATEAU / 2), 0.0, 1.0)
    white_weight = 1.0 - t * t * (3.0 - 2.0 * t)
    colors[:, :3] = colors[:, :3] * (1.0 - white_weight[:, None]) + white_weight[:, None]

    return LinearSegmentedColormap.from_list("vik_white", colors, N=VIK_WHITE_N)
