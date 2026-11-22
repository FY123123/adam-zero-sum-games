"""Shared figure style for all experiments of the lecture notes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

BLUE, RED, ORANGE, AQUA = "#2a78d6", "#d9483b", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#0f4c9c", BLUE, "#f0efec", RED, "#9a1f17"])
SEQUENTIAL = LinearSegmentedColormap.from_list("grey_blue", ["#f0efec", BLUE, "#0f4c9c"])

plt.rcParams.update({
    "font.family": "cmr10", "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
    "legend.frameon": False,
})


def heatmap(ax, data, label, limit=None, signed=True, colorbar=True):
    """Grid of cells with white gaps. Signed data use a diverging map centred at zero."""
    data = np.asarray(data)
    rows, cols = data.shape
    if signed:
        limit = limit or np.abs(data).max()
        kwargs = dict(cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit))
    else:
        kwargs = dict(cmap=SEQUENTIAL, vmin=0.0, vmax=limit or data.max())
    image = ax.pcolormesh(np.arange(cols + 1) - 0.5, np.arange(rows + 1) - 0.5, data,
                          edgecolors="white", linewidth=0.6, **kwargs)
    ax.invert_yaxis()
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    if colorbar:
        bar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        bar.outline.set_visible(False)
        bar.set_label(label)
    return image


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
