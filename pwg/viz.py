"""Shared publication style for matplotlib figures (Nature-like single/double column)."""
from __future__ import annotations

import matplotlib as mpl

COL1 = 3.4   # in
COL2 = 7.0

# categorical slots (validated reference palette, light mode), fixed order
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948",
         ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e4e3df")
CAT = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"],
       C["violet"], C["red"]]


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "axes.titleweight": "regular", "axes.titlepad": 3,
        "lines.linewidth": 1.0, "lines.markersize": 3,
        "legend.frameon": False, "legend.handlelength": 1.4,
        "savefig.dpi": 300, "figure.dpi": 150,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "image.interpolation": "nearest",
        "axes.prop_cycle": mpl.cycler(color=CAT),
    })


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right", color=C["ink"])


def save(fig, path_noext):
    fig.savefig(path_noext + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path_noext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
