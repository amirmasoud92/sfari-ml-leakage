"""Shared plotting conventions for the individual figure panels.

Colour meanings (also coded by marker shape where groups of points are compared):

- identity (the child's individual signature): teal;
- children kept whole: black; blocks of a child in both training and test sets (leaky): orange;
- runs with permuted labels: grey boxes or bands with a dark outline and a black median line;
- groups: TD blue, SIB green, ASD magenta;
- marker types: spectral teal circles, evoked purple squares, behavioural brown triangles.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

MM = 1 / 25.4

INK = "#000000"
GREY = "#B4BAC4"
GREY_EDGE = "#3F444C"
IDENT_C = "#00736B"
LEAK = "#E66100"
GROUP = {"TD": "#1A5FB4", "SIB": "#1B7F3B", "ASD": "#B0125A"}
TYPE = {"spectral": ("#00736B", "o"), "evoked": ("#6A3D9A", "s"), "behaviour": ("#8C4A00", "^")}

FS_MIN = 8.5
FS_TICK = 8.5
FS_LABEL = 9.5
FS_TITLE = 10.5

R_HEAD = 0.095  # head radius of the BioSemi 64 layout as MNE projects it


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "ident_hc", ["#FFFFFF", "#D7EFEA", "#8FD0C4", "#3DA79E", "#00736B", "#0B4F5C", "#0A2F45"]
    )


IDENT = _cmap()


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": FS_LABEL,
            "axes.titlesize": FS_TITLE,
            "axes.labelsize": FS_LABEL,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelpad": 3,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.pad": 2.5,
            "ytick.major.pad": 2.5,
            "legend.fontsize": FS_MIN,
            "legend.frameon": False,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.2,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "text.color": INK,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 100,
            "savefig.dpi": 600,
        }
    )
    return plt


def save(fig, folder: Path, name: str = "panel"):
    """Write the panel as PDF and 600-dpi PNG next to its script."""
    folder = Path(folder)
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{name}.png", dpi=600, bbox_inches="tight")


def chance(ax, v: float = 0.5, axis: str = "h"):
    f = ax.axhline if axis == "h" else ax.axvline
    f(v, color="#333333", lw=0.9, ls=(0, (1, 2.2)), zorder=0.4)


def swarm_x(v, step: float = 0.075, half: float = 0.3, nbins: int = 24) -> np.ndarray:
    """Horizontal offsets that spread equal or close values side by side (a dot histogram); values are not moved."""
    v = np.asarray(v, float)
    x = np.zeros(len(v))
    if len(v) < 2 or np.ptp(v) == 0:
        b = np.zeros(len(v), int)
    else:
        b = np.round((v - v.min()) / np.ptp(v) * nbins).astype(int)
    counts = np.bincount(b)
    if counts.max() * step > 2 * half:
        step = 2 * half / counts.max()
    for k in np.unique(b):
        idx = np.flatnonzero(b == k)
        idx = idx[np.argsort(v[idx], kind="stable")]
        off = (np.arange(len(idx)) - (len(idx) - 1) / 2) * step
        order = np.argsort(np.abs(off), kind="stable")  # centre first
        x[idx] = off[order]
    return x


def swarm(ax, x: float, v, color, s: float = 15, marker: str = "o", **kw):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    ax.scatter(x + swarm_x(v, **kw), v, s=s, color=color, marker=marker, lw=0, zorder=3)
    return v


def median_iqr(ax, x: float, v, ms: float = 5.5):
    """Interquartile range as a thick black bar with the median as a white-filled circle."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([x, x], [q1, q3], color=INK, lw=2.4, solid_capstyle="butt", zorder=4)
    ax.plot(x, med, "o", ms=ms, mfc="white", mec=INK, mew=1.6, zorder=5)


def null_box(ax, x: float, q, half: float = 0.3, horizontal: bool = False):
    """Central 95% of permuted-label runs as a grey box with a dark outline and a black median line."""
    from matplotlib.patches import Rectangle

    lo, hi = np.percentile(q, [2.5, 97.5])
    med = float(np.median(q))
    if hi - lo < 0.012:  # degenerate distribution (e.g. every run 1.00): keep the box visible
        lo, hi = med - 0.006, med + 0.006
    if horizontal:
        ax.add_patch(Rectangle((lo, x - half), hi - lo, 2 * half, fc=GREY, ec=GREY_EDGE, lw=0.8, zorder=1))
        ax.plot([med, med], [x - half, x + half], color=INK, lw=1.8, zorder=2, solid_capstyle="butt")
    else:
        ax.add_patch(Rectangle((x - half, lo), 2 * half, hi - lo, fc=GREY, ec=GREY_EDGE, lw=0.8, zorder=1))
        ax.plot([x - half, x + half], [med, med], color=INK, lw=1.8, zorder=2, solid_capstyle="butt")


_POS = None


def _info():
    import mne

    mont = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(mont.ch_names, 256, "eeg")
    info.set_montage(mont)
    return info


def sensor_xy() -> pd.DataFrame:
    """2-D sensor positions of the BioSemi 64 montage as MNE projects them."""
    global _POS
    if _POS is None:
        from mne.channels.layout import _find_topomap_coords

        info = _info()
        xy = _find_topomap_coords(info, picks=list(range(len(info.ch_names))), sphere=None)
        _POS = pd.DataFrame(xy, index=info.ch_names, columns=["x", "y"])
    return _POS


def head_outline(ax, lw: float = 1.3, z: float = 6):
    t = np.linspace(0, 2 * np.pi, 300)
    r = R_HEAD
    ax.plot(r * np.cos(t), r * np.sin(t), color=INK, lw=lw, zorder=z)
    nose = np.array([[-0.09, 0.995], [0, 1.11], [0.09, 0.995]]) * r
    ax.plot(nose[:, 0], nose[:, 1], color=INK, lw=lw, zorder=z, solid_joinstyle="round")
    ear = np.array(
        [[0.995, 0.09], [1.04, 0.12], [1.075, 0.10], [1.085, -0.02], [1.07, -0.14], [1.03, -0.17], [0.99, -0.13]]
    )
    for s in (1, -1):
        ax.plot(s * ear[:, 0] * r, ear[:, 1] * r, color=INK, lw=lw, zorder=z, solid_joinstyle="round")
    ax.set_xlim(-1.13 * r, 1.13 * r)
    ax.set_ylim(-1.06 * r, 1.14 * r)
    ax.set_aspect("equal")
    ax.axis("off")


def topomap(
    ax, values: pd.Series, vmin: float, vmax: float, cmap=None, sensors: bool = True, mark: Iterable[str] = ()
):
    """Interpolated scalp map (MNE) with a bold head outline; no contour lines."""
    import mne

    info = _info()
    v = values.reindex(info.ch_names).to_numpy(float)
    n_lines = len(ax.lines)
    im, _ = mne.viz.plot_topomap(
        np.nan_to_num(v, nan=np.nanmean(v)),
        info,
        axes=ax,
        show=False,
        contours=0,
        cmap=cmap or IDENT,
        vlim=(vmin, vmax),
        sensors=False,
        outlines="head",
        sphere=(0, 0, 0, R_HEAD),
        extrapolate="head",
        image_interp="cubic",
        res=256,
    )
    for ln in ax.lines[n_lines:]:
        ln.remove()
    pos = sensor_xy()
    if sensors:
        ax.scatter(pos.x, pos.y, s=2.2, color=INK, lw=0, zorder=7)
    mark = list(mark)
    if mark:
        p = pos.loc[mark]
        ax.scatter(p.x, p.y, s=34, facecolor="white", edgecolor=INK, lw=1.4, zorder=8)
    head_outline(ax)
    return im


def electrode_map(ax, chans: Iterable[str], color, ref: Iterable[str] = (), s: float = 26):
    """Head with every sensor as a small dot, the measured electrodes filled and reference electrodes open."""
    from matplotlib.patches import Circle

    pos = sensor_xy()
    ax.add_patch(Circle((0, 0), R_HEAD, fc="white", ec="none", zorder=0))
    ax.scatter(pos.x, pos.y, s=3.5, color="#555555", lw=0, zorder=2)
    p = pos.loc[list(chans)]
    ax.scatter(p.x, p.y, s=s, color=color, edgecolor=INK, lw=0.6, zorder=4)
    ref = list(ref)
    if ref:
        p = pos.loc[ref]
        ax.scatter(p.x, p.y, s=s, facecolor="white", edgecolor=INK, lw=1.2, zorder=5)
    head_outline(ax, lw=1.2)


def colorbar(fig, ax, im, label: str, ticks=None):
    """Horizontal colour bar under ``ax``."""
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=ticks, fraction=0.06, pad=0.04)
    cb.outline.set_linewidth(0.9)
    cb.ax.tick_params(labelsize=FS_TICK, length=3, width=0.9, pad=2)
    cb.set_label(label, fontsize=FS_LABEL, labelpad=2)
    return cb
