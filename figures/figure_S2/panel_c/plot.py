"""Supplementary Figure S2c: change in self-similarity (correlation r) per hour between recordings of the same visit,
for log power, spectral shape and the sensor covariance, with 95% intervals. Columns of data.csv: representation,
slope_r_per_hour, ci95_lo, ci95_hi (all in r per hour)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
CMP = "#4D535C"  # comparator
LOOK = {"Log power": (S.IDENT_C, None), "Shape": (S.IDENT_C, "white"), "Covariance": (CMP, None)}


def errh(ax, x, y, lo, hi, col, mfc=None, ms=7, cap=0.16):
    """Horizontal estimate with a capped interval."""
    ax.plot([lo, hi], [y, y], color=col, lw=1.8, solid_capstyle="butt", zorder=2)
    for e in (lo, hi):
        ax.plot([e, e], [y - cap, y + cap], color=col, lw=1.8, zorder=2)
    ax.plot(x, y, "o", ms=ms, color=col, mfc=col if mfc is None else mfc, mec=col, mew=1.6, zorder=3)


d = pd.read_csv(HERE / "data.csv")

plt = S.setup()
fig = plt.figure(figsize=(3.4, 2.1))
ax = fig.add_axes([1.15 / 3.4, 0.6 / 2.1, 2.05 / 3.4, 1.38 / 2.1])
for i, r in enumerate(d.itertuples()):
    col, mfc = LOOK[r.representation]
    errh(ax, r.slope_r_per_hour, -i, r.ci95_lo, r.ci95_hi, col, mfc=mfc)
ax.axvline(0, color=S.INK, lw=1.0)
ax.set_yticks(-np.arange(len(d)), d.representation)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, pad=5)
ax.set_ylim(-2.6, 0.6)
ax.set_xlabel("r per hour")

S.save(fig, HERE)
