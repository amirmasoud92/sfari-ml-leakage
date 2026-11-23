"""Supplementary Figure S2b: identification margin of log power and spectral shape over the sensor covariance, for
same-visit and other-visit pairs, with 95% intervals. Columns of data.csv: representation, visit (same or other),
margin_points, ci95_lo_points, ci95_hi_points (percentage points)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
REPS = [("log power", None), ("spectral shape", "white")]  # (representation, marker face; None = filled)
VISITS = ["same", "other"]


def errv(ax, x, y, lo, hi, col, mfc=None, ms=7, cap=0.1):
    """Vertical estimate with a capped interval."""
    ax.plot([x, x], [lo, hi], color=col, lw=1.8, solid_capstyle="butt", zorder=2)
    for e in (lo, hi):
        ax.plot([x - cap, x + cap], [e, e], color=col, lw=1.8, zorder=2)
    ax.plot(x, y, "o", ms=ms, color=col, mfc=col if mfc is None else mfc, mec=col, mew=1.6, zorder=3)


d = pd.read_csv(HERE / "data.csv").set_index(["representation", "visit"])

plt = S.setup()
fig = plt.figure(figsize=(3.3, 2.7))
ax = fig.add_axes([0.6 / 3.3, 0.6 / 2.7, 2.45 / 3.3, 1.95 / 2.7])
for i, (rep, mfc) in enumerate(REPS):
    for j, vis in enumerate(VISITS):
        r = d.loc[(rep, vis)]
        errv(ax, i * 2.4 + j, r.margin_points, r.ci95_lo_points, r.ci95_hi_points, S.IDENT_C, mfc=mfc)
ax.axhline(0, color=S.INK, lw=1.0)
ax.set_xticks([0, 1, 2.4, 3.4], VISITS * 2)
ax.set_xlim(-0.5, 3.9)
for x, (rep, _mfc) in zip((0.5, 2.9), REPS, strict=True):
    ax.text(x, -0.17, rep, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=S.FS_LABEL)
ax.set_ylabel("margin (points)")

S.save(fig, HERE)
