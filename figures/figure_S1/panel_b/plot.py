"""Supplementary Figure S1b: mean identification accuracy over the 30 ordered paradigm pairs for each feature family,
with 95% bootstrap intervals; the dotted line is chance. Columns of data.csv: feature_family, role (feature family or
comparator, the latter drawn in grey), identified_pct, ci95_lo_pct, ci95_hi_pct, chance_pct."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
CMP = "#4D535C"  # comparators (sensor covariance, external electrodes)


def errh(ax, x, y, lo, hi, col, ms=7, cap=0.16):
    """Horizontal estimate with a capped interval."""
    ax.plot([lo, hi], [y, y], color=col, lw=1.8, solid_capstyle="butt", zorder=2)
    for e in (lo, hi):
        ax.plot([e, e], [y - cap, y + cap], color=col, lw=1.8, zorder=2)
    ax.plot(x, y, "o", ms=ms, color=col, mec=col, mew=1.6, zorder=3)


d = pd.read_csv(HERE / "data.csv")

plt = S.setup()
fig = plt.figure(figsize=(3.2, 3.6))
ax = fig.add_axes([1.4 / 3.2, 0.6 / 3.6, 1.62 / 3.2, 2.75 / 3.6])
y = np.arange(len(d))[::-1]
for yi, r in zip(y, d.itertuples(), strict=True):
    col = CMP if r.role == "comparator" else S.IDENT_C
    errh(ax, r.identified_pct, yi, r.ci95_lo_pct, r.ci95_hi_pct, col)
S.chance(ax, d.chance_pct.iloc[0], axis="v")
ax.set_yticks(y, d.feature_family)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, pad=5)
ax.set(xlim=(0, 100), ylim=(-0.6, len(d) - 0.4), xticks=[0, 50, 100])
ax.set_xlabel("identified (%)")

S.save(fig, HERE)
