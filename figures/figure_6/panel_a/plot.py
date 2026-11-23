"""Figure 6a: percentage of candidate markers on which each child deviates extremely (|z| > 1.96) from a normative
model of typically developing children, by group; the dotted line marks the 5% expected under that model.

data.csv: group (TD, SIB, ASD) and extreme_markers_pct (one row per child).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
GROUPS = ("TD", "SIB", "ASD")
d = pd.read_csv(HERE / "data.csv")

plt = S.setup()
fw, fh = 3.4, 2.45
fig = plt.figure(figsize=(fw, fh))
ax = fig.add_axes([0.72 / fw, 0.6 / fh, 2.55 / fw, 1.72 / fh])

for i, gr in enumerate(GROUPS):
    val = S.swarm(ax, i - 0.08, d.extreme_markers_pct[d.group == gr].to_numpy(), S.GROUP[gr], s=16, step=0.05,
                  half=0.3)
    S.median_iqr(ax, i + 0.4, val)
ax.axhline(5, color=S.INK, lw=0.9, ls=(0, (1, 2.2)), zorder=0.4)
ax.set_xticks(range(3), [f"{gr}\n{(d.group == gr).sum()}" for gr in GROUPS])
ax.tick_params(axis="x", length=0, pad=3)
ax.set(xlim=(-0.55, 2.65), ylabel="extreme markers (%)")

S.save(fig, HERE)
