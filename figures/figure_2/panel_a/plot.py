"""Figure 2a: identification of the same child across paradigms from log power and under each control representation.

data.csv gives, per row, the percentage of children identified (identified_pct) with its 95% child-bootstrap CI
(ci95_lo_pct, ci95_hi_pct), the chance level of the identification (chance_pct) and, for the matched-competitor
row only, that row's own chance level (row_chance_pct); kind marks the external-electrode row drawn in grey.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
COV = "#4D535C"  # rows that are not the EEG signature itself

d = pd.read_csv(HERE / "data.csv")
plt = S.setup()
fw, fh = 6.3, 2.35
fig = plt.figure(figsize=(fw, fh))
ax = fig.add_axes([1.95 / fw, 0.5 / fh, 3.95 / fw, 1.62 / fh])

y = np.arange(len(d))[::-1]
for yi, r in zip(y, d.itertuples(), strict=True):
    col = COV if r.kind == "external electrodes" else S.IDENT_C
    ax.plot([0, r.identified_pct], [yi, yi], color=col, lw=3.2, solid_capstyle="butt", zorder=1)
    ax.plot([r.ci95_lo_pct, r.ci95_hi_pct], [yi, yi], color=S.INK, lw=1.6, solid_capstyle="butt", zorder=3)
    for e in (r.ci95_lo_pct, r.ci95_hi_pct):
        ax.plot([e, e], [yi - 0.2, yi + 0.2], color=S.INK, lw=1.6, zorder=3)
    ax.plot(r.identified_pct, yi, "o", ms=7, color=col, mec=S.INK, mew=0.8, zorder=2)
    if np.isfinite(r.row_chance_pct):
        ax.plot([r.row_chance_pct] * 2, [yi - 0.32, yi + 0.32], color=S.INK, lw=1.6, zorder=4)
    ax.text(103.5, yi, f"{r.identified_pct:.0f}%", va="center", ha="left", fontsize=S.FS_LABEL, clip_on=False)

ch0 = d.chance_pct.iloc[0]
ax.axvline(ch0, color=S.INK, lw=1.0, ls=(0, (1, 2.2)), zorder=5)
ax.text(ch0 + 1.5, len(d) - 0.42, "chance", fontsize=S.FS_MIN, va="bottom", ha="left")
ax.set_yticks(y, d.control, fontsize=S.FS_LABEL)
ax.tick_params(axis="y", length=0, pad=5)
ax.spines["left"].set_visible(False)
ax.set(xlim=(0, 100), ylim=(-0.6, len(d) - 0.4), xticks=[0, 25, 50, 75, 100])
ax.set_xlabel("same child identified (%)")

S.save(fig, HERE)
