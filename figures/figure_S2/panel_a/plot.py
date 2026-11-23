"""Supplementary Figure S2a: identification accuracy from a recording of the same visit and of another visit, in the
children with both, for six representations. Columns of data.csv: representation, same_visit_identified_pct,
other_visit_identified_pct."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
CMP = "#4D535C"  # comparators
# representation -> colour, marker, face colour (None = filled)
LOOK = {
    "Spectral shape": (S.IDENT_C, "o", "white"),
    "Log power": (S.IDENT_C, "o", None),
    "Sensor covariance": (CMP, "s", None),
    "Periodic": (S.INK, "^", None),
    "Aperiodic": (S.INK, "v", None),
    "External": (CMP, "D", "white"),
}

d = pd.read_csv(HERE / "data.csv")

plt = S.setup()
fig = plt.figure(figsize=(3.2, 2.6))
ax = fig.add_axes([0.6 / 3.2, 0.5 / 2.6, 1.35 / 3.2, 1.95 / 2.6])
for r in d.itertuples():
    col, mk, mfc = LOOK[r.representation]
    a, b = r.same_visit_identified_pct, r.other_visit_identified_pct
    ax.plot([0, 1], [a, b], "-", color=col, lw=2.0, zorder=2)
    ax.plot([0, 1], [a, b], mk, ms=7, color=col, mfc=col if mfc is None else mfc, mew=1.6, zorder=3)
    ax.text(1.12, b, r.representation, va="center", fontsize=S.FS_MIN, color=col, clip_on=False)
ax.set(xticks=[0, 1], xlim=(-0.2, 1.08), ylim=(30, 100), yticks=[40, 60, 80, 100])
ax.set_xticklabels(["same", "other"])
ax.set_xlabel("visit")
ax.set_ylabel("identified (%)")

S.save(fig, HERE)
