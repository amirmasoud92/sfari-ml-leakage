"""Figure 5a: split-half reliability (Spearman-Brown) of 16 candidate markers against the absolute autistic-TD
difference (Hedges' g adjusted for age and sex), with dashed curves where the error-corrected AUC is 0.6, 0.7 and 0.8.

data.csv: marker, marker_type (spectral, evoked, behaviour), split_half_reliability, abs_hedges_g_adjusted and
annotation (the label written next to four highlighted markers; empty otherwise).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "data.csv", keep_default_na=False)

plt = S.setup()
fw, fh = 3.55, 3.4
fig = plt.figure(figsize=(fw, fh))
ax = fig.add_axes([0.68 / fw, 0.6 / fh, 2.75 / fw, 2.62 / fh])

# iso-AUC curves: |g| = sqrt(2 r) * Phi^-1(AUC)
rr = np.linspace(0.005, 1.08, 400)
for auc in (0.6, 0.7, 0.8):
    ax.plot(rr, np.sqrt(2 * rr) * stats.norm.ppf(auc), color=S.INK, lw=1.0, ls=(0, (4, 2.5)), zorder=1)
    ax.text(0.17, np.sqrt(2 * 0.17) * stats.norm.ppf(auc), f"AUC {auc:.1f}", fontsize=S.FS_MIN,
            ha="center", va="center", bbox=dict(fc="white", ec="none", pad=1.0), zorder=2)

for r in d.itertuples():
    col, mk = S.TYPE[r.marker_type]
    ax.plot(r.split_half_reliability, r.abs_hedges_g_adjusted, mk, ms=8, color=col, mec="white", mew=0.7, zorder=4)

# label positions (data units), placed between the dashed curves
place = {"ASSR 40 Hz": (0.55, 0.8, "left"), "Alpha rest − task": (0.69, 0.4, "right"),
         "Rest alpha": (1.07, 0.57, "right"), "Multisensory": (0.37, 0.055, "left")}
for r in d[d.annotation != ""].itertuples():
    tx, ty, ha = place[r.annotation]
    ax.annotate(r.annotation, (r.split_half_reliability, r.abs_hedges_g_adjusted), xytext=(tx, ty),
                textcoords="data", fontsize=S.FS_MIN, ha=ha, va="center", zorder=6,
                bbox=dict(fc="white", ec="none", pad=1.2),
                arrowprops=dict(arrowstyle="-", lw=0.9, color=S.INK, shrinkA=0, shrinkB=4.5))

ax.set(xlim=(0, 1.08), ylim=(0, 1.3), xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, 0.4, 0.8, 1.2])
ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
ax.set_xlabel("split-half reliability")
ax.set_ylabel("group difference, |g|")
handles = [Line2D([], [], marker=mk, ls="", ms=7, color=col, label=k) for k, (col, mk) in S.TYPE.items()]
ax.legend(handles=handles, loc="upper left", borderaxespad=0.3, labelspacing=0.3, handletextpad=0.2)

S.save(fig, HERE)
