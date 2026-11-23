"""Figure 5b: error-corrected AUC of each candidate marker with its 95% CI (coloured line) against its permutation
null (grey bar, median to 95th percentile); filled symbols mark an uncorrected permutation p < .05.

data.csv (rows bottom to top): marker, marker_type, auc_corrected, ci_low, ci_high, null_median,
null_95th_percentile, p_permutation.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "data.csv")

plt = S.setup()
fw, fh = 3.25, 3.4
fig = plt.figure(figsize=(fw, fh))
ax = fig.add_axes([1.7 / fw, 0.6 / fh, 1.45 / fw, 2.62 / fh])

for i, r in d.iterrows():
    col, mk = S.TYPE[r.marker_type]
    ax.plot([r.null_median, r.null_95th_percentile], [i, i], color=S.GREY, lw=7, solid_capstyle="butt", zorder=0)
    ax.plot([r.ci_low, min(r.ci_high, 1.0)], [i, i], color=col, lw=1.4, solid_capstyle="butt", zorder=1)
    sig = r.p_permutation < 0.05
    ax.plot(r.auc_corrected, i, mk, ms=6.5, color=col, mfc=col if sig else "white", mec=col, mew=1.5, zorder=3)

ax.set_yticks(range(len(d)), d.marker, fontsize=S.FS_MIN)
ax.tick_params(axis="y", length=0, pad=4)
ax.spines["left"].set_visible(False)
ax.set(xlim=(0.48, 1.0), ylim=(-0.7, len(d) - 0.3), xticks=[0.5, 0.75, 1.0])
ax.set_xticklabels(["0.5", "0.75", "1"])
ax.set_xlabel("error-corrected AUC")

S.save(fig, HERE)
