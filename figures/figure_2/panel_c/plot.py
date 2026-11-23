"""Figure 2c: share of the block-to-block variance of log power that lies between children and between diagnostic groups.

data.csv gives, per frequency bin (freq_hz), the between-children and diagnosis shares of variance (as proportions,
mean over electrodes) and the 2.5th and 97.5th percentiles of the diagnosis share over permutations of diagnosis
between children (grey band); the axis is logarithmic and labelled in percent.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent

d = pd.read_csv(HERE / "data.csv")
lo, hi = d.permuted_diagnosis_share_p2_5, d.permuted_diagnosis_share_p97_5

plt = S.setup()
fw, fh = 3.4, 1.85
fig = plt.figure(figsize=(fw, fh))
ax = fig.add_axes([0.62 / fw, 0.5 / fh, 2.55 / fw, 1.25 / fh])

ax.fill_between(d.freq_hz, lo, hi, color=S.GREY, lw=0, zorder=1)
for edge in (lo, hi):
    ax.plot(d.freq_hz, edge, color=S.GREY_EDGE, lw=0.8, zorder=1.5)
ax.plot(d.freq_hz, d.between_children_share, color=S.IDENT_C, lw=2.4, zorder=3)
ax.plot(d.freq_hz, d.diagnosis_share, color=S.GROUP["ASD"], lw=2.4, zorder=3)
ax.set_yscale("log")
ax.set(xlim=(2, 34), ylim=(3e-4, 2.5), xticks=[5, 10, 15, 20, 25, 30])
ax.set_yticks([1e-3, 1e-2, 1e-1, 1], ["0.1", "1", "10", "100"])
ax.minorticks_off()
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("share of variance (%)")

kw = dict(fontsize=S.FS_MIN, fontweight="bold", ha="right")
ax.text(33.3, 0.42, "between children", color=S.IDENT_C, va="top", **kw)
ax.text(33.3, hi.iloc[-1] * 0.78, "permuted labels", color=S.INK, va="top", **kw)
ax.text(33.3, 6.2e-4, "diagnosis", color=S.GROUP["ASD"], va="center", **kw)

S.save(fig, HERE)
