"""Figure 2b: identification of the same child from a single 1-Hz frequency bin of all 64 electrodes.

data.csv gives, for each reference (average reference, surface Laplacian) and frequency bin (freq_hz), the
percentage of children identified (identified_pct); the dot marks the best bin of the average reference.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent

d = pd.read_csv(HERE / "data.csv")
avg = d[d.reference == "average reference"].sort_values("freq_hz")
csd = d[d.reference == "surface Laplacian"].sort_values("freq_hz")

plt = S.setup()
fw, fh = 3.4, 1.85
fig = plt.figure(figsize=(fw, fh))
ax = fig.add_axes([0.62 / fw, 0.5 / fh, 2.55 / fw, 1.25 / fh])

ax.plot(avg.freq_hz, avg.identified_pct, color=S.IDENT_C, lw=2.4, label="average reference")
ax.plot(csd.freq_hz, csd.identified_pct, color=S.INK, lw=1.5, ls=(0, (4, 2)), label="surface Laplacian")
pk = avg.loc[avg.identified_pct.idxmax()]
ax.plot(pk.freq_hz, pk.identified_pct, "o", ms=6.5, color=S.IDENT_C, mec=S.INK, mew=0.9, zorder=4)
ax.plot([4, 12], [96, 96], color=S.INK, lw=1.6, solid_capstyle="butt")  # theta-alpha range
ax.text(8, 97, "4–12 Hz", fontsize=S.FS_MIN, ha="center", va="bottom")
ax.set(xlim=(2, 34), ylim=(45, 104), yticks=[50, 75, 100], xticks=[5, 10, 15, 20, 25, 30])
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("identified (%)")
ax.legend(loc="lower left", handlelength=2.2, borderaxespad=0.3)

S.save(fig, HERE)
