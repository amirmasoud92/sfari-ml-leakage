"""Figure S3b: cross-visit leak, child-level AUC on the scored visit with labels permuted between children when
k blocks from the child's other visit are added to training, for the 13 core children with two visits (orange;
grey band, central 95% of runs) and the 28 children with two visits among all children (dark orange).
summary.csv holds the plotted values (series, k, median, q025, q975); runs.csv the individual permutation runs
(series, k, run, auc_child).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "summary.csv")
DARK_ORANGE = "#9E3D00"


def curve(ax, s, col, ls="-", marker="o", mfc=None, label=None, band=True):
    s = s.sort_values("k")
    xs = np.arange(len(s))
    if band:
        ax.fill_between(xs, s.q025, s.q975, color=S.GREY, lw=0, zorder=1)
        ax.plot(xs, s.q025, color=S.GREY_EDGE, lw=0.8, zorder=1.5)
        ax.plot(xs, s.q975, color=S.GREY_EDGE, lw=0.8, zorder=1.5)
    ax.plot(xs, s["median"], color=col, ls=ls, lw=2.2, marker=marker, ms=6,
            mfc=col if mfc is None else mfc, mew=1.5, zorder=4, label=label)
    return s.k.to_numpy()


plt = S.setup()
fig = plt.figure(figsize=(3.25, 2.6))
ax = fig.add_axes([0.62 / 3.25, 0.55 / 2.6, 2.55 / 3.25, 1.95 / 2.6])

curve(ax, d[d.series == "28 two-visit children"], DARK_ORANGE, ls=(0, (4, 2)), marker="s", mfc="white",
      label="28 two-visit children", band=False)
ks = curve(ax, d[d.series == "13 core children"], S.LEAK, label="13 core children")
S.chance(ax)
ax.set_xticks(np.arange(len(ks)), [str(k) for k in ks])
ax.set(ylim=(0.1, 1.0), xlim=(-0.3, len(ks) - 0.7))
ax.set_xlabel("leaked blocks per test child (k)")
ax.set_ylabel("child-level AUC")
ax.legend(loc="upper left", borderaxespad=0.2, handlelength=2.4, labelspacing=0.3)

S.save(fig, HERE)
