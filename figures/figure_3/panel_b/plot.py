"""Figure 3b: child-level AUC of the logistic pipeline as k blocks of each test child are added to training.

summary.csv holds the plotted values per k (median and central 95% of the permuted-label runs, median of the
permuted-label runs leaking only other paradigms, and mean and range of the true-label repeats); runs.csv holds the
individual runs (series, k, run, auc_child) from which they are computed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "summary.csv").sort_values("k")
ks = d.k.to_numpy()
xs = np.arange(len(ks))

plt = S.setup()
fig = plt.figure(figsize=(3.2, 2.15))
ax = fig.add_axes([0.7 / 3.2, 0.6 / 2.15, 2.2 / 3.2, 1.35 / 2.15])

# permuted labels: central 95% band and median
ax.fill_between(xs, d.permuted_q025, d.permuted_q975, color=S.GREY, lw=0, zorder=1)
ax.plot(xs, d.permuted_q025, color=S.GREY_EDGE, lw=0.8, zorder=1.5)
ax.plot(xs, d.permuted_q975, color=S.GREY_EDGE, lw=0.8, zorder=1.5)
ax.plot(xs, d.permuted_median, color=S.LEAK, lw=2.4, marker="o", ms=5.5, zorder=4, label="permuted labels")
# permuted labels, leaked blocks from the child's other paradigms only
ax.plot(
    xs,
    d.permuted_other_paradigms_median,
    color=S.LEAK,
    lw=1.8,
    ls=(0, (4, 2)),
    marker="s",
    ms=5,
    mfc="white",
    mew=1.4,
    zorder=3,
    label="permuted, other paradigms",
)
# true labels: mean and range over repeats
ax.errorbar(
    xs,
    d.true_mean,
    yerr=[d.true_mean - d.true_min, d.true_max - d.true_mean],
    color=S.INK,
    lw=1.8,
    marker="o",
    ms=5,
    capsize=0,
    zorder=5,
    label="true labels",
)
ax.axhline(0.5, color=S.INK, lw=0.9, ls=(0, (1, 2.2)), zorder=2)

ax.set_xticks(xs, [str(k) for k in ks])
ax.set(ylim=(0.3, 0.97), yticks=[0.3, 0.5, 0.7, 0.9], xlim=(-0.3, len(ks) - 0.7))
ax.set_xlabel("leaked blocks per test child (k)")
ax.set_ylabel("child-level AUC")
ax.legend(loc="upper left", handlelength=2.4, borderaxespad=0.1, labelspacing=0.25)

S.save(fig, HERE)
