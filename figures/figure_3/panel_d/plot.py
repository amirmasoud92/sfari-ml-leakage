"""Figure 3d: AUC with 95% confidence intervals for two convolutional networks, with children kept whole
(child-level AUC) and segments split (segment-level AUC). Columns of data.csv: network, scheme, auc_level, auc,
ci95_low, ci95_high.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "data.csv")
NETS = ["EEGNet", "ShallowConvNet"]
SCHEMES = [("kept whole", S.INK, "o"), ("segments split", S.LEAK, "s")]

plt = S.setup()
fig = plt.figure(figsize=(3.2, 2.35))
ax = fig.add_axes([0.62 / 3.2, 0.45 / 2.35, 2.5 / 3.2, 1.7 / 2.35])

for j, net in enumerate(NETS):
    for s_, (scheme, col, mk) in enumerate(SCHEMES):
        r = d[(d.network == net) & (d.scheme == scheme)].iloc[0]
        x = j * 3.0 + s_ * 1.0
        ax.plot([x, x], [r.ci95_low, r.ci95_high], color=col, lw=2.0, solid_capstyle="butt")
        for cap in (r.ci95_low, r.ci95_high):
            ax.plot([x - 0.12, x + 0.12], [cap, cap], color=col, lw=2.0)
        ax.plot(x, r.auc, mk, color=col, ms=7.5, mec="white", mew=0.6, zorder=3)
S.chance(ax)
ax.set_xticks([0.5, 3.5], NETS, fontsize=S.FS_LABEL)
ax.tick_params(axis="x", length=0, pad=4)
ax.set(xlim=(-0.8, 4.8), ylim=(0.2, 1.05), yticks=[0.3, 0.5, 0.7, 0.9])
ax.set_ylabel("AUC")

S.save(fig, HERE)
