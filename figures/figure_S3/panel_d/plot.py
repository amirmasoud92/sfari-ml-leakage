"""Figure S3d: EEGNet child-level AUC for runs with diagnosis permuted between children (dots; bars, medians) and
for the true labels (diamonds), with children kept whole and with segments split.
Columns of data.csv: scheme, labels (permuted or true), run, auc_child.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "data.csv")

plt = S.setup()
fig = plt.figure(figsize=(3.25, 2.4))
ax = fig.add_axes([0.62 / 3.25, 0.6 / 2.4, 2.55 / 3.25, 1.75 / 2.4])

for j, (scheme, col) in enumerate((("kept whole", S.INK), ("segments split", S.LEAK))):
    v = d[(d.labels == "permuted") & (d.scheme == scheme)].auc_child.to_numpy()
    val = S.swarm(ax, j - 0.12, v, col, s=16, step=0.055, half=0.16, nbins=22)
    ax.plot([j - 0.38, j + 0.14], [np.median(val)] * 2, color=S.INK, lw=2.6, zorder=5, solid_capstyle="butt")
    t = d[(d.labels == "true") & (d.scheme == scheme)].auc_child
    if len(t):
        ax.plot(j + 0.32, t.iloc[0], "D", ms=8, color=col, mec=S.INK, mew=1.0, zorder=6)
S.chance(ax)
ax.set_xticks([0, 1], ["kept whole", "segments split"])
ax.tick_params(axis="x", length=0)
ax.set(xlim=(-0.6, 1.6), ylim=(0.2, 1.04), yticks=[0.3, 0.5, 0.7, 0.9])
ax.set_ylabel("child-level AUC")

S.save(fig, HERE)
