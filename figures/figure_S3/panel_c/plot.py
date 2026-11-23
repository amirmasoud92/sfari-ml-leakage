"""Figure S3c: child-level AUC with permuted labels (logistic pipeline) for balanced subsamples of 20, 40 and 70
core children, children kept whole (black) or blocks split (orange); dots are draws and bars medians.
Columns of data.csv: n_children, scheme, draw, auc_child.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "data.csv")
ns = sorted(d.n_children.unique())

plt = S.setup()
fig = plt.figure(figsize=(3.25, 2.4))
ax = fig.add_axes([0.62 / 3.25, 0.6 / 2.4, 2.55 / 3.25, 1.75 / 2.4])

for j, n in enumerate(ns):
    for scheme, col, dx in (("kept whole", S.INK, -0.2), ("blocks split", S.LEAK, 0.2)):
        v = d[(d.n_children == n) & (d.scheme == scheme)].auc_child.to_numpy()
        val = S.swarm(ax, j + dx, v, col, s=14, step=0.04, half=0.1, nbins=22)
        ax.plot([j + dx - 0.19, j + dx + 0.19], [np.median(val)] * 2, color=S.INK, lw=2.6, zorder=5,
                solid_capstyle="butt")
S.chance(ax)
ax.set_xticks(range(len(ns)), [str(n) for n in ns])
ax.set(xlim=(-0.6, 2.6), ylim=(0.2, 1.04), yticks=[0.3, 0.5, 0.7, 0.9])
ax.set_xlabel("children (balanced)")
ax.set_ylabel("child-level AUC")

S.save(fig, HERE)
