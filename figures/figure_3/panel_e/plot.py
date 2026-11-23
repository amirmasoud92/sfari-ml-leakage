"""Figure 3e: positive controls for the logistic pipeline, child-level AUC with age above the median and sex as
labels, children kept whole or blocks split. True-label repeats are shown as dots and the permuted-label runs as a
box (central 95%) with a median line. Columns of data.csv: target, scheme, labels (true or permuted), run, auc_child.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "data.csv")
TARGETS = ["age", "sex"]
SCHEMES = [("kept whole", S.INK), ("blocks split", S.LEAK)]


def strip_pair(ax, true_v, perm_v, x, col):
    """Permuted-label box (left) and the true-label repeats as dots ordered by value (right)."""
    S.null_box(ax, x - 0.22, perm_v, half=0.2)
    true_v = np.asarray(true_v, float)
    off = np.zeros(len(true_v))
    if len(true_v) > 1:
        off[np.argsort(true_v, kind="stable")] = np.linspace(-0.14, 0.14, len(true_v))
    ax.scatter(x + 0.27 + off, true_v, s=26, color=col, lw=0.5, edgecolor="white", zorder=3)


plt = S.setup()
fig = plt.figure(figsize=(3.2, 2.35))
ax = fig.add_axes([0.62 / 3.2, 0.45 / 2.35, 2.5 / 3.2, 1.7 / 2.35])

for j, target in enumerate(TARGETS):
    for s_, (scheme, col) in enumerate(SCHEMES):
        sel = d[(d.target == target) & (d.scheme == scheme)]
        strip_pair(
            ax,
            sel[sel.labels == "true"].auc_child.to_numpy(),
            sel[sel.labels == "permuted"].auc_child.to_numpy(),
            j * 3.0 + s_ * 1.0,
            col,
        )
S.chance(ax)
ax.set_xticks([0.5, 3.5], TARGETS, fontsize=S.FS_LABEL)
ax.tick_params(axis="x", length=0, pad=4)
ax.set(xlim=(-0.8, 4.8), ylim=(0.2, 1.05), yticks=[0.3, 0.5, 0.7, 0.9])
ax.set_ylabel("child-level AUC")

S.save(fig, HERE)
