"""Figure 3c: child-level AUC (autistic versus TD core children) for four model families with children kept whole
or blocks split. True-label repeats are shown as dots and the permuted-label runs as a box (central 95%) with a
median line. Columns of data.csv: model, scheme, labels (true or permuted), run, auc_child.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "data.csv")
MODELS = ["logistic", "RBF SVM", "random forest", "10-NN"]
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

fig = plt.figure(figsize=(6.9, 2.15))
ax = fig.add_axes([0.62 / 6.9, 0.45 / 2.15, 6.13 / 6.9, 1.25 / 2.15])

for j, model in enumerate(MODELS):
    for s_, (scheme, col) in enumerate(SCHEMES):
        sel = d[(d.model == model) & (d.scheme == scheme)]
        strip_pair(
            ax,
            sel[sel.labels == "true"].auc_child.to_numpy(),
            sel[sel.labels == "permuted"].auc_child.to_numpy(),
            j * 3.0 + s_ * 1.0,
            col,
        )
S.chance(ax)
ax.set_xticks([j * 3.0 + 0.5 for j in range(len(MODELS))], MODELS, fontsize=S.FS_LABEL)
ax.tick_params(axis="x", length=0, pad=4)
ax.set(xlim=(-0.8, 10.8), ylim=(0.25, 1.05), yticks=[0.3, 0.5, 0.7, 0.9])
ax.set_ylabel("child-level AUC")
handles = [
    Line2D([], [], marker="o", ls="", ms=6, color=S.INK, label="kept whole"),
    Line2D([], [], marker="o", ls="", ms=6, color=S.LEAK, label="blocks split"),
    Patch(fc=S.GREY, ec=S.GREY_EDGE, lw=0.8, label="permuted labels"),
]
ax.legend(
    handles=handles,
    loc="lower right",
    bbox_to_anchor=(1.0, 1.06),
    ncol=3,
    columnspacing=1.2,
    handletextpad=0.3,
    borderaxespad=0,
)

S.save(fig, HERE)
