"""Supplementary Figure S2d: shares of variance attributed to the child, the visit and the residual for three posterior
features, in children with several visits. Columns of data.csv: feature, child_share, visit_share, residual_share
(fractions that sum to 1)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
PURPLE = "#6A3D9A"
LIGHT_GREY = "#C3C8D0"
PARTS = [("child_share", S.IDENT_C, "child"), ("visit_share", PURPLE, "visit"), ("residual_share", LIGHT_GREY, "residual")]

d = pd.read_csv(HERE / "data.csv")

plt = S.setup()
fig = plt.figure(figsize=(3.2, 2.1))
ax = fig.add_axes([1.1 / 3.2, 0.5 / 2.1, 2.0 / 3.2, 1.2 / 2.1])
left = np.zeros(len(d))
yy = np.arange(len(d))[::-1]
for col_name, col, label in PARTS:
    vals = d[col_name].to_numpy()
    ax.barh(yy, vals, left=left, color=col, height=0.62, label=label, edgecolor=S.INK, lw=0.6)
    left += vals
ax.set_yticks(yy, d.feature)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, pad=5)
ax.set(xlim=(0, 1), xticks=[0, 0.5, 1])
ax.set_xlabel("share of variance")
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.45, 1.02),
    ncol=3,
    columnspacing=0.8,
    handlelength=1.2,
    handletextpad=0.3,
    borderaxespad=0,
)

S.save(fig, HERE)
