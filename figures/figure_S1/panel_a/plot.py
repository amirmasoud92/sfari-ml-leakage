"""Supplementary Figure S1a: log-power identification accuracy for every ordered paradigm pair (source paradigm in
rows, target in columns), with within-paradigm split-half identification on the diagonal. Columns of data.csv:
source_paradigm, target_paradigm, comparison (cross-paradigm or within-paradigm split half), identified_pct."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
ORDER = ["ASSR", "AVSRT", "Beep-Flash", "FAST", "IC", "Rest"]

d = pd.read_csv(HERE / "data.csv")
M = d.pivot(index="source_paradigm", columns="target_paradigm", values="identified_pct")
M = M.reindex(index=ORDER, columns=ORDER).to_numpy(float)

plt = S.setup()
fig = plt.figure(figsize=(3.4, 3.4))
ax = fig.add_axes([0.28, 0.25, 2.35 / 3.4, 2.35 / 3.4])
ax.imshow(M, vmin=80, vmax=100, cmap=S.IDENT)
for i in range(len(ORDER)):
    for j in range(len(ORDER)):
        if np.isfinite(M[i, j]):
            ax.text(
                j,
                i,
                f"{M[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=S.FS_LABEL,
                fontweight="bold" if i == j else "normal",
                color="white" if M[i, j] >= 92 else S.INK,
            )
ax.set_xticks(range(len(ORDER)), ORDER, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(ORDER)), ORDER)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(True)
ax.set_xlabel("target paradigm")
ax.set_ylabel("source paradigm")

S.save(fig, HERE)
