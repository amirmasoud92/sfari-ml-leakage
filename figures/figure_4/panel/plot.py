"""Figure 4: cross-validated AUC for separating autistic from typically developing children, from covariates and from EEG.

data.csv gives, per model, the AUC estimate (filled circle: mean within-fold AUC; open circle: pooled AUC), the
lower and upper 95% interval bounds and the interval type. permuted_label_runs.csv holds the within-fold AUC of all
EEG features in each label-permuted rerun, drawn as a density shape behind that row.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
COV = "#4D535C"  # covariate models

d = pd.read_csv(HERE / "data.csv")
null = pd.read_csv(HERE / "permuted_label_runs.csv").within_fold_auc.to_numpy()

plt = S.setup()
fw, fh = 5.9, 3.15
fig = plt.figure(figsize=(fw, fh))
ax = fig.add_axes([1.75 / fw, 0.6 / fh, 2.75 / fw, 2.42 / fh])
ys = np.array([5.5, 4.5, 3.5, 2.0, 1.0, 0.0])

# label-permuted reruns of all EEG features, as a symmetric density shape on that row
y_null = ys[d.index[d.model == "All EEG features"][0]]
grid = np.linspace(null.min(), null.max(), 200)
dens = gaussian_kde(null)(grid)
dens = dens / dens.max() * 0.42
ax.fill_between(grid, y_null - dens, y_null + dens, color=S.GREY, lw=0, zorder=0)
ax.plot(grid, y_null - dens, color=S.GREY_EDGE, lw=0.8, zorder=0.5)
ax.plot(grid, y_null + dens, color=S.GREY_EDGE, lw=0.8, zorder=0.5)
S.chance(ax, 0.5, axis="v")

for yi, r in zip(ys, d.itertuples(), strict=True):
    col = COV if r.feature_set == "covariates" else S.INK
    filled = r.estimate == "mean within-fold AUC"
    if np.isfinite(r.ci95_lo):
        ax.plot([r.ci95_lo, r.ci95_hi], [yi, yi], color=col, lw=2.0, zorder=2, solid_capstyle="butt")
        for cap in (r.ci95_lo, r.ci95_hi):
            ax.plot([cap, cap], [yi - 0.16, yi + 0.16], color=col, lw=2.0, zorder=2)
        txt = f"{r.auc:.2f} ({r.ci95_lo:.2f}–{r.ci95_hi:.2f})"
    else:
        txt = f"{r.auc:.2f}"
    ax.plot(r.auc, yi, "o", color=col, ms=8, mfc=col if filled else "white", mew=1.8, zorder=3)
    ax.text(1.04, yi, txt, fontsize=S.FS_LABEL, va="center", ha="left", transform=ax.get_yaxis_transform())

ax.axhline(2.75, color=S.INK, lw=0.8)  # covariate models above, EEG models below
ax.set_yticks(ys, d.model, fontsize=S.FS_LABEL)
ax.tick_params(axis="y", length=0, pad=5)
ax.spines["left"].set_visible(False)
ax.set(xlim=(0.2, 1.0), ylim=(-0.7, 6.1), xticks=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("cross-validated AUC, autistic vs TD")

S.save(fig, HERE)
