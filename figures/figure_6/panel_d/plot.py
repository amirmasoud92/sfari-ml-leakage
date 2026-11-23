"""Figure 6d: self-identifiability (z) against clean recording minutes in all children with at least two
paradigms, with the Spearman correlation and its 95% CI.

data.csv: group (TD, SIB, ASD), clean_minutes and self_identifiability_z (one row per child). correlation.csv:
spearman_rho, ci_low, ci_high.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
GROUPS = ("TD", "SIB", "ASD")
MARKS = {"TD": "o", "SIB": "^", "ASD": "s"}
d = pd.read_csv(HERE / "data.csv")
c = pd.read_csv(HERE / "correlation.csv").iloc[0]

plt = S.setup()
fw, fh = 3.4, 2.45
fig = plt.figure(figsize=(fw, fh))
ax = fig.add_axes([0.72 / fw, 0.6 / fh, 2.55 / fw, 1.72 / fh])

for gr in GROUPS:
    g = d[d.group == gr]
    ax.scatter(g.clean_minutes, g.self_identifiability_z, s=20, color=S.GROUP[gr], marker=MARKS[gr], lw=0, label=gr)
ax.set(xlabel="clean minutes", ylabel="self-identifiability (z)")
label = f"ρ = {c.spearman_rho:.2f} ({c.ci_low:.2f} to {c.ci_high:.2f})".replace("-", "−")
ax.text(0.03, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=S.FS_MIN)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), handletextpad=0.1, borderaxespad=0.2, labelspacing=0.2,
          markerscale=1.2)

S.save(fig, HERE)
