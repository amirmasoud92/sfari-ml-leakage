"""Figure 6b: percentage of candidate markers with an extreme deviation from the typically developing normative
model against clean recording minutes, in autistic children, with the Spearman correlation.

data.csv: clean_minutes and extreme_markers_pct (one row per autistic child). correlation.csv: spearman_rho and p.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "data.csv")
c = pd.read_csv(HERE / "correlation.csv").iloc[0]

plt = S.setup()
fw, fh = 3.4, 2.45
fig = plt.figure(figsize=(fw, fh))
ax = fig.add_axes([0.72 / fw, 0.6 / fh, 2.55 / fw, 1.72 / fh])

ax.scatter(d.clean_minutes, d.extreme_markers_pct, s=24, color=S.GROUP["ASD"], marker="s", lw=0)
ax.set(xlabel="clean minutes", ylabel="extreme markers (%)")
label = f"ρ = {c.spearman_rho:.2f}, p = {c.p:.3f}".replace("p = 0.", "p = .").replace("-", "−")
ax.text(0.98, 0.97, label, transform=ax.transAxes, ha="right", va="top", fontsize=S.FS_MIN)

S.save(fig, HERE)
