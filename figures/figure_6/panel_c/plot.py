"""Figure 6c: self-identifiability (z) of each child's log-power spectrum in the core paradigm set, by group.

data.csv: group (TD, SIB, ASD) and self_identifiability_z (one row per child).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
GROUPS = ("TD", "SIB", "ASD")
d = pd.read_csv(HERE / "data.csv")

plt = S.setup()
fw, fh = 3.4, 2.45
fig = plt.figure(figsize=(fw, fh))
ax = fig.add_axes([0.72 / fw, 0.6 / fh, 2.55 / fw, 1.72 / fh])

for i, gr in enumerate(GROUPS):
    val = S.swarm(ax, i - 0.08, d.self_identifiability_z[d.group == gr].to_numpy(), S.GROUP[gr], s=16, step=0.085,
                  half=0.3, nbins=20)
    S.median_iqr(ax, i + 0.4, val)
ax.set_xticks(range(3), [f"{gr}\n{(d.group == gr).sum()}" for gr in GROUPS])
ax.tick_params(axis="x", length=0, pad=3)
ax.set(xlim=(-0.55, 2.65), ylabel="self-identifiability (z)")

S.save(fig, HERE)
