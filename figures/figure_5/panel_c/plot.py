"""Figure 5c: split-half reliability of each candidate marker in autistic against typically developing children,
with the line of identity.

data.csv: marker, marker_type (spectral, evoked, behaviour), reliability_td, reliability_autistic.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "data.csv")

plt = S.setup()
fw, fh = 2.4, 2.25
fig = plt.figure(figsize=(fw, fh))
ax = fig.add_axes([0.68 / fw, 0.55 / fh, 1.6 / fw, 1.6 / fh])

ax.plot([0, 1], [0, 1], color=S.INK, lw=1.0, zorder=0)
for r in d.itertuples():
    col, mk = S.TYPE[r.marker_type]
    ax.plot(r.reliability_td, r.reliability_autistic, mk, color=col, ms=7, mec="white", mew=0.6)
ax.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], aspect="equal")
ax.set_xticklabels(["0", "0.5", "1"])
ax.set_yticklabels(["0", "0.5", "1"])
ax.set_xlabel("TD")
ax.set_ylabel("autistic")

S.save(fig, HERE)
