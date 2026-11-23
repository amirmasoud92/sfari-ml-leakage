"""Figure 2e: shares of the variance of alpha-band (8-12 Hz) power that lie between children and between diagnostic groups, per electrode.

data.csv gives, per BioSemi 64 channel, the between-children and diagnosis shares of variance in percent; the
medians over electrodes are written above the maps.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent

d = pd.read_csv(HERE / "data.csv").set_index("channel")
maps = [("between children", d.between_children_share_pct), ("diagnosis", d.diagnosis_share_pct)]

plt = S.setup()
fw, fh = 3.35, 2.4
fig = plt.figure(figsize=(fw, fh))
for i, (title, vals) in enumerate(maps):
    x0 = 0.3 + i * 1.52
    hx = fig.add_axes([x0 / fw, 0.74 / fh, 1.4 / fw, 1.36 / fh])
    im = S.topomap(hx, vals, 0, 100)
    med = vals.median()
    label = f"{title}: {med:.0f}%" if med >= 1 else f"{title}: {med:.2f}%"
    fig.text((x0 + 0.7) / fw, 2.14 / fh, label, fontsize=S.FS_MIN, ha="center", va="bottom")

cax = fig.add_axes([0.75 / fw, 0.47 / fh, 2.0 / fw, 0.085 / fh])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
cb.outline.set_linewidth(0.9)
cb.ax.tick_params(labelsize=S.FS_TICK, length=3, width=0.9, pad=2)
cb.set_label("share of variance (%)", fontsize=S.FS_LABEL, labelpad=2)

S.save(fig, HERE)
