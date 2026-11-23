"""Figure 2d: identification of the same child from the spectrum of a single electrode, interpolated over the scalp.

data.csv gives, per BioSemi 64 channel, the percentage of children identified with the average reference and after
a surface Laplacian; the ring marks the best electrode of the average-reference map.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent

d = pd.read_csv(HERE / "data.csv").set_index("channel")
maps = [
    ("average reference", d.identified_pct_average_reference),
    ("surface Laplacian", d.identified_pct_surface_laplacian),
]
best = d.identified_pct_average_reference.idxmax()

plt = S.setup()
fw, fh = 3.35, 2.4
fig = plt.figure(figsize=(fw, fh))
for i, (title, vals) in enumerate(maps):
    x0 = 0.3 + i * 1.52
    hx = fig.add_axes([x0 / fw, 0.74 / fh, 1.4 / fw, 1.36 / fh])
    im = S.topomap(hx, vals, 45, 92, mark=[best] if i == 0 else [])
    fig.text((x0 + 0.7) / fw, 2.14 / fh, title, fontsize=S.FS_MIN, ha="center", va="bottom")
    if i == 0:
        p = S.sensor_xy().loc[best]
        hx.annotate(
            best,
            (p.x, p.y),
            xytext=(0.098, -0.092),
            textcoords="data",
            fontsize=S.FS_MIN,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=1.0, color=S.INK, shrinkA=0, shrinkB=3),
            annotation_clip=False,
        )

cax = fig.add_axes([0.75 / fw, 0.47 / fh, 2.0 / fw, 0.085 / fh])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[45, 60, 75, 90])
cb.outline.set_linewidth(0.9)
cb.ax.tick_params(labelsize=S.FS_TICK, length=3, width=0.9, pad=2)
cb.set_label("identified from one electrode (%)", fontsize=S.FS_LABEL, labelpad=2)

S.save(fig, HERE)
