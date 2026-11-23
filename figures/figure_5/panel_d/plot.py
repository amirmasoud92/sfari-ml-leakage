"""Figure 5d: individual values by group of the 40-Hz auditory steady-state transient response
(180 to 250 ms), with the electrodes it is computed from and the adjusted autistic-TD difference.

children.csv: group (TD, SIB, ASD) and value (one row per child, in the unit given in effect.csv). electrodes.csv:
channel and role (measured, or reference) on the BioSemi 64-channel layout. effect.csv: marker, marker_type, unit and
the age- and sex-adjusted autistic-TD Hedges' g with its 95% CI.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import style as S  # noqa: E402

HERE = Path(__file__).parent
GROUPS = ("TD", "SIB", "ASD")
kids = pd.read_csv(HERE / "children.csv")
elec = pd.read_csv(HERE / "electrodes.csv")
eff = pd.read_csv(HERE / "effect.csv").iloc[0]


def minus(x: float) -> str:
    return f"{x:.2f}".replace("-", "−")


plt = S.setup()
fw, fh = 1.95, 3.35
fig = plt.figure(figsize=(fw, fh))

# electrode inset with the adjusted group difference beside it
hx = fig.add_axes([0.1 / fw, 2.54 / fh, 0.7 / fw, 0.7 / fh])
S.electrode_map(hx, elec.channel[elec.role == "measured"], S.TYPE[eff.marker_type][0],
                ref=elec.channel[elec.role == "reference"], s=20)
fig.text(0.85 / fw, 2.99 / fh, f"g = {minus(eff.hedges_g_adjusted)}", fontsize=S.FS_LABEL, fontweight="bold",
         va="center", color=S.INK)
fig.text(0.85 / fw, 2.79 / fh, f"{minus(eff.ci_low)} to {minus(eff.ci_high)}", fontsize=S.FS_MIN, va="center",
         color=S.INK)

# children by group: dots, then median (white circle) and interquartile range (black bar)
ax = fig.add_axes([0.65 / fw, 0.72 / fh, 0.92 / fw, 1.62 / fh])
for i, gr in enumerate(GROUPS):
    val = S.swarm(ax, i - 0.1, kids.value[kids.group == gr].to_numpy(), S.GROUP[gr], s=11, step=0.11, half=0.26,
                  nbins=18)
    S.median_iqr(ax, i + 0.36, val, ms=5)
ax.set_xticks(range(3), [f"{gr}\n{(kids.group == gr).sum()}" for gr in GROUPS])
ax.tick_params(axis="x", length=0, pad=3)
ax.set_xlim(-0.55, 2.62)
ax.set_ylabel(eff.unit, labelpad=3)

S.save(fig, HERE)
