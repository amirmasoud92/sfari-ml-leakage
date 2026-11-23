"""Child-bootstrap 95% intervals of the cross-paradigm ICCs of the four posterior spectral markers over the six
paradigms in the core set (Additional file 1, Table S4). Same ICC(2,1) / ICC(2,k) definition as
:mod:`markers.reliability_validity`; 2,000 resamples of children.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from markers.reliability_validity import icc2
from stats.rng import generator
from utils import common as C
from utils.inputs import Inputs


def main() -> None:
    out = C.out_dir("cross_paradigm_icc")
    core = Inputs(C.CFG).core()
    sp = pd.read_parquet(C.OUT / "marker_halves/spectral_by_paradigm.parquet")
    rng, seed = generator(C.SEED, "v2_b5")
    rows = []
    for mk, d in sp.groupby("marker"):
        Y = d.pivot(index="child", columns="paradigm", values="v").reindex(core).dropna().to_numpy(float)
        a, k = icc2(Y)
        bs = np.array([icc2(Y[rng.integers(0, len(Y), len(Y))]) for _ in range(2000)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        rows.append(
            {
                "marker": mk,
                "n": len(Y),
                "icc_2_1": a,
                "icc_2_1_lo": lo[0],
                "icc_2_1_hi": hi[0],
                "icc_2_k": k,
                "icc_2_k_lo": lo[1],
                "icc_2_k_hi": hi[1],
                "seed": seed,
            }
        )
    T = pd.DataFrame(rows)
    T.to_parquet(out / "icc.parquet")
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
