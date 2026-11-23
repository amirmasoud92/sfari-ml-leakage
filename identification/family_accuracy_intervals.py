"""Child-bootstrap 95% intervals of each feature family's mean cross-paradigm identification accuracy (Fig. S1b;
Additional file 1, Table S16, H1a).

The stored correct-identification indicators of :mod:`identification.analysis` (child x ordered paradigm pair) are
resampled over children (``prediction.n_bootstrap`` = 2,000 resamples) with the competitor sets of the original
identification held fixed; in every resample each pair's accuracy is the mean correctness of the resampled children
who have that pair, and the statistic is the mean over pairs. Written to
``<results>/confirmatory/identification_accuracy_ci.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.loader import Config
from stats.rng import generator


def _read(cfg: Config, stage: str, name: str) -> pd.DataFrame | None:
    p = cfg.path("results") / stage / f"{name}.parquet"
    return pd.read_parquet(p) if p.is_file() else None


def identification_accuracy_ci(cfg: Config) -> pd.DataFrame | None:
    """95% CI of each family's mean cross-paradigm identification accuracy: a bootstrap over children of the stored
    correct-identification indicators (the gallery of the original analysis held fixed), mean over ordered pairs.
    Written to ``<results>/confirmatory/identification_accuracy_ci.csv``."""
    pc = _read(cfg, "confirmatory", "identification__per_child")
    if pc is None:
        return None
    out = cfg.path("results") / "confirmatory" / "identification_accuracy_ci.csv"
    res = accuracy_ci(cfg, pc)
    res.to_csv(out, index=False)
    return res.set_index("family")


def accuracy_ci(cfg: Config, pc: pd.DataFrame, tag: str = "") -> pd.DataFrame:
    """Per family: mean accuracy over ordered pairs and its child-bootstrap 95% CI, from a ``per_child`` table."""
    rows = []
    for fam, d in pc.groupby("family"):
        C = (
            d.assign(pair=d.source + "|" + d.target, correct=d.correct.astype(float))
            .pivot_table(index="child", columns="pair", values="correct")
            .to_numpy()
        )
        P = np.isfinite(C).astype(float)
        C = np.nan_to_num(C)
        key = ("paper", "identification_ci", fam) if not tag else ("paper", "identification_ci", tag, fam)
        rng, seed = generator(cfg.project.seed, *key)
        W = rng.multinomial(C.shape[0], np.full(C.shape[0], 1 / C.shape[0]), size=cfg.prediction.n_bootstrap)
        draws = ((W @ C) / np.maximum(W @ P, 1)).mean(axis=1)
        est = (C.sum(0) / P.sum(0)).mean()
        lo, hi = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "family": fam,
                "mean_accuracy": est,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_children": C.shape[0],
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from config.loader import load_config

    print(identification_accuracy_ci(load_config()))
