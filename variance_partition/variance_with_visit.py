"""Variance components with a visit term (Additional file 1, Table S5 and Fig. S2d).

For most children every paradigm comes from one visit, so the child term also holds visit-level variance. The
children recorded on two or three dates identify a visit-within-child term. For the posterior region-of-interest
aperiodic exponent, aperiodic offset and periodic alpha (block level), a linear mixed model with paradigm as fixed
effect, a random child intercept and a random visit-within-child intercept is fitted by REML in all children and in
the multi-visit children alone. Shares: child, visit within child, residual (child x paradigm + block).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import common as cm  # not "C": patsy formulas use C()

ROI = list(cm.CFG.variance.h1b_roi)


def main() -> None:
    out = cm.out_dir("variance_with_visit")
    t = pd.read_parquet(cm.OUT / "visit_structure/recording_times_relative.parquet")
    feats = {
        "aperiodic_exponent": ("aperiodic", "exponent"),
        "aperiodic_offset": ("aperiodic", "offset"),
        "periodic_alpha": ("periodic", "alpha"),
    }
    rows = []
    for name, (fam, suf) in feats.items():
        w = pd.read_parquet(cm.BANK / f"{fam}.parquet")
        d = w[["child", "paradigm", "block"]].assign(v=w[[f"{c}_{suf}" for c in ROI]].mean(axis=1))
        d = d.merge(t[["child", "paradigm", "day", "n_days_child"]], on=["child", "paradigm"]).dropna(
            subset=["v"]
        )
        d["v"] = (d.v - d.v.mean()) / d.v.std()
        d["visit"] = d.child + "_" + d.day.astype(str)
        for subset, dd in (("all children", d), ("multi-visit children", d[d.n_days_child > 1])):
            fit = smf.mixedlm(
                "v ~ C(paradigm)", dd, groups=dd.child, re_formula="1", vc_formula={"visit": "0 + C(visit)"}
            ).fit(reml=True)
            vc_child = float(fit.cov_re.iloc[0, 0])
            vc_visit = float(fit.vcomp[0])
            res = float(fit.scale)
            tot = vc_child + vc_visit + res
            rows.append(
                {
                    "feature": name,
                    "subset": subset,
                    "n_children": dd.child.nunique(),
                    "n_visits": dd.visit.nunique(),
                    "child_share": vc_child / tot,
                    "visit_share": vc_visit / tot,
                    "residual_share": res / tot,
                    "visit_over_child": vc_visit / vc_child if vc_child else np.nan,
                }
            )
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_parquet(out / "shares.parquet")


if __name__ == "__main__":
    main()
