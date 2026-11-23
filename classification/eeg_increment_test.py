"""Increment of EEG over covariates (Methods; Results; Additional file 1, Supplementary Methods S3).

On the stored out-of-fold predictions of :mod:`classification.covariate_baselines` (core, medication-free and male
children), diagnosis is regressed by logistic regression on logit(p_covariates) with and without
logit(p_all EEG features), and the two models are compared by a likelihood-ratio test. Unlike a single penalised
model of all features and covariates, this test is not affected by the dilution of a few covariates among thousands
of EEG features under one penalty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from utils import common as C


def main() -> None:
    out = C.out_dir("eeg_increment")
    o = pd.read_parquet(C.OUT / "covariate_baselines" / "oof.parquet")
    inc = []
    for s in ("core", "medication_free", "male"):
        d = o[o.set == s]
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
        y = d.y.to_numpy()
        base = "a_covariates[all_covariates]" if "a_covariates[all_covariates]" in d else "a_covariates[no_medication]"
        Xa = sm.add_constant(lg(d[base].to_numpy()))
        Xb = sm.add_constant(np.column_stack([lg(d[base].to_numpy()), lg(d["e_all"].to_numpy())]))
        fa, fb = sm.Logit(y, Xa).fit(disp=0), sm.Logit(y, Xb).fit(disp=0)
        lr = 2 * (fb.llf - fa.llf)
        inc.append(
            {
                "set": s,
                "n": len(y),
                "baseline": base,
                "coef_eeg": fb.params[2],
                "se_eeg": fb.bse[2],
                "lr_stat": lr,
                "p_lr": stats.chi2.sf(lr, 1),
            }
        )
    I = pd.DataFrame(inc)
    I.to_parquet(out / "incremental_value.parquet")
    print(I.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
