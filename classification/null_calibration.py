"""Null calibration of the cross-validated AUC estimators (Methods; Fig. 4; Additional file 1, Supplementary
Methods S3 and Table S16).

For rung (e) (all EEG features) and rung (a) (covariates) on the 70 autistic and TD core children, diagnosis is
permuted 200 times (keeping the 35/35 split), and for every permutation the complete 20 x 5 nested cross-validation
is rerun with folds re-stratified on the permuted labels, with the ladder's own unsupervised step (fold PCA) and
supervised step (inner-CV grid). For each null dataset: pooled AUC with its child-bootstrap 95% CI (500 resamples),
mean per-repeat AUC and mean within-fold AUC.

Reported: the null mean and SD of each estimator, the coverage of 0.5 by the child-bootstrap CI, the permutation
p-value of the observed value, and the null-calibrated interval observed +- 1.96 null SD.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from classification.model_ladder import ModelLadder, Target
from stats.rng import generator
from utils import common as C
from utils.inputs import Inputs

N_NULL, N_BOOT = 200, 500


def stats_for(yv, P, folds, pos, rng):
    m = np.nanmean(P, axis=0)
    pooled = roc_auc_score(yv, m)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[i])) == 2:
            bs.append(roc_auc_score(yv[i], m[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    per_rep = np.mean([roc_auc_score(yv, P[r]) for r in range(P.shape[0])])
    fw = [
        roc_auc_score(yv[[pos[k] for k in f.test]], P[f.repeat, [pos[k] for k in f.test]])
        for f in folds
        if len(np.unique(yv[[pos[k] for k in f.test]])) == 2
    ]
    return {
        "pooled": pooled,
        "boot_lo": lo,
        "boot_hi": hi,
        "mean_of_repeats": per_rep,
        "within_fold": np.mean(fw),
    }


def one(seed: int, kids, groups, X, meta, cfg) -> list[dict]:
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(seed)
    with threadpool_limits(1):
        perm = rng.permutation(groups)
        y = pd.Series((perm == "ASD").astype(int), index=kids)
        t = Target("asd_vs_td", y, "classification", pd.Series(perm, index=kids))
        c1 = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"n_jobs": 1})})
        lad = ModelLadder(c1, X, meta, use_mofa=False)
        folds = lad._folds(t, y, c1.prediction.repeats)
        pos = {k: i for i, k in enumerate(kids)}
        P = lad.cv_predictions(t, y, c1.prediction.repeats, ["e_all", "a_covariates"], parallel=False)
        return [{"seed": seed, "rung": r, **stats_for(y.to_numpy(), P[r], folds, pos, rng)} for r in P]


def main() -> None:
    t0 = time.time()
    out = C.out_dir("null_calibration")
    inp = Inputs(C.CFG)
    core, meta = inp.core(), inp.meta()
    kids = [k for k in core if meta.loc[k, "group"] in ("ASD", "TD")]
    X = inp.child_matrix(core).loc[kids]
    groups = meta.loc[kids, "group"].to_numpy()
    rng, seed = generator(C.SEED, "f04")
    seeds = rng.integers(0, 2**31, N_NULL)
    res = Parallel(n_jobs=30)(delayed(one)(int(s), kids, groups, X, meta, C.CFG) for s in seeds)
    R = pd.DataFrame([r for rr in res for r in rr])
    R.to_parquet(out / "null_runs.parquet")
    obs = pd.read_parquet(C.OUT / "ladder_estimators/rungs.parquet")
    obs = obs[obs.target == "asd_vs_td"].set_index("rung")
    rows = []
    for rung, d in R.groupby("rung"):
        o = obs.loc[rung]
        row = {
            "rung": rung,
            "n_null": len(d),
            "coverage_boot_ci": float(((d.boot_lo <= 0.5) & (d.boot_hi >= 0.5)).mean()),
            "boot_ci_width_mean": float((d.boot_hi - d.boot_lo).mean()),
        }
        for est, ocol in (
            ("pooled", "pooled"),
            ("mean_of_repeats", "mean_of_repeats"),
            ("within_fold", "within_fold_mean"),
        ):
            row[f"null_mean_{est}"] = float(d[est].mean())
            row[f"null_sd_{est}"] = float(d[est].std(ddof=1))
            row[f"observed_{est}"] = float(o[ocol])
            row[f"p_null_{est}"] = float((1 + (d[est] >= o[ocol]).sum()) / (1 + len(d)))
            row[f"calibrated_ci_{est}"] = [
                float(o[ocol] - 1.96 * d[est].std(ddof=1)),
                float(o[ocol] + 1.96 * d[est].std(ddof=1)),
            ]
        rows.append(row)
    S = pd.DataFrame(rows)
    S.to_json(out / "calibration.json", orient="records", indent=1)
    print(S.T.to_string())
    C.write_json(
        out / "summary.json",
        {"seed": seed, "seconds": time.time() - t0, "n_null": N_NULL, "n_boot": N_BOOT},
    )


if __name__ == "__main__":
    main()
