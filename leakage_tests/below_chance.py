"""Below-chance true-label estimates with children kept whole (Results; Additional file 1, Supplementary Methods
S5 and Table S12).

Data, children and logistic pipeline as in :mod:`leakage_tests.model_families_and_dose_response`; children kept
whole (StratifiedGroupKFold, 5 folds); child-level AUC = AUC of the mean score over a child's test blocks.

1. Conditions, each with 50 fold seeds (true labels) and 50 child-level label permutations (one fold seed each):
     base        as in the main analysis;
     resid       features residualised on age and sex inside each training fold (linear regression fitted on the
                 training blocks, applied to training and test blocks);
     matched     the sex-matched subset of Table S8 (TD matched 1:1 to autistic children on age within sex,
                 caliper 0.5 SD);
     strat_sex   folds stratified on diagnosis x sex (four strata) instead of diagnosis.
   For every run: the pooled child-level AUC, the mean within-fold child-level AUC and a fold-centred pooled AUC
   (each child's score minus the mean score of its fold's test children).
2. Between-child dispersion of child-mean log power (2,112 features, standardised over children, 30 principal
   components) per group: mean distance of each child to its group centroid, with a permutation null of the group
   difference (2,000 permutations of diagnosis between children).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from utils import leakage_common as RC
from utils.leakage_common import IA, C, generator

N = 50


def run(X, y, g, cov, seed, model_seed, condition, labels, i, strata=None):
    """One 5-fold run with children kept whole; returns pooled, within-fold and fold-centred child AUCs."""
    strat = y if strata is None else strata
    s = np.full(len(y), np.nan)
    fold_of = np.full(len(y), -1)
    within, train_share = [], []
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, strat, g)):
        Xtr, Xte = X[tr], X[te]
        if cov is not None:
            lr = LinearRegression().fit(cov[tr], Xtr)
            Xtr, Xte = Xtr - lr.predict(cov[tr]), Xte - lr.predict(cov[te])
        m = RC.logistic(model_seed).fit(Xtr, y[tr])
        s[te] = IA._score(m, Xte)
        fold_of[te] = f
        d = pd.DataFrame({"g": g[te], "y": y[te], "s": s[te]}).groupby("g").mean()
        if d.y.nunique() == 2:
            within.append(roc_auc_score(d.y, d.s))
        train_share.append(pd.Series(y[tr], index=g[tr]).groupby(level=0).first().mean())
    d = (
        pd.DataFrame({"g": g, "y": y, "s": s, "f": fold_of})
        .groupby("g")
        .agg(y=("y", "first"), s=("s", "mean"), f=("f", "first"))
    )
    d["c"] = d.s - d.groupby("f").s.transform("mean")
    test_share = d.groupby("f").y.mean()
    return {
        "condition": condition,
        "labels": labels,
        "run": i,
        "auc_pooled": float(roc_auc_score(d.y, d.s)),
        "auc_within_fold": float(np.mean(within)),
        "auc_fold_centred": float(roc_auc_score(d.y, d.c)),
        "corr_train_share_vs_fold_mean_score": float(np.corrcoef(train_share, d.groupby("f").s.mean())[0, 1]),
        "train_share_range": [float(min(train_share)), float(max(train_share))],
        "test_share_range": [float(test_share.min()), float(test_share.max())],
    }


def main():
    out = RC.OUT / "below_chance"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, y, g, _, _, meta = RC.core_asd_td()
    from classification.robustness_reruns import matched

    kids = sorted(set(g))
    sexm = set(matched(meta, kids, "age", within="sex_male", seed=C.CFG.project.seed))
    cov = np.column_stack([meta.loc[g, "age"].to_numpy(float), meta.loc[g, "sex_male"].to_numpy(float)])
    sex = meta.loc[g, "sex_male"].to_numpy(int)
    rng, seed = generator(C.SEED, "revBI_b4")
    mm = np.isin(g, list(sexm))
    sets = {
        "base": (X, y, g, None, None),
        "resid": (X, y, g, cov, None),
        "matched": (X[mm], y[mm], g[mm], None, None),
        "strat_sex": (X, y, g, None, 2 * y + sex),
    }
    jobs = []
    for cname, (Xc, yc, gc, cc, st) in sets.items():
        for i in range(N):
            jobs.append((Xc, yc, gc, cc, int(rng.integers(0, 10**5)), i, cname, "true", i, st))
        for i in range(N):
            yp = IA.permute_subject_labels(yc, gc, rng)
            stp = None if st is None else 2 * yp + st % 2  # diagnosis x sex strata on the permuted labels
            jobs.append((Xc, yp, gc, cc, int(rng.integers(0, 10**5)), i, cname, "permuted", i, stp))
    D = pd.DataFrame(Parallel(n_jobs=RC.N_JOBS, verbose=2)(delayed(run)(*j) for j in jobs))
    D.to_parquet(out / "runs.parquet")

    summ = {
        "seed": seed,
        "n": N,
        "n_children": {c: int(len(set(v[2]))) for c, v in sets.items()},
        "matched_groups": meta.loc[sorted(sexm), "group"].value_counts().to_dict(),
        "matched_male": int(meta.loc[sorted(sexm), "sex_male"].sum()),
        "conditions": {},
    }
    for c in sets:
        r = {}
        for met in ("auc_pooled", "auc_within_fold", "auc_fold_centred"):
            t = D[(D.condition == c) & (D.labels == "true")][met]
            p = D[(D.condition == c) & (D.labels == "permuted")][met]
            r[met] = {
                "true_mean_sd": [float(t.mean()), float(t.std())],
                "true_band": RC.band(t),
                "perm_band": RC.band(p),
                "true_mean_minus_perm_median": float(t.mean() - p.median()),
                "mannwhitney_p": float(__import__("scipy").stats.mannwhitneyu(t, p).pvalue),
            }
        r["corr_train_share_vs_fold_mean_score_true_median"] = float(
            D[(D.condition == c) & (D.labels == "true")].corr_train_share_vs_fold_mean_score.median()
        )
        summ["conditions"][c] = r
    summ["minutes"] = (time.time() - t0) / 60
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


def dispersion(Z, lab):
    """Mean distance of each child to its group centroid; returns (autistic - TD, per group)."""
    out = {}
    for v in (0, 1):
        z = Z[lab == v]
        out[v] = float(np.linalg.norm(z - z.mean(0), axis=1).mean())
    return out[1] - out[0], out


def between_child_dispersion():
    """Between-child dispersion of child-mean log power by group, with a 2,000-permutation null (Table S12)."""
    out = RC.OUT / "below_chance"
    out.mkdir(parents=True, exist_ok=True)
    X, y, g, _, _, _ = RC.core_asd_td()
    d = pd.DataFrame(X).assign(g=g, y=y).groupby("g").mean()
    lab = d.pop("y").to_numpy(int)
    Z = PCA(30, random_state=0).fit_transform(StandardScaler().fit_transform(d.to_numpy()))
    obs, per = dispersion(Z, lab)
    rng, seed = generator(C.SEED, "revBI_b4b")
    null = np.array([dispersion(Z, rng.permutation(lab))[0] for _ in range(2000)])
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))
    ratio = Z[lab == 1].var(0, ddof=1) / Z[lab == 0].var(0, ddof=1)
    res = {
        "seed": seed,
        "n_children": int(len(lab)),
        "mean_distance_to_centroid": {"ASD": per[1], "TD": per[0]},
        "difference_asd_minus_td": float(obs),
        "perm_p_two_sided": p,
        "variance_ratio_asd_td_first10_pcs": [float(r) for r in ratio[:10]],
        "n_pcs_ratio_gt1_of_30": int((ratio > 1).sum()),
    }
    (out / "dispersion.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
    between_child_dispersion()
