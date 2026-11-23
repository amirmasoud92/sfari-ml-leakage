"""Cross-visit leak (Methods, Tests of identity leakage; Additional file 1, Supplementary Methods S5 and Fig. S3b).

Visits are the relative day index of each recording (:mod:`identification.visit_structure`; no dates). Children
kept whole in 5 folds; for each test child with two visits, all blocks of its smaller visit are scored (at least 4
blocks) and the leaked pool (k = 0, 1, 2, 4, 8, 16) comes only from its other visit (any paradigm). Children with one
visit stay in the folds and are scored without leaks but are excluded from the AUCs. Logistic pipeline;
labels permuted between children, 20 permutations, plus 5 true-label repeats in the core analysis.

A. Core analysis: the 70 core autistic and TD children (13 with two visits). The block-level AUC over the scored
   blocks is reported next to the child-level AUC.
B. All children in the feature bank (129), with balanced binary labels permuted between children; AUCs over the 28
   children with two visits.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import leakage_common as RC
from utils.leakage_common import IA, C, generator

N_PERM = 20


def job(X, y, g, day, seed, labels, run, multi, analysis):
    return RC.dose_general(
        X, y, g, RC.logistic, RC.KS, seed, conditions=day, choose="smaller", score_subjects=multi
    ).assign(labels=labels, run=run, analysis=analysis)


def multi_visit(g, day):
    d = pd.DataFrame({"g": g, "day": day}).dropna()
    n = d.groupby("g").day.nunique()
    return sorted(n[n > 1].index)


def main():
    out = RC.OUT / "cross_visit_leak"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng, seed = generator(C.SEED, "revBI_b2")

    # A. core 70
    X, y, g, _, day, meta = RC.core_asd_td()
    if np.isnan(day.astype(float)).any():
        raise SystemExit("blocks without a visit index")
    mA = multi_visit(g, day)
    jobs = [(X, y, g, day, int(rng.integers(0, 10**5)) + r, "true", r, mA, "core70") for r in range(5)]
    for i in range(N_PERM):
        jobs.append(
            (
                X,
                IA.permute_subject_labels(y, g, rng),
                g,
                day,
                int(rng.integers(0, 10**5)),
                "permuted",
                i,
                mA,
                "core70",
            )
        )

    # B. all children in the bank, balanced labels permuted between children
    w = pd.read_parquet(C.BANK / "spectral_logpow.parquet", columns=["child"])
    allk = sorted(w.child.unique())
    XB, gB, _, dayB = RC.blocks(allk)
    keep = ~pd.isna(dayB)
    XB, gB, dayB = XB[keep], gB[keep], dayB[keep]
    kidsB = np.unique(gB)
    mB = multi_visit(gB, dayB)
    base = np.zeros(len(kidsB), int)
    base[: len(kidsB) // 2] = 1
    for i in range(N_PERM):
        lab = dict(zip(kidsB, rng.permutation(base), strict=True))
        yB = np.array([lab[k] for k in gB])
        jobs.append((XB, yB, gB, dayB, int(rng.integers(0, 10**5)), "permuted", i, mB, "all children"))

    D = pd.concat(Parallel(n_jobs=RC.N_JOBS, verbose=2)(delayed(job)(*j) for j in jobs), ignore_index=True)
    D.to_parquet(out / "dose.parquet")

    nb = pd.DataFrame({"g": g, "day": day}).query("g in @mA").groupby(["g", "day"]).size().unstack()
    summ = {
        "seed": seed,
        "n_perm": N_PERM,
        "ks": list(RC.KS),
        "core70": {
            "n_children": int(len(np.unique(g))),
            "n_two_visits": len(mA),
            "two_visit_groups": meta.loc[mA, "group"].value_counts().to_dict(),
            "blocks_scored_visit_median_range": [
                float(nb.min(axis=1).median()),
                int(nb.min(axis=1).min()),
                int(nb.min(axis=1).max()),
            ],
            "blocks_other_visit_median_range": [
                float(nb.max(axis=1).median()),
                int(nb.max(axis=1).min()),
                int(nb.max(axis=1).max()),
            ],
        },
        "all_children": {"n_children": int(len(kidsB)), "n_two_visits": len(mB), "n_blocks": int(len(gB))},
        "curves": {},
        "gain_16": {},
    }
    for an in ("core70", "all children"):
        for lab in ("permuted", "true"):
            d = D[(D.analysis == an) & (D.labels == lab)]
            if d.empty:
                continue
            f = RC.band if lab == "permuted" else RC.mean_range
            for met in ("auc_subject_scored", "auc_block_scored"):
                summ["curves"][f"{an} | {lab} | {met}"] = {int(k): f(d[d.k == k][met]) for k in RC.KS}
                wv = d.pivot(index="run", columns="k", values=met)
                gain = (wv[16] - wv[0]).dropna().to_numpy()
                summ["gain_16"][f"{an} | {lab} | {met}"] = {
                    "summary": f(gain),
                    "range": [float(gain.min()), float(gain.max())],
                    "n_positive": int((gain > 0).sum()),
                    "n": int(len(gain)),
                }
    summ["minutes"] = (time.time() - t0) / 60
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
