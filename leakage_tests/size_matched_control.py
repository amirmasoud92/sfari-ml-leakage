"""Size-matched control of the dose-response test (Methods, Tests of identity leakage; Additional file 1,
Supplementary Methods S5 and Fig. S3a).

Same data, children, logistic pipeline, folds, scored blocks, leaked pools, 20 child-level label permutations and
5 true-label repeats as the dose-response test of :mod:`leakage_tests.model_families_and_dose_response`.

0. Reproduction: :func:`utils.leakage_common.dose_general` without a reserve must reproduce the stored dose-response
   AUCs exactly (all 20 + 5 runs, k = 0-16, to 1e-12); the script stops otherwise.
1. Before training, 4 blocks of every training child are set aside (56 training children x 4 = 224 blocks = 14 test
   children x 16 leaked blocks, the largest number added at k = 16), so the baseline at k = 0 excludes them. At
   k = 0, 1, 2, 4, 8, 16 two conditions add the same number of blocks to this baseline: leaky (k blocks of every
   test child) and size-matched (the same total taken from the reserves of training children). Scored blocks and
   baseline are shared, so only the source of the added blocks differs.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import leakage_common as RC
from utils.leakage_common import C, generator

RESERVE = 4


def job(X, y, g, seed, labels, run, reserve, rseed):
    return RC.dose_general(X, y, g, RC.logistic, RC.KS, seed, reserve=reserve, reserve_seed=rseed).assign(
        labels=labels, run=run
    )


def main():
    out = RC.OUT / "size_matched_control"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, y, g, _, _, _ = RC.core_asd_td()
    perms, true_seed, _ = RC.dose_permutations(y, g)
    runs = [(y, true_seed + r, "true", r) for r in range(5)] + [(yp, s, "permuted", i) for yp, s, i in perms]

    # 0. reproduction of the dose-response test of leakage_tests.model_families_and_dose_response
    REF = pd.read_parquet(RC.OUT / "model_families_dose_response" / "dose.parquet")
    R0 = pd.concat(
        Parallel(n_jobs=RC.N_JOBS)(delayed(job)(X, yy, g, s, lab, r, 0, 0) for yy, s, lab, r in runs),
        ignore_index=True,
    )
    m = R0.merge(REF, on=["labels", "run", "k"], suffixes=("", "_ref"))
    diff = float(np.max(np.abs(m.auc_subject - m.auc_subject_ref))) if len(m) == len(REF) else np.inf
    print("reproduction max |diff|", diff, len(m), len(REF), flush=True)
    if not diff < 1e-12:
        raise SystemExit("dose_general does not reproduce the dose-response test; stopping")

    # 1. size-matched control
    rng, rseed0 = generator(C.SEED, "revBI_b1")
    rseeds = [int(rng.integers(0, 10**9)) for _ in runs]
    D = pd.concat(
        Parallel(n_jobs=RC.N_JOBS, verbose=2)(
            delayed(job)(X, yy, g, s, lab, r, RESERVE, rs)
            for (yy, s, lab, r), rs in zip(runs, rseeds, strict=True)
        ),
        ignore_index=True,
    )
    D.to_parquet(out / "dose.parquet")

    summ = {
        "reserve_per_training_child": RESERVE,
        "seed": rseed0,
        "reproduction_max_abs_diff": diff,
        "n_perm": 20,
        "repeats": 5,
        "ks": list(RC.KS),
        "curves": {},
        "gain_16": {},
    }
    for lab in ("permuted", "true"):
        f = RC.band if lab == "permuted" else RC.mean_range
        for arm in ("leaky", "matched"):
            d = D[(D.labels == lab) & (D.arm == arm)]
            summ["curves"][f"{lab} | {arm}"] = {int(k): f(d[d.k == k].auc_subject) for k in RC.KS}
            w = d.pivot(index="run", columns="k", values="auc_subject")
            gain = (w[16] - w[0]).to_numpy()
            summ["gain_16"][f"{lab} | {arm}"] = {
                "summary": f(gain) if lab == "true" else RC.band(gain),
                "range": [float(gain.min()), float(gain.max())],
                "n_positive": int((gain > 0).sum()),
                "n": len(gain),
            }
        r0 = REF[REF.labels == lab]
        summ["curves"][f"{lab} | dose-response (full baseline)"] = {int(k): f(r0[r0.k == k].auc_subject) for k in RC.KS}
        # matched minus leaky at k = 16, paired by run
        p = D[(D.labels == lab) & (D.k == 16)].pivot(index="run", columns="arm", values="auc_subject")
        dl = (p["leaky"] - p["matched"]).to_numpy()
        summ["gain_16"][f"{lab} | leaky minus matched at k=16"] = {
            "summary": RC.band(dl),
            "range": [float(dl.min()), float(dl.max())],
            "n_positive": int((dl > 0).sum()),
            "n": len(dl),
        }
    summ["n_added_k16_total_over_folds"] = int(D[(D.k == 16)].n_added.iloc[0])
    summ["minutes"] = (time.time() - t0) / 60
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
