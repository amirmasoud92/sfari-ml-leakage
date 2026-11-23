"""Dependence of the leak on the number of children (Methods, Other classifiers; Additional file 1,
Supplementary Methods S5 and Fig. S3c).

Balanced subsamples of N = 20, 40 and 70 core children (N/2 autistic, N/2 TD), 20 draws each (at N = 70 every draw
is the full set). In each draw diagnosis is permuted between the drawn children and the logistic pipeline is run
once with blocks split and once with children kept whole; child-level AUC.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import leakage_common as RC
from utils.leakage_common import IA, C, generator

NS = (20, 40, 70)
DRAWS = 20


def job(X, y, g, kids, fold_seed, n, draw):
    m = np.isin(g, kids)
    R = IA.split_contrast(X[m], y[m], g[m], RC.logistic, [0], fold_seed)
    return R.assign(n=n, draw=draw)


def main():
    out = RC.OUT / "number_of_children"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, y, g, _, _, meta = RC.core_asd_td()
    asd = sorted(set(g[y == 1]))
    td = sorted(set(g[y == 0]))
    rng, seed = generator(C.SEED, "revBI_b5")
    jobs = []
    for n in NS:
        for d in range(DRAWS):
            kids = list(rng.choice(asd, n // 2, replace=False)) + list(rng.choice(td, n // 2, replace=False))
            m = np.isin(g, kids)
            yp = y.copy()
            yp[m] = IA.permute_subject_labels(y[m], g[m], rng)
            jobs.append((X, yp, g, kids, int(rng.integers(0, 10**5)), n, d))
    D = pd.concat(Parallel(n_jobs=RC.N_JOBS, verbose=2)(delayed(job)(*j) for j in jobs), ignore_index=True)
    D.to_parquet(out / "runs.parquet")
    summ = {"seed": seed, "draws": DRAWS, "ns": list(NS), "permuted": {}}
    for n in NS:
        for sc in ("by block", "by subject"):
            summ["permuted"][f"N={n} | {sc}"] = RC.band(D[(D.n == n) & (D.scheme == sc)].auc_subject)
    summ["minutes"] = (time.time() - t0) / 60
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
