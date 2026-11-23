"""Permuted-label control and block identification (Methods, Tests of identity leakage; Fig. 1, cards 3 and 5).

1. Permuted labels. Diagnosis is permuted between children (every block of a child gets the child's permuted
   label), and the pipeline of :mod:`leakage_tests.split_contrast` is rerun with blocks split and with children
   kept whole; 40 permutations x 2 repeats per scheme. A leaky AUC that stays high with labels that carry no
   diagnostic information measures recognition of children.
2. Block identification. The same pipeline is trained to say which of the 70 children produced a held-out block
   (block split, 3,052 blocks), with a child-bootstrap interval (2,000 resamples of children) because blocks are
   nested in children.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from leakage_tests.split_contrast import run
from stats.rng import generator
from utils import common as C
from utils.inputs import Inputs

N_PERM = 40


def one_perm(X, g, kids, ylab_child, scheme, seed):
    y = np.array([ylab_child[c] for c in g], int)
    if scheme == "by child":
        sp = lambda r: StratifiedGroupKFold(5, shuffle=True, random_state=seed + r)  # noqa: E731
    else:
        sp = lambda r: StratifiedKFold(5, shuffle=True, random_state=seed + r)  # noqa: E731
    R = run(X, y, g, sp, repeats=2)
    return {"scheme": scheme, "seed": seed, "auc_block": R.auc_block.mean(), "auc_child": R.auc_child.mean()}


def main() -> None:
    out = C.out_dir("permuted_labels")
    inp = Inputs(C.CFG)
    meta, core = inp.meta(), inp.core()
    kids = [k for k in core if meta.loc[k, "group"] in ("ASD", "TD")]
    w = pd.read_parquet(C.BANK / "spectral_logpow.parquet")
    w = w[w.child.isin(kids)].reset_index(drop=True)
    feats = [c for c in w.columns if c not in ("child", "paradigm", "block")]
    X = w[feats].to_numpy(float)
    g = w.child.to_numpy()
    lab = (meta.loc[kids, "group"] == "ASD").astype(int).to_numpy()
    rng, seed = generator(C.SEED, "v2_b7")
    jobs = []
    for _ in range(N_PERM):
        perm = dict(zip(kids, rng.permutation(lab), strict=True))
        s = int(rng.integers(0, 10**5))
        jobs += [(perm, "by child", s), (perm, "by block (leaky)", s)]
    rows = Parallel(n_jobs=20)(delayed(one_perm)(X, g, kids, p, sc, s) for p, sc, s in jobs)
    P = pd.DataFrame(rows)
    P.to_parquet(out / "permuted_labels.parquet")
    print(P.groupby("scheme")[["auc_block", "auc_child"]].describe().round(3).T, flush=True)
    # identification from one block with a child bootstrap
    labels = pd.factorize(g)[0]
    pred = np.empty(len(g), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed % 10**5).split(X, labels):
        m = make_pipeline(StandardScaler(), PCA(30, random_state=0), LogisticRegression(C=1.0, max_iter=3000))
        m.fit(X[tr], labels[tr])
        pred[te] = m.predict(X[te])
    per_child = pd.Series(pred == labels).groupby(g).agg(["sum", "count"])
    sums, counts = per_child["sum"].to_numpy(float), per_child["count"].to_numpy(float)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(sums), len(sums))
        bs.append(sums[i].sum() / counts[i].sum())
    acc = float(sums.sum() / counts.sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    res = {
        "n_perm": N_PERM,
        "identification_accuracy": acc,
        "identification_ci_child_bootstrap": [float(lo), float(hi)],
        "n_blocks": int(len(g)),
        "n_children": len(kids),
        "children_all_blocks_identified": int((sums == counts).sum()),
        "seed": seed,
    }
    for sc, d in P.groupby("scheme"):
        res[sc] = {
            "auc_child_mean": float(d.auc_child.mean()),
            "auc_child_2.5": float(np.percentile(d.auc_child, 2.5)),
            "auc_child_97.5": float(np.percentile(d.auc_child, 97.5)),
            "auc_block_mean": float(d.auc_block.mean()),
        }
    C.write_json(out / "summary.json", res)
    print(res)


if __name__ == "__main__":
    main()
