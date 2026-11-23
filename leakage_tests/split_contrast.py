"""Split contrast for the conventional pipeline (Methods, Classification and evaluation schemes; Fig. 1, card 4;
Additional file 1, Table S9a).

Block-level log power (64 channels x 33 1-Hz bins, all six paradigms) of the 70 autistic and TD core children,
standardised, PCA to 30 components and L2 logistic regression (C = 1; fixed, no tuning), all fitted on training
blocks only. Two cross-validation schemes, 10 repeats x 5 folds each:

    children kept whole: StratifiedGroupKFold (no child in both training and test),
    blocks split: StratifiedKFold over blocks (blocks of the same child in training and test; leaky).

AUC is reported at block level and at child level (mean predicted probability over a child's test blocks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import common as C
from utils.inputs import Inputs


def run(X, y, groups, splitter, repeats=10):
    rows = []
    for r in range(repeats):
        sp = splitter(r)
        split = sp.split(X, y, groups) if isinstance(sp, StratifiedGroupKFold) else sp.split(X, y)
        p = np.full(len(y), np.nan)
        for tr, te in split:
            m = make_pipeline(
                StandardScaler(), PCA(30, random_state=r), LogisticRegression(C=1.0, max_iter=3000)
            )
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        d = pd.DataFrame({"g": groups, "y": y, "p": p})
        ch = d.groupby("g").agg(y=("y", "first"), p=("p", "mean"))
        rows.append({"repeat": r, "auc_block": roc_auc_score(y, p), "auc_child": roc_auc_score(ch.y, ch.p)})
    return pd.DataFrame(rows)


def main() -> None:
    out = C.out_dir("split_contrast")
    inp = Inputs(C.CFG)
    meta, core = inp.meta(), inp.core()
    kids = [k for k in core if meta.loc[k, "group"] in ("ASD", "TD")]
    w = pd.read_parquet(C.BANK / "spectral_logpow.parquet")
    w = w[w.child.isin(kids)].reset_index(drop=True)
    feats = [c for c in w.columns if c not in ("child", "paradigm", "block")]
    X = w[feats].to_numpy(float)
    y = (meta.loc[w.child, "group"] == "ASD").to_numpy(int)
    g = w.child.to_numpy()
    seed = C.SEED % 100000
    by_child = run(X, y, g, lambda r: StratifiedGroupKFold(5, shuffle=True, random_state=seed + r))
    by_block = run(X, y, g, lambda r: StratifiedKFold(5, shuffle=True, random_state=seed + r))
    R = pd.concat([by_child.assign(split="by child"), by_block.assign(split="by block (leaky)")])
    R.to_parquet(out / "auc.parquet")
    # identification of the child from a held-out block (block split): accuracy among 70 children
    acc = []
    labels = pd.factorize(g)[0]
    for r in range(3):
        p = np.empty(len(y), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, labels):
            m = make_pipeline(
                StandardScaler(), PCA(30, random_state=r), LogisticRegression(C=1.0, max_iter=3000)
            )
            m.fit(X[tr], labels[tr])
            p[te] = m.predict(X[te])
        acc.append(float((p == labels).mean()))
    S = R.groupby("split")[["auc_block", "auc_child"]].agg(["mean", "std"])
    C.write_json(
        out / "summary.json",
        {
            "n_blocks": int(len(y)),
            "n_children": len(kids),
            "n_features": len(feats),
            "child_identification_accuracy": float(np.mean(acc)),
            "chance_identification": 1 / len(kids),
            "auc": {
                s: {k: float(v) for k, v in R[R.split == s][["auc_block", "auc_child"]].mean().items()}
                for s in R.split.unique()
            },
        },
    )
    print(S.round(3))
    print("child identification from one block:", round(np.mean(acc), 3), "chance", round(1 / len(kids), 3))


if __name__ == "__main__":
    main()
