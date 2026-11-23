"""Model families and the dose-response test (Methods, Tests of identity leakage; Fig. 3b, c; Table 2;
Additional file 1, Supplementary Methods S5 and Table S10).

Data: block-level log power (64 channels x 33 1-Hz bins, six paradigms) of the 70 autistic and TD core children
(3,052 blocks); diagnosis is the label of every block of a child. The analyses use :mod:`identity_audit`.

0. Reproduction check. ``identity_audit.split_contrast`` with the pipeline and seeds of
   :mod:`leakage_tests.split_contrast` must reproduce its 10 x 2 AUCs exactly (to 1e-12); otherwise the script
   stops.
1. Model families. Four fixed pipelines (no tuning), each with children kept whole and with blocks split:
     logistic   standardise, PCA 30, L2 logistic regression (C = 1)
     svm-rbf    standardise, PCA 30, RBF support vector machine (C = 1, gamma "scale"; decision function)
     forest     random forest on all 2,112 features (300 trees, sqrt features, min leaf 1)
     knn        standardise, PCA 30, 10 nearest neighbours (Euclidean)
   5 repeats x 5 folds with the true labels and 20 child-level label permutations x 1 repeat per scheme (the same
   permutations for every model). Child-level AUC = AUC of the mean score over a child's test blocks.
2. Dose-response (logistic pipeline). Children kept whole (5 folds); k = 0, 1, 2, 4, 8, 16 blocks of every test
   child added to training; child-level AUC on the remaining blocks (the same blocks for every k). 5 repeats with
   the true labels and the same 20 child-level label permutations.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from identity_audit import identity_audit as IA
from stats.rng import generator
from utils import common as C
from utils.inputs import Inputs

N_JOBS = 24
REPEATS = 5
N_PERM = 20
KS = (0, 1, 2, 4, 8, 16)


def logistic(seed):
    return make_pipeline(StandardScaler(), PCA(30, random_state=seed), LogisticRegression(C=1.0, max_iter=3000))


def svm_rbf(seed):
    return make_pipeline(StandardScaler(), PCA(30, random_state=seed), SVC(C=1.0, gamma="scale"))


def forest(seed):
    return RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=1, random_state=seed)


def knn(seed):
    return make_pipeline(StandardScaler(), PCA(30, random_state=seed), KNeighborsClassifier(10))


MODELS = {"logistic": logistic, "svm-rbf": svm_rbf, "forest": forest, "knn": knn}


def data():
    inp = Inputs(C.CFG)
    meta, core = inp.meta(), inp.core()
    kids = [k for k in core if meta.loc[k, "group"] in ("ASD", "TD")]
    w = pd.read_parquet(C.BANK / "spectral_logpow.parquet")
    w = w[w.child.isin(kids)].reset_index(drop=True)
    feats = [c for c in w.columns if c not in ("child", "paradigm", "block")]
    X = w[feats].to_numpy(float)
    y = (meta.loc[w.child, "group"] == "ASD").to_numpy(int)
    return X, y, w.child.to_numpy()


def contrast_job(X, y, g, model, scheme, repeats, seed, label, run):
    R = IA.split_contrast(X, y, g, MODELS[model], repeats, seed, (scheme,))
    return R.assign(model=model, labels=label, run=run)


def dose_job(X, y, g, seed, label, run):
    return IA.dose_response(X, y, g, logistic, KS, seed).assign(labels=label, run=run)


def main() -> None:
    out = C.out_dir("model_families_dose_response")
    t0 = time.time()
    X, y, g = data()
    split_seed = C.SEED % 100000  # fold seed of leakage_tests.split_contrast

    # 0. reproduction of leakage_tests.split_contrast with the module
    ref = pd.read_parquet(C.OUT / "split_contrast" / "auc.parquet")
    pairs = (("by subject", "by child"), ("by block", "by block (leaky)"))
    runs = Parallel(n_jobs=N_JOBS)(
        delayed(IA.split_contrast)(X, y, g, logistic, [r], split_seed, (sc,)) for sc, _ in pairs for r in range(10)
    )
    R_all = pd.concat(runs, ignore_index=True)
    rep = []
    for sc, ref_sc in pairs:
        R = R_all[R_all.scheme == sc].sort_values("repeat")
        r0 = ref[ref.split == ref_sc].sort_values("repeat")
        d_block = np.abs(R.auc_block.to_numpy() - r0.auc_block.to_numpy()).max()
        d_sub = np.abs(R.auc_subject.to_numpy() - r0.auc_child.to_numpy()).max()
        rep.append({"scheme": sc, "max_abs_diff_block": float(d_block), "max_abs_diff_subject": float(d_sub)})
    ok = all(r["max_abs_diff_block"] < 1e-12 and r["max_abs_diff_subject"] < 1e-12 for r in rep)
    C.write_json(out / "reproduce_split_contrast.json", {"reproduced": ok, "schemes": rep})
    print("split contrast reproduced by identity_audit:", ok, rep, flush=True)
    if not ok:
        raise SystemExit("identity_audit does not reproduce the split contrast; stopping")

    rng, seed = generator(C.SEED, "v3_b8")
    perms = []
    for i in range(N_PERM):
        perms.append((IA.permute_subject_labels(y, g, rng), int(rng.integers(0, 10**5)), i))
    true_seed = int(rng.integers(0, 10**5))

    # 1. model families
    jobs = []
    for m in MODELS:
        for sc in IA.SCHEMES:
            jobs.append((X, y, g, m, sc, REPEATS, true_seed, "true", -1))
            jobs += [(X, yp, g, m, sc, 1, s, "permuted", i) for yp, s, i in perms]
    M = pd.concat(Parallel(n_jobs=N_JOBS, verbose=2)(delayed(contrast_job)(*j) for j in jobs), ignore_index=True)
    M.to_parquet(out / "models.parquet")
    print(f"models done {time.time() - t0:.0f} s", flush=True)

    # 2. dose-response
    jobs = [(X, y, g, true_seed + r, "true", r) for r in range(REPEATS)]
    jobs += [(X, yp, g, s, "permuted", i) for yp, s, i in perms]
    D = pd.concat(Parallel(n_jobs=N_JOBS, verbose=2)(delayed(dose_job)(*j) for j in jobs), ignore_index=True)
    D.to_parquet(out / "dose.parquet")

    def band(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    summ = {"seed": seed, "true_seed": true_seed, "n_perm": N_PERM, "repeats": REPEATS, "ks": list(KS)}
    summ["models"] = {
        m: {
            sc: {
                "true_auc_subject_mean": float(d[d.labels == "true"].auc_subject.mean()),
                "true_auc_subject_range": [
                    float(d[d.labels == "true"].auc_subject.min()),
                    float(d[d.labels == "true"].auc_subject.max()),
                ],
                "true_auc_block_mean": float(d[d.labels == "true"].auc_block.mean()),
                "perm_auc_subject_mean": float(d[d.labels == "permuted"].auc_subject.mean()),
                "perm_auc_subject_band": band(d[d.labels == "permuted"].auc_subject),
            }
            for sc, d in M[M.model == m].groupby("scheme")
        }
        for m in MODELS
    }
    summ["dose"] = {
        lab: {
            int(k): {"auc_subject_mean": float(d.auc_subject.mean()), "auc_subject_band": band(d.auc_subject)}
            for k, d in D[D.labels == lab].groupby("k")
        }
        for lab in ("true", "permuted")
    }
    summ["minutes"] = (time.time() - t0) / 60
    C.write_json(out / "summary.json", summ)
    print(pd.DataFrame(summ["models"]).T, flush=True)
    print(D.groupby(["labels", "k"]).auc_subject.describe().round(3), flush=True)


if __name__ == "__main__":
    main()
