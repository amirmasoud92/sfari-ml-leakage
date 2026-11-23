"""Positive controls and the cross-paradigm leak (Methods, Tests of identity leakage; Fig. 3b, e; Additional file 1,
Supplementary Methods S5 and Table S11).

Data and pipeline (standardise, PCA 30, L2 logistic regression) as in
:mod:`leakage_tests.model_families_and_dose_response`: block-level log power of the 70 autistic and TD core children.

1. Positive controls: labels that EEG should carry, age above versus at or below the core-set median, and sex. For
   each label, the split contrast (children kept whole versus blocks split; 5 repeats) with 20 child-level
   permutations of that label, and the dose-response test (k = 0, 1, 2, 4, 8, 16; 5 repeats, 20 permutations).
2. Cross-paradigm leak: diagnosis, true and permuted labels, but the scored blocks of each test child are all blocks
   of one randomly chosen paradigm with at least four blocks, and leaked blocks come only from the child's other
   paradigms (``identity_audit.dose_response(..., conditions=paradigm)``).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from identity_audit import identity_audit as IA
from leakage_tests.model_families_and_dose_response import KS, logistic
from stats.rng import generator
from utils import common as C
from utils.inputs import Inputs

N_JOBS = 24
REPEATS = 5
N_PERM = 20


def data():
    inp = Inputs(C.CFG)
    meta, core = inp.meta(), inp.core()
    kids = [k for k in core if meta.loc[k, "group"] in ("ASD", "TD")]
    w = pd.read_parquet(C.BANK / "spectral_logpow.parquet")
    w = w[w.child.isin(kids)].reset_index(drop=True)
    feats = [c for c in w.columns if c not in ("child", "paradigm", "block")]
    m = meta.loc[kids]
    labels = {
        "diagnosis": (m.group == "ASD").astype(int),
        "age above median": (m.age > m.age.median()).astype(int),
        "sex (male)": m.sex_male.astype(int),
    }
    Y = {k: v.loc[w.child].to_numpy() for k, v in labels.items()}
    return w[feats].to_numpy(float), Y, w.child.to_numpy(), w.paradigm.to_numpy(), m


def contrast_job(X, y, g, scheme, repeats, seed, target, labels, run):
    return IA.split_contrast(X, y, g, logistic, repeats, seed, (scheme,)).assign(target=target, labels=labels, run=run)


def dose_job(X, y, g, seed, target, labels, run, cond):
    return IA.dose_response(X, y, g, logistic, KS, seed, conditions=cond).assign(
        target=target, labels=labels, run=run, leak="other paradigms" if cond is not None else "any block"
    )


def main() -> None:
    out = C.out_dir("positive_controls_cross_paradigm")
    t0 = time.time()
    X, Y, g, par, m = data()
    rng, seed = generator(C.SEED, "v3_b9")
    true_seed = int(rng.integers(0, 10**5))
    perms = {
        t: [(IA.permute_subject_labels(Y[t], g, rng), int(rng.integers(0, 10**5)), i) for i in range(N_PERM)] for t in Y
    }
    cjobs, djobs = [], []
    for t in ("age above median", "sex (male)"):
        for sc in IA.SCHEMES:
            cjobs.append((X, Y[t], g, sc, REPEATS, true_seed, t, "true", -1))
            cjobs += [(X, yp, g, sc, 1, s, t, "permuted", i) for yp, s, i in perms[t]]
        djobs += [(X, Y[t], g, true_seed + r, t, "true", r, None) for r in range(REPEATS)]
        djobs += [(X, yp, g, s, t, "permuted", i, None) for yp, s, i in perms[t]]
    t = "diagnosis"
    djobs += [(X, Y[t], g, true_seed + r, t, "true", r, par) for r in range(REPEATS)]
    djobs += [(X, yp, g, s, t, "permuted", i, par) for yp, s, i in perms[t]]
    Cn = pd.concat(Parallel(n_jobs=N_JOBS, verbose=2)(delayed(contrast_job)(*j) for j in cjobs), ignore_index=True)
    Cn.to_parquet(out / "positive_controls.parquet")
    print(f"contrasts done {time.time() - t0:.0f} s", flush=True)
    D = pd.concat(Parallel(n_jobs=N_JOBS, verbose=2)(delayed(dose_job)(*j) for j in djobs), ignore_index=True)
    D.to_parquet(out / "dose.parquet")

    def band(v):
        return [float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    summ = {
        "seed": seed,
        "true_seed": true_seed,
        "n_perm": N_PERM,
        "repeats": REPEATS,
        "age_median": float(m.age.median()),
        "n_age_above": int((m.age > m.age.median()).sum()),
        "n_male": int(m.sex_male.sum()),
        "n_children": len(m),
        "contrast": {
            f"{t} | {sc} | {lab}": (
                [float(d.auc_subject.mean()), float(d.auc_subject.min()), float(d.auc_subject.max())]
                if lab == "true"
                else band(d.auc_subject)
            )
            for (t, sc, lab), d in Cn.groupby(["target", "scheme", "labels"])
        },
        "dose": {
            f"{t} | {lk} | {lab} | k={k}": (
                [float(d.auc_subject.mean()), float(d.auc_subject.min()), float(d.auc_subject.max())]
                if lab == "true"
                else band(d.auc_subject)
            )
            for (t, lk, lab, k), d in D.groupby(["target", "leak", "labels", "k"])
        },
        "minutes": (time.time() - t0) / 60,
    }
    C.write_json(out / "summary.json", summ)
    for k, v in summ["contrast"].items():
        print(k, np.round(v, 3))
    for k, v in summ["dose"].items():
        print(k, np.round(v, 3))


if __name__ == "__main__":
    main()
