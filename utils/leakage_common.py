"""Shared helpers of the leakage tests (Additional file 1, Supplementary Methods S5).

- :func:`core_asd_td`: block-level log power (64 channels x 33 1-Hz bins = 2,112 features, six paradigms) of the
  70 autistic and TD core children, with the paradigm and the visit index of every block (visits are relative day
  indices; no dates).
- :func:`dose_permutations`: the 20 child-level label permutations and seeds shared by the dose-response analyses.
- :func:`dose_general`: the dose-response test of :func:`identity_audit.dose_response` with two extensions. With
  ``reserve > 0``, ``reserve`` blocks of every training child are set aside before training and the size-matched
  condition adds, at each k, the same number of blocks from these reserves as the leaky condition adds from the test
  children (size-matched control; Fig. S3a). With ``conditions`` and ``choose="smaller"``, the scored condition of
  a test child is its smaller condition (visit) and the leaked blocks come only from its other condition
  (cross-visit leak; Fig. S3b). Without these options it consumes the same random numbers in the same order as
  :func:`identity_audit.dose_response` and returns the same AUCs.
- :func:`band` and :func:`mean_range`: median with central 95%, and mean with range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from identity_audit import identity_audit as IA
from leakage_tests.model_families_and_dose_response import KS, logistic  # noqa: F401  (used as RC.logistic)
from stats.rng import generator
from utils import common as C
from utils.inputs import Inputs

OUT = C.OUT
N_JOBS = 24


def meta_core():
    inp = Inputs(C.CFG)
    return inp.meta(), inp.core()


def blocks(kids):
    """Block-level log power (2,112 features) of ``kids`` with paradigm and visit (day index, relative only)."""
    w = pd.read_parquet(C.BANK / "spectral_logpow.parquet")
    w = w[w.child.isin(kids)].reset_index(drop=True)
    t = pd.read_parquet(C.OUT / "visit_structure/recording_times_relative.parquet")
    day = (
        w[["child", "paradigm"]]
        .merge(t[["child", "paradigm", "day"]], on=["child", "paradigm"], how="left")
        .day
    )
    feats = [c for c in w.columns if c not in ("child", "paradigm", "block")]
    return w[feats].to_numpy(float), w.child.to_numpy(), w.paradigm.to_numpy(), day.to_numpy()


def core_asd_td():
    meta, core = meta_core()
    kids = [k for k in core if meta.loc[k, "group"] in ("ASD", "TD")]
    X, g, par, day = blocks(kids)
    y = (meta.loc[g, "group"] == "ASD").to_numpy(int)
    return X, y, g, par, day, meta


def dose_permutations(y, g):
    """The 20 child-level permutations and seeds and the true-label seed of the dose-response test (same generator
    calls as :mod:`leakage_tests.model_families_and_dose_response`)."""
    rng, seed = generator(C.SEED, "v3_b8")
    perms = [(IA.permute_subject_labels(y, g, rng), int(rng.integers(0, 10**5)), i) for i in range(20)]
    true_seed = int(rng.integers(0, 10**5))
    return perms, true_seed, seed


def _auc(scores, y, g, subjects=None):
    s = np.asarray(scores, float)
    ok = ~np.isnan(s)
    if subjects is not None:
        ok &= np.isin(g, list(subjects))
    d = pd.DataFrame({"g": g[ok], "y": y[ok], "s": s[ok]})
    if d.y.nunique() < 2:
        return np.nan, np.nan, 0
    sub = d.groupby("g").agg(y=("y", "first"), s=("s", "mean"))
    a_sub = float(roc_auc_score(sub.y, sub.s)) if sub.y.nunique() == 2 else np.nan
    return float(roc_auc_score(d.y, d.s)), a_sub, len(sub)


def dose_general(
    X,
    y,
    groups,
    make_model,
    ks=KS,
    seed=0,
    n_splits=5,
    min_eval=4,
    conditions=None,
    choose="random",
    reserve=0,
    reserve_seed=0,
    score_subjects=None,
):
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(reserve_seed)
    g = np.asarray(groups)
    cond = None if conditions is None else np.asarray(conditions)
    arms = ["leaky"] + (["matched"] if reserve else [])
    scores = {(a, k): np.full(len(y), np.nan) for a in arms for k in ks}
    kmax = max(ks)
    n_added = {k: [] for k in ks}
    for tr, te in IA._splits(X, y, g, "by subject", n_splits, seed):
        pool, evals = {}, []
        for subj in np.unique(g[te]):
            if cond is None:
                idx = rng.permutation(te[g[te] == subj])
                p = min(kmax, len(idx) - min_eval)
                pool[subj] = idx[:p]
                evals.append(idx[p:])
            else:
                own = te[g[te] == subj]
                cs, n = np.unique(cond[own], return_counts=True)
                if choose == "smaller":
                    ok = n >= min_eval
                    c = cs[ok][n[ok].argmin()] if ok.any() else cs[n.argmax()]
                else:
                    eligible = cs[n >= min_eval] if (n >= min_eval).any() else cs[[n.argmax()]]
                    c = eligible[rng.integers(len(eligible))]
                evals.append(own[cond[own] == c])
                pool[subj] = rng.permutation(own[cond[own] != c])[:kmax]
        ev = np.concatenate(evals)
        base = tr
        resorder = np.array([], int)
        if reserve:
            res = {s: rng2.permutation(tr[g[tr] == s])[:reserve] for s in np.unique(g[tr])}
            order = rng2.permutation(list(res))
            resorder = np.array([res[s][j] for j in range(reserve) for s in order if j < len(res[s])], int)
            base = np.setdiff1d(tr, resorder)
        for k in ks:
            add = np.concatenate([pool[s][:k] for s in pool]).astype(int) if k else np.array([], int)
            n_added[k].append(len(add))
            for a in arms:
                extra = add if a == "leaky" else resorder[: len(add)]
                if a == "matched" and len(extra) < len(add):
                    raise ValueError("reserve too small for the size-matched condition")
                idx = np.concatenate([base, extra]).astype(int)
                m = make_model(seed)
                m.fit(X[idx], y[idx])
                scores[(a, k)][ev] = IA._score(m, X[ev])
    rows = []
    for a in arms:
        for k in ks:
            ab, asub, _ = _auc(scores[(a, k)], y, g)
            row = {"arm": a, "k": k, "auc_block": ab, "auc_subject": asub, "n_added": int(np.sum(n_added[k]))}
            if score_subjects is not None:
                sb, ss, nsub = _auc(scores[(a, k)], y, g, score_subjects)
                row.update({"auc_block_scored": sb, "auc_subject_scored": ss, "n_subjects_scored": nsub})
            rows.append(row)
    return pd.DataFrame(rows)


def band(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))]


def mean_range(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.mean(v)), float(np.min(v)), float(np.max(v)), int(len(v))]
