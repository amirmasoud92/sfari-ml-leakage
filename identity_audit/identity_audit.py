"""identity_audit: a subject-level leakage audit for EEG (or any repeated-measures) classifiers.

The module packages the evaluation steps of the paper so that they can be applied to another dataset. All
functions take

    X       array (n_blocks, n_features): one row per block (segment, epoch, trial average, recording part)
    y       array (n_blocks,): binary label of the block's subject (constant within a subject)
    groups  array (n_blocks,): subject identifier of each block
    make_model(seed) -> unfitted scikit-learn estimator with predict_proba or decision_function

and return plain numbers or pandas data frames. Nothing is tuned: the caller supplies a fixed pipeline, and every
transformation is fitted on training blocks only (put scalers and projections inside the estimator).

Analyses, numbered as the cards of Fig. 1 of the paper (cards 2 and 7 need dataset-specific code and are not here;
Additional file 1, Supplementary Methods S5):

Card 1. ``fingerprint_accuracy``: top-1 identification of subjects between two conditions from z-scored profiles.
Card 3. ``identify_blocks``: which subject produced a held-out block (block-level split), with a subject bootstrap.
Card 4. ``split_contrast``: the same pipeline cross-validated with subjects kept whole (``"by subject"``) and with
        blocks of a subject in training and test folds (``"by block"``, leaky); AUC at block and subject level.
Card 5. ``permuted_label_control``: labels permuted between subjects (all blocks of a subject keep one label) and
        the split contrast rerun. A high leaky AUC with permuted labels measures recognition of subjects, not the
        label.
Card 6. ``dose_response``: subjects kept whole, then k blocks of each test subject added to training; AUC on the test
        subjects' remaining blocks as a function of k. With ``conditions``, the leaked blocks come only from
        conditions (paradigms, sessions) other than the one that is scored, which separates subject identity from
        recording identity.
Card 8 (part). ``error_corrected_auc``, ``auc_from_g`` and ``n_per_group``: the classification an effect size g
        supports without within-session measurement error (reliability r), and the sample size for a two-group
        test. Reliability itself is estimated by the caller.

The module is tested on synthetic data with known ground truth (``tests/test_identity_audit.py``).

Seeds: ``split_contrast`` uses ``seed + repeat`` for the folds and ``repeat`` for the model, so that results are
reproducible run by run.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SCHEMES = ("by subject", "by block")


def _splits(X, y, groups, scheme: str, n_splits: int, seed: int):
    if scheme == "by subject":
        return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups)
    if scheme == "by block":
        return StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)
    raise ValueError(f"unknown scheme {scheme!r}; use one of {SCHEMES}")


def _score(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:  # e.g. SVC without probability=True exposes the attribute but fails
            pass
    return model.decision_function(X)


def subject_auc(scores, y, groups) -> tuple[float, float]:
    """(block-level AUC, subject-level AUC); the subject score is the mean score over its scored blocks."""
    s = np.asarray(scores, float)
    ok = ~np.isnan(s)
    d = pd.DataFrame({"g": np.asarray(groups)[ok], "y": np.asarray(y)[ok], "s": s[ok]})
    sub = d.groupby("g").agg(y=("y", "first"), s=("s", "mean"))
    return float(roc_auc_score(d.y, d.s)), float(roc_auc_score(sub.y, sub.s))


def oof_scores(X, y, groups, make_model: Callable, scheme: str, split_seed: int, model_seed: int, n_splits=5):
    """Out-of-fold scores of every block under one splitting scheme."""
    s = np.full(len(y), np.nan)
    for tr, te in _splits(X, y, groups, scheme, n_splits, split_seed):
        m = make_model(model_seed)
        m.fit(X[tr], y[tr])
        s[te] = _score(m, X[te])
    return s


def split_contrast(X, y, groups, make_model, repeats=10, seed=0, schemes=SCHEMES, n_splits=5) -> pd.DataFrame:
    """Card 4: block- and subject-level AUC under each scheme, one row per scheme and repeat.

    ``repeats`` is a number of repeats (0, 1, ...) or an iterable of repeat indices (to split a run across
    workers); repeat r uses fold seed ``seed + r`` and model seed r.
    """
    rows = []
    rs = range(repeats) if isinstance(repeats, int) else list(repeats)
    for scheme in schemes:
        for r in rs:
            s = oof_scores(X, y, groups, make_model, scheme, seed + r, r, n_splits)
            ab, asub = subject_auc(s, y, groups)
            rows.append({"scheme": scheme, "repeat": r, "auc_block": ab, "auc_subject": asub})
    return pd.DataFrame(rows)


def permute_subject_labels(y, groups, rng: np.random.Generator) -> np.ndarray:
    """Permute labels between subjects; every block of a subject receives that subject's new label."""
    g = np.asarray(groups)
    subj, first = np.unique(g, return_index=True)
    lab = np.asarray(y)[first]
    new = dict(zip(subj, rng.permutation(lab), strict=True))
    return np.array([new[k] for k in g])


def permuted_label_control(
    X, y, groups, make_model, n_perm=40, repeats=1, seed=0, schemes=SCHEMES, n_jobs=1
) -> pd.DataFrame:
    """Card 5: ``split_contrast`` on labels permuted between subjects (same permutation for every scheme)."""
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_perm):
        yp = permute_subject_labels(y, groups, rng)
        s = int(rng.integers(0, 10**5))
        jobs += [(i, sc, yp, s) for sc in schemes]

    def one(i, sc, yp, s):
        R = split_contrast(X, yp, groups, make_model, repeats, s, (sc,))
        return {"perm": i, "scheme": sc, "auc_block": R.auc_block.mean(), "auc_subject": R.auc_subject.mean()}

    return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(one)(*j) for j in jobs))


def dose_response(
    X,
    y,
    groups,
    make_model,
    ks=(0, 1, 2, 4, 8, 16),
    seed=0,
    n_splits=5,
    min_eval=4,
    conditions=None,
    return_index=False,
):
    """Card 6: subject-level AUC on held-out blocks when k blocks of each test subject are in training.

    Folds keep subjects whole. Without ``conditions``, each test subject's blocks are put in a random order; the first
    ``min(max(ks), n_blocks - min_eval)`` form a pool from which the first k are added to training, and the
    remaining blocks are scored for every k (the same evaluation blocks for all k).

    With ``conditions`` (one label per block, e.g. the paradigm), one condition of each test subject is drawn at
    random among those with at least ``min_eval`` blocks; all its blocks are scored, and the pool is drawn in random
    order from the subject's other conditions only (up to max(ks) blocks; fewer if the subject has fewer).

    ``return_index=True`` also returns {k: (training indices, evaluation indices)} per fold, for tests.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    cond = None if conditions is None else np.asarray(conditions)
    scores = {k: np.full(len(y), np.nan) for k in ks}
    kmax = max(ks)
    index = []
    for tr, te in _splits(X, y, g, "by subject", n_splits, seed):
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
                eligible = cs[n >= min_eval] if (n >= min_eval).any() else cs[[n.argmax()]]
                c = eligible[rng.integers(len(eligible))]
                evals.append(own[cond[own] == c])
                pool[subj] = rng.permutation(own[cond[own] != c])[:kmax]
        ev = np.concatenate(evals)
        fold = {}
        for k in ks:
            add = np.concatenate([pool[s][:k] for s in pool]) if k else np.array([], int)
            idx = np.concatenate([tr, add]).astype(int)
            m = make_model(seed)
            m.fit(X[idx], y[idx])
            scores[k][ev] = _score(m, X[ev])
            fold[k] = (idx, ev)
        index.append(fold)
    rows = []
    for k in ks:
        ab, asub = subject_auc(scores[k], y, g)
        rows.append({"k": k, "auc_block": ab, "auc_subject": asub})
    return (pd.DataFrame(rows), index) if return_index else pd.DataFrame(rows)


def identify_blocks(X, groups, make_model, seed=0, n_splits=5, n_boot=2000) -> dict:
    """Card 3: attribute each held-out block to a subject (block split); subject-bootstrap interval."""
    g = np.asarray(groups)
    labels = pd.factorize(g)[0]
    pred = np.empty(len(g), int)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, labels):
        m = make_model(seed)
        m.fit(X[tr], labels[tr])
        pred[te] = m.predict(X[te])
    per = pd.Series(pred == labels).groupby(g).agg(["sum", "count"])
    a, n = per["sum"].to_numpy(float), per["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    bs = [(a[i].sum() / n[i].sum()) for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    return {
        "accuracy": float(a.sum() / n.sum()),
        "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "chance": 1 / len(a),
        "subjects_all_correct": int((a == n).sum()),
        "n_subjects": len(a),
        "n_blocks": int(n.sum()),
    }


def fingerprint_accuracy(A, B) -> float:
    """Card 1: top-1 identification between two conditions.

    A and B are (n_subjects, n_features) with rows in the same subject order. Features are z-scored across
    subjects within each condition; subject i is identified if its most correlated row of B is row i.
    """
    keep = (A.std(0, ddof=1) > 0) & (B.std(0, ddof=1) > 0)  # constant features carry no identity
    A, B = A[:, keep], B[:, keep]
    za = (A - A.mean(0)) / A.std(0, ddof=1)
    zb = (B - B.mean(0)) / B.std(0, ddof=1)
    za = (za - za.mean(1, keepdims=True)) / za.std(1, keepdims=True)
    zb = (zb - zb.mean(1, keepdims=True)) / zb.std(1, keepdims=True)
    r = za @ zb.T / A.shape[1]
    return float((r.argmax(1) == np.arange(len(A))).mean())


def error_corrected_auc(g, r):
    """AUC supported by effect size g without within-session error (equal-variance normal model).

    Phi(|g| / sqrt(2 r)). The correction is undefined for r <= 0; as in :mod:`markers.reliability_validity`, such
    values are set to 1.0, the upper bound, so bootstrap upper limits of 1.00 can reflect resamples with r <= 0. NaN
    inputs return NaN.
    """
    g, r = np.asarray(g, float), np.asarray(r, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = stats.norm.cdf(np.abs(g) / np.sqrt(2 * r))
    return np.where(np.isnan(g) | np.isnan(r), np.nan, np.where(r > 0, out, 1.0))


def auc_from_g(g):
    """AUC supported by an observed effect size g (equal-variance normal model): Phi(|g| / sqrt(2))."""
    return stats.norm.cdf(np.abs(np.asarray(g, float)) / np.sqrt(2))


def n_per_group(g, power=0.8, alpha=0.05) -> float:
    """Children per group for a two-sided two-sample t-test to detect g (normal approximation)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float(np.ceil(2 * ((za + zb) / abs(g)) ** 2)) if g else float("inf")
