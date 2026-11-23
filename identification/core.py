"""Identification statistics as plain functions on child x feature matrices: z-scoring across children,
residualisation on covariates, Pearson correlation between children, top-1 identification, differential
identifiability (I_diff, mean self-similarity minus mean other-similarity x 100) and per-child self-identifiability
(mean of the row and column z of r_ii against r_ij; Additional file 1, Supplementary Methods S2)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_columns(X: np.ndarray) -> np.ndarray:
    """z-score each feature across children; constant / all-missing features become 0, missing entries 0."""
    m = np.nanmean(X, axis=0)
    s = np.nanstd(X, axis=0)
    Z = (X - m) / np.where(s > 0, s, 1.0)
    Z[:, ~(s > 0)] = 0.0
    return np.nan_to_num(Z, nan=0.0)


def residualise(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Residuals of every column of ``X`` on covariates ``C`` (intercept added; rows with missing C kept as is).
    A column with missing values is fitted on its observed rows only; missing entries stay missing."""
    D = np.column_stack([np.ones(len(C)), C])
    ok = np.all(np.isfinite(D), axis=1)
    out = X.copy()
    Xo, Do = X[ok], D[ok]
    seen = np.isfinite(Xo)
    full = seen.all(axis=0)
    if full.any():
        beta, *_ = np.linalg.lstsq(Do, Xo[:, full], rcond=None)
        out[np.ix_(ok, full)] = Xo[:, full] - Do @ beta
    rows = np.flatnonzero(ok)
    for j in np.flatnonzero(~full):
        r = seen[:, j]
        if r.sum() <= D.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(Do[r], Xo[r, j], rcond=None)
        out[rows[r], j] = Xo[r, j] - Do[r] @ beta
    return out


def corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between every row of ``A`` and every row of ``B`` (rows = children)."""
    a = A - A.mean(axis=1, keepdims=True)
    b = B - B.mean(axis=1, keepdims=True)
    a /= np.where((na := np.linalg.norm(a, axis=1, keepdims=True)) > 0, na, 1)
    b /= np.where((nb := np.linalg.norm(b, axis=1, keepdims=True)) > 0, nb, 1)
    return a @ b.T


def accuracy(R: np.ndarray) -> float:
    """Fraction of rows whose maximum is on the diagonal (row i = child i in the source paradigm)."""
    return float(np.mean(correct_rows(R)))


def correct_rows(R: np.ndarray) -> np.ndarray:
    """Row i is correct if r_ii is strictly larger than every r_ij (a tie, e.g. an all-zero row, is not)."""
    off = np.array(R, dtype=float, copy=True)
    np.fill_diagonal(off, -np.inf)
    return np.diag(R) > off.max(axis=1)


def differential_identifiability(R: np.ndarray) -> float:
    """100 x (mean self-similarity - mean other-similarity) (Amico and Goni 2018)."""
    n = R.shape[0]
    off = ~np.eye(n, dtype=bool)
    return float(100 * (np.diag(R).mean() - R[off].mean()))


def self_identifiability_z(R: np.ndarray) -> np.ndarray:
    """Per child: mean of the row and column z of r_ii against r_ij (j != i)."""
    n = R.shape[0]
    off = ~np.eye(n, dtype=bool)
    d = np.diag(R)
    rows = R[off].reshape(n, n - 1)
    cols = R.T[off].reshape(n, n - 1)
    zr = (d - rows.mean(1)) / rows.std(1, ddof=1)
    zc = (d - cols.mean(1)) / cols.std(1, ddof=1)
    return (zr + zc) / 2


def permutation_accuracies(R: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """Accuracy when the target paradigm's children are relabelled by each permutation (rows of ``perms``)."""
    am = np.argmax(R, axis=1)
    n = R.shape[0]
    # relabelling the target columns by pi: child i is "correct" if argmax column maps to i, i.e. pi[am[i]] == i
    return (perms[:, am] == np.arange(n)).mean(axis=1)


def paradigm_matrix(
    wide: pd.DataFrame, paradigm: str, children: list[str], blocks: str = "all"
) -> np.ndarray:
    """Child x feature matrix: mean over the paradigm's blocks (``all``, ``odd`` or ``even`` block order)."""
    d = wide[(wide.paradigm == paradigm) & wide.child.isin(children)].copy()
    if blocks != "all":
        d["rank"] = d.groupby("child")["block"].rank(method="first").astype(int)
        d = d[(d["rank"] % 2 == 1) if blocks == "odd" else (d["rank"] % 2 == 0)].drop(columns="rank")
    feats = [c for c in d.columns if c not in ("child", "paradigm", "block")]
    m = d.groupby("child")[feats].mean()
    return m.reindex(children).to_numpy(float)
