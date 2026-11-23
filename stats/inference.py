"""Small inference helpers: bootstrap intervals, sign-flip and Freedman-Lane permutation, Holm and BH, Hedges' g."""

from __future__ import annotations

import numpy as np


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    a = (1 - level) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(c - h, 0.0)), float(min(c + h, 1.0))


def paired_bootstrap_ci(
    d: np.ndarray, n: int, rng: np.random.Generator, level: float = 0.95
) -> tuple[float, float]:
    """Percentile CI of the mean of ``d`` (bootstrap over units)."""
    d = np.asarray(d, float)
    idx = rng.integers(0, d.size, (n, d.size))
    return percentile_ci(d[idx].mean(axis=1), level)


def sign_flip_p(d: np.ndarray, n: int, rng: np.random.Generator) -> float:
    """Two-sided sign-flip permutation p for mean(d) = 0."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    s = rng.choice([-1.0, 1.0], size=(n, d.size))
    null = np.abs((s * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-15).sum()) / (1 + n))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    m = p.size
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for k in range(m - 1, -1, -1):
        i = order[k]
        prev = min(prev, p[i] * m / (k + 1))
        q[i] = prev
    return q


def hedges_g(diff: float, sd_pooled: float, n1: int, n2: int) -> float:
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(j * diff / sd_pooled)


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and residuals (``X`` includes the intercept)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def freedman_lane_p(y: np.ndarray, X: np.ndarray, col: int, n: int, rng: np.random.Generator) -> float:
    """Freedman-Lane permutation p (two-sided) for coefficient ``col`` of ``y ~ X`` (``X`` with intercept)."""
    Z = np.delete(X, col, axis=1)
    bz, rz = ols(y, Z)
    fitted = Z @ bz

    def tstat(yy):
        b, r = ols(yy, X)
        s2 = r @ r / (len(yy) - X.shape[1])
        cov = s2 * np.linalg.inv(X.T @ X)
        return b[col] / np.sqrt(cov[col, col])

    obs = abs(tstat(y))
    null = np.array([abs(tstat(fitted + rng.permutation(rz))) for _ in range(n)])
    return float((1 + (null >= obs).sum()) / (1 + n))
