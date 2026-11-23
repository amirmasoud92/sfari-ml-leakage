"""Self-identifiability and data quality (Fig. 6d; Additional file 1, Supplementary Methods S4).

In all children with at least two kept paradigms, self-identifiability of log power (per unordered paradigm pair the
mean of the row and column z of r_ii, averaged over the child's pairs) is related to age, data quality and the
phenotype: Spearman correlations with child-bootstrap intervals, partial correlations controlling age and data
quality, Benjamini-Hochberg q over the phenotype tests, and the variance of self-identifiability explained by the
data-quality measures (linear regression R^2).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from identification import core
from stats.rng import generator
from utils import common as C
from utils.inputs import Inputs

QUALITY = ["log_clean_minutes", "mean_bad_fraction", "mean_asr_fraction", "mean_ica_removed"]
PHENO = ["srs2_total_t", "ados_css", "mabc_total_ss", "fsiq"]


def self_ident(rep: dict[str, pd.DataFrame], columns=None, children=None) -> pd.Series:
    """Mean self-identifiability z per child over the unordered paradigm pairs it has."""
    vals: dict[str, list[float]] = {}
    for a, b in itertools.combinations(C.PARADIGMS, 2):
        kids = sorted(set(rep[a].index) & set(rep[b].index))
        if children is not None:
            kids = [k for k in kids if k in set(children)]
        A, B = rep[a].loc[kids], rep[b].loc[kids]
        if columns is not None:
            A, B = A[columns], B[columns]
        R = core.corr_matrix(core.zscore_columns(A.to_numpy(float)), core.zscore_columns(B.to_numpy(float)))
        for k, z in zip(kids, core.self_identifiability_z(R), strict=True):
            vals.setdefault(k, []).append(float(z))
    return pd.Series({k: np.mean(v) for k, v in vals.items()}, name="si")


def spearman_ci(x, y, rng, n=2000):
    r = stats.spearmanr(x, y).statistic
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        bs.append(stats.spearmanr(x[i], y[i]).statistic)
    return r, *np.nanpercentile(bs, [2.5, 97.5])


def partial_spearman(x, y, Z):
    """Spearman partial correlation: Pearson r of rank residuals on ranked covariates."""
    rk = lambda v: stats.rankdata(v)  # noqa: E731
    D = np.column_stack([np.ones(len(x)), *[rk(z) for z in Z.T]])
    rx = rk(x) - D @ np.linalg.lstsq(D, rk(x), rcond=None)[0]
    ry = rk(y) - D @ np.linalg.lstsq(D, rk(y), rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return r, p


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def main() -> None:
    out = C.out_dir("self_identifiability_correlates")
    inp = Inputs(C.CFG)
    meta = inp.meta()
    rep = C.bank_family("spectral_logpow")
    si = self_ident(rep)
    npar = pd.Series({k: sum(k in rep[p].index for p in C.PARADIGMS) for k in si.index})
    d = meta.loc[si.index].assign(si=si, n_paradigms=npar)
    d.to_parquet(out / "self_identifiability_all_available.parquet")
    rng, seed = generator(C.SEED, "committee_a3")

    rows = []
    for grp in ("all", "ASD", "TD", "SIB"):
        g = d if grp == "all" else d[d.group == grp]
        for v in ["age", "n_paradigms", *QUALITY, *PHENO]:
            s = g[list(dict.fromkeys(["si", v, "age", *QUALITY]))].dropna()
            if len(s) < 15 or s[v].nunique() < 3:
                continue
            r, lo, hi = spearman_ci(s.si.to_numpy(), s[v].to_numpy(float), rng)
            p = stats.spearmanr(s.si, s[v]).pvalue
            ctrl = [c for c in ["age", *QUALITY] if c != v]
            pr, pp = partial_spearman(s.si.to_numpy(), s[v].to_numpy(float), s[ctrl].to_numpy(float))
            rows.append(
                {
                    "group": grp,
                    "variable": v,
                    "n": len(s),
                    "rho": r,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p": p,
                    "partial_rho_age_quality": pr,
                    "partial_p": pp,
                }
            )
    T = pd.DataFrame(rows)
    ph = T.variable.isin(PHENO)
    T.loc[ph, "q_bh_phenotype"] = bh(T.loc[ph, "partial_p"])
    T.to_parquet(out / "correlates.parquet")
    print(T.round(3).to_string(index=False), flush=True)

    # how much of the variance of self-identifiability do recording-quality measures explain?
    s = d[["si", "age", *QUALITY]].dropna()
    X = np.column_stack([np.ones(len(s)), s[QUALITY].to_numpy(float)])
    beta = np.linalg.lstsq(X, s.si.to_numpy(), rcond=None)[0]
    r2_q = 1 - np.var(s.si - X @ beta) / np.var(s.si)

    C.write_json(
        out / "summary.json",
        {"n_all_available": len(d), "r2_quality": r2_q, "seed": seed},
    )
    print("R2 of self-identifiability on data quality:", round(r2_q, 3))


if __name__ == "__main__":
    main()
