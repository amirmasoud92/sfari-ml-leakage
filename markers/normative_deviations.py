"""Individual deviations from a TD normative model (Methods, Reliability and validity of candidate markers;
Fig. 6a, b; Additional file 1, Supplementary Methods S4).

For each of the 16 markers with split-half values, a TD normative model is fitted (OLS on age and sex in TD
children): TD children are scored leave-one-out, autistic children and siblings against the full TD model, as
z = residual / TD residual SD. A deviation is extreme if |z| > 1.96. Children with at least eight markers enter.

1. Per child: proportion of available markers with an extreme deviation; autistic and sibling children against TD
   (difference in means with a bootstrap CI; two-sided permutation p with the whole procedure rerun on 200
   relabellings of autistic and TD children).
2. Per marker: proportion of autistic children with an extreme deviation against TD (Fisher exact; BH over markers).
3. Overlap: among children with at least one extreme deviation, mean pairwise Jaccard index of their deviating
   marker sets, autistic against TD (two-sided permutation as in 1).
4. Within autistic children: Spearman correlation of the deviation proportion with SRS-2, ADOS calibrated severity
   and data quality.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from stats.rng import generator
from utils import common as C

THR = 1.96


def deviations(V: pd.DataFrame, m: pd.DataFrame, td: np.ndarray) -> pd.DataFrame:
    """z per child x marker against the TD normative model (TD leave-one-out)."""
    Z = pd.DataFrame(np.nan, index=V.index, columns=V.columns)
    age, sex = m.age.reindex(V.index).to_numpy(float), m.sex_male.reindex(V.index).to_numpy(float)
    for j, col in enumerate(V.columns):
        y = V[col].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(age) & np.isfinite(sex)
        X = np.column_stack([np.ones(len(y)), age, sex])
        ref = ok & td
        for i in np.flatnonzero(ok):
            r = ref.copy()
            if td[i]:
                r[i] = False
            beta, *_ = np.linalg.lstsq(X[r], y[r], rcond=None)
            sd = np.std(y[r] - X[r] @ beta, ddof=3)
            Z.iat[i, j] = (y[i] - X[i] @ beta) / sd
    return Z


def jaccard_mean(E: pd.DataFrame) -> float:
    sets = [set(E.columns[E.loc[k].fillna(False).to_numpy(bool)]) for k in E.index]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return np.nan
    return float(np.mean([len(a & b) / len(a | b) for a, b in combinations(sets, 2)]))


def summarise(Z: pd.DataFrame, groups: pd.Series) -> dict:
    E = Z.abs() > THR
    E = E.where(Z.notna())
    prop = E.astype(float).mean(axis=1)
    out = {}
    for g in ("ASD", "TD", "SIB"):
        k = groups.index[groups == g]
        out[f"prop_{g}"] = float(prop.reindex(k).mean())
        out[f"any_{g}"] = float((prop.reindex(k) > 0).mean())
        out[f"jacc_{g}"] = jaccard_mean(E.reindex(k))
    return out


def main() -> None:
    out = C.out_dir("normative_deviations")
    mk = pd.read_parquet(C.OUT / "reliability_validity/markers.parquet")
    markers = list(mk.marker)
    v = pd.read_parquet(C.OUT / "marker_halves/marker_values.parquet")
    V = v[(v.half == -1) & v.marker.isin(markers)].pivot_table(
        index="child", columns="marker", values="value"
    )[markers]
    m = C.meta()
    V = V[m.group.reindex(V.index).isin(["ASD", "TD", "SIB"])]
    V = V[V.notna().sum(axis=1) >= 8]
    groups = m.group.reindex(V.index)
    Z = deviations(V, m, (groups == "TD").to_numpy())
    Z.to_parquet(out / "z.parquet")
    obs = summarise(Z, groups)
    rng, seed = generator(C.SEED, "v2_b3")
    at = groups[groups.isin(["ASD", "TD"])]
    null = []
    for _ in range(200):
        g2 = groups.copy()
        g2[at.index] = rng.permutation(at.to_numpy())
        null.append(summarise(deviations(V, m, (g2 == "TD").to_numpy()), g2))
    N = pd.DataFrame(null)
    E = (Z.abs() > THR).where(Z.notna())
    prop = E.astype(float).mean(axis=1)
    a, t = prop[groups == "ASD"], prop[groups == "TD"]
    bs = [rng.choice(a, len(a)).mean() - rng.choice(t, len(t)).mean() for _ in range(2000)]
    res = {
        **obs,
        "n_asd": int((groups == "ASD").sum()),
        "n_td": int((groups == "TD").sum()),
        "n_sib": int((groups == "SIB").sum()),
        "n_markers": len(markers),
        "n_perm": len(N),
        "diff_prop_asd_td": float(a.mean() - t.mean()),
        "diff_ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
        # two-sided (absolute differences)
        "p_perm_prop": float(
            (1 + ((N.prop_ASD - N.prop_TD).abs() >= abs(obs["prop_ASD"] - obs["prop_TD"])).sum())
            / (1 + len(N))
        ),
        "p_perm_jacc": float(
            (1 + ((N.jacc_ASD - N.jacc_TD).abs() >= abs(obs["jacc_ASD"] - obs["jacc_TD"])).sum())
            / (1 + len(N))
        ),
        "seed": seed,
    }
    per = []
    for mkr in markers:
        ea, et = E.loc[groups == "ASD", mkr].dropna(), E.loc[groups == "TD", mkr].dropna()
        tab = [[int(ea.sum()), int(len(ea) - ea.sum())], [int(et.sum()), int(len(et) - et.sum())]]
        per.append(
            {
                "marker": mkr,
                "asd_extreme": float(ea.mean()),
                "td_extreme": float(et.mean()),
                "n_asd": len(ea),
                "n_td": len(et),
                "p_fisher": stats.fisher_exact(tab)[1],
            }
        )
    P = pd.DataFrame(per)
    o = np.argsort(P.p_fisher.to_numpy())
    q = np.minimum.accumulate((P.p_fisher.to_numpy()[o] * len(P) / np.arange(1, len(P) + 1))[::-1])[::-1]
    P.loc[P.index[o], "q_bh"] = np.minimum(q, 1)
    P.to_parquet(out / "per_marker.parquet")
    asd = (
        prop[groups == "ASD"]
        .to_frame("prop")
        .join(m[["srs2_total_t", "ados_css", "mean_bad_fraction", "log_clean_minutes"]])
    )
    rel = {}
    for col in ("srs2_total_t", "ados_css", "mean_bad_fraction", "log_clean_minutes"):
        s = asd[["prop", col]].dropna()
        r = stats.spearmanr(s.prop, s[col])
        rel[col] = {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(s)}
    res["within_asd"] = rel
    C.write_json(out / "summary.json", res)
    prop.to_frame("prop_extreme").join(groups).to_parquet(out / "per_child.parquet")
    print(pd.Series({k: v for k, v in res.items() if k != "within_asd"}).to_string())
    print(rel)
    print(P.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
