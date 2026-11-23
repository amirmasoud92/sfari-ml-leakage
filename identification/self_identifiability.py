"""Autistic-versus-TD difference in self-identifiability (Methods, Reliability and validity of candidate
markers; Fig. 6c; Additional file 1, Table S16, H2).

Per child, self-identifiability is the mean over its paradigm pairs of the row and column z of r_ii. The group
difference is adjusted for age, sex and data quality (clean minutes, interpolated-channel fraction, ASR fraction):
Freedman-Lane permutation p, bootstrap interval within groups, Hedges' g with the 90% interval of the adjusted
group coefficient for an equivalence test at +-0.5, plus unadjusted, fully adjusted, sibling and ordinal-trend
companions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config.loader import Config
from stats.inference import freedman_lane_p, hedges_g, ols, percentile_ci
from stats.rng import generator

PRIMARY_COVARIATES = ["age", "sex_male", "log_clean_minutes", "mean_bad_fraction", "mean_asr_fraction"]
FULL_EXTRA = ["fsiq", "med_stimulant_or_ssri", "med_other", "med_unknown"]


def si_table(self_ident: pd.DataFrame, family: str) -> pd.Series:
    d = self_ident[(self_ident.family == family) & (self_ident.get("set", "core") == "core")]
    n_pairs = d.groupby("child")["pair"].nunique()
    return d.groupby("child")["si_z"].mean()[n_pairs == n_pairs.max()]


def _design(
    df: pd.DataFrame, cols: list[str], contrast: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    g1, g0 = contrast
    d = df[df.group.isin([g1, g0])].dropna(subset=["si", *cols])
    X = np.column_stack([np.ones(len(d)), (d.group == g1).astype(float), d[cols].to_numpy(float)])
    return d["si"].to_numpy(float), X, d


def contrast(
    df: pd.DataFrame,
    cols: list[str],
    cfg: Config,
    g: tuple[str, str] = ("ASD", "TD"),
    n_perm: int | None = None,
    n_boot: int | None = None,
    key: str = "h2",
) -> dict:
    y, X, d = _design(df, cols, g)
    n1, n0 = int((d.group == g[0]).sum()), int((d.group == g[1]).sum())
    if len(y) < X.shape[1] + 3 or min(n1, n0) < 3 or np.linalg.matrix_rank(X) < X.shape[1]:
        return {
            "contrast": f"{g[0]} - {g[1]}",
            "covariates": ",".join(cols) or "none",
            "n_1": n1,
            "n_0": n0,
            "estimate": np.nan,
            "se": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "p_freedman_lane": np.nan,
            "hedges_g": np.nan,
            "g_90_lo": np.nan,
            "g_90_hi": np.nan,
            "verdict": "not estimable",
            "seed": None,
        }
    beta, res = ols(y, X)
    s2 = res @ res / (len(y) - X.shape[1])
    se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
    n_perm = cfg.stats.h2_permutations if n_perm is None else n_perm
    n_boot = cfg.identification.h2_bootstrap if n_boot is None else n_boot
    rng, seed = generator(cfg.project.seed, key, *g, len(cols))
    p = freedman_lane_p(y, X, 1, n_perm, rng) if n_perm else np.nan
    idx1 = np.flatnonzero(d.group.to_numpy() == g[0])
    idx0 = np.flatnonzero(d.group.to_numpy() == g[1])
    draws = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(idx1, idx1.size), rng.choice(idx0, idx0.size)])
        draws.append(ols(y[i], X[i])[0][1])
    lo, hi = percentile_ci(np.array(draws), cfg.stats.ci_level)
    sd_res = float(np.sqrt(s2))
    g_es = hedges_g(float(beta[1]), sd_res, idx1.size, idx0.size)
    # the interval of g is the adjusted model's own interval of the group coefficient, in g units
    # (se * J / sd_res, t quantile on the residual df)
    se_g = float(se * hedges_g(1.0, sd_res, idx1.size, idx0.size))
    t90 = stats.t.ppf(0.95, len(y) - X.shape[1])
    b = cfg.identification.equivalence_bound_g
    g_lo90, g_hi90 = g_es - t90 * se_g, g_es + t90 * se_g
    verdict = (
        "different"
        if (lo > 0 or hi < 0)
        else ("equivalent" if (g_lo90 > -b and g_hi90 < b) else "inconclusive")
    )
    return {
        "contrast": f"{g[0]} - {g[1]}",
        "covariates": ",".join(cols) or "none",
        "n_1": int(idx1.size),
        "n_0": int(idx0.size),
        "estimate": float(beta[1]),
        "se": se,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_freedman_lane": p,
        "hedges_g": g_es,
        "g_90_lo": g_lo90,
        "g_90_hi": g_hi90,
        "verdict": verdict,
        "seed": seed,
    }


def run_h2(
    self_ident: pd.DataFrame, families: list[str], meta: pd.DataFrame, cfg: Config
) -> dict[str, pd.DataFrame]:
    """``meta`` (index child): group, covariates of PRIMARY_COVARIATES + FULL_EXTRA, srs2_total_t."""
    prim = cfg.identification.primary_family
    rows, per_child = [], []
    for fam in families:
        si = si_table(self_ident, fam)
        df = meta.join(si.rename("si"), how="inner")
        per_child.append(df[["group", "si"]].assign(family=fam).reset_index(names="child"))
        is_primary = fam == prim
        conf = {"n_perm": None, "n_boot": None} if is_primary else {"n_perm": 0, "n_boot": 500}
        r = contrast(df, PRIMARY_COVARIATES, cfg, **conf)
        rows.append(
            {
                "family": fam,
                "model": "primary (age, sex, data quality)",
                "role": "confirmatory" if is_primary else "estimation",
                **r,
            }
        )
        if is_primary:
            rows.append(
                {
                    "family": fam,
                    "model": "unadjusted",
                    "role": "estimation",
                    **contrast(df, [], cfg, n_perm=0, n_boot=1000, key="h2u"),
                }
            )
            rows.append(
                {
                    "family": fam,
                    "model": "fully adjusted (+ FSIQ, medication)",
                    "role": "estimation",
                    **contrast(df, PRIMARY_COVARIATES + FULL_EXTRA, cfg, n_perm=0, n_boot=1000, key="h2f"),
                }
            )
            for gg in (("SIB", "TD"), ("ASD", "SIB")):
                rows.append(
                    {
                        "family": fam,
                        "model": "primary (age, sex, data quality)",
                        "role": "estimation",
                        **contrast(df, PRIMARY_COVARIATES, cfg, g=gg, n_perm=0, n_boot=1000, key="h2s"),
                    }
                )
            o = df.dropna(subset=["si", *PRIMARY_COVARIATES]).copy()
            o["ordinal"] = o.group.map(cfg.reproduction.ordinal_codes)
            X = np.column_stack([np.ones(len(o)), o["ordinal"], o[PRIMARY_COVARIATES].to_numpy(float)])
            if len(o) < X.shape[1] + 3 or np.linalg.matrix_rank(X) < X.shape[1]:
                b, se = np.full(2, np.nan), np.nan
            else:
                b, res = ols(o["si"].to_numpy(), X)
                s2 = res @ res / (len(o) - X.shape[1])
                se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
            rows.append(
                {
                    "family": fam,
                    "model": "ordinal trend (ASD -1, SIB 0, TD +1)",
                    "role": "estimation",
                    "contrast": "per ordinal step",
                    "estimate": float(b[1]),
                    "se": se,
                    "ci_lo": float(b[1] - 1.96 * se),
                    "ci_hi": float(b[1] + 1.96 * se),
                }
            )
    pc = pd.concat(per_child, ignore_index=True)
    rel = []
    pm = pc[pc.family == prim].set_index("child").join(meta[["age", "srs2_total_t"]])
    for var in ("srs2_total_t", "age"):
        d = pm.dropna(subset=[var, "si"])
        if len(d) < 5:
            continue
        r, p = stats.pearsonr(d[var], d["si"])
        z = np.arctanh(r)
        se = 1 / np.sqrt(len(d) - 3)
        rel.append(
            {
                "family": prim,
                "variable": var,
                "n": len(d),
                "pearson_r": float(r),
                "ci_lo": float(np.tanh(z - 1.96 * se)),
                "ci_hi": float(np.tanh(z + 1.96 * se)),
                "p": float(p),
                "role": "estimation",
            }
        )
    return {"h2": pd.DataFrame(rows), "si_per_child": pc, "si_relations": pd.DataFrame(rel)}


def main() -> None:
    """Self-identifiability contrast from the stored identification result (``identification.analysis``)."""
    from config.loader import load_config
    from features.bank import SHARED_FAMILIES
    from utils.base import Result
    from utils.inputs import Inputs
    from utils.store import ResultsStore

    cfg = load_config()
    store = ResultsStore(cfg)
    inp = Inputs(cfg)
    core, meta = inp.core(), inp.meta()
    si = store.table("confirmatory", "identification", "self_ident")
    h2 = run_h2(si, list(SHARED_FAMILIES), meta.loc[core], cfg)
    store.write(Result("h2", h2, {"primary": h2["h2"].iloc[0].to_dict()}, label="confirmatory"), "confirmatory")


if __name__ == "__main__":
    main()
