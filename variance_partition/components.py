"""Variance components: child, paradigm, child x paradigm and block-residual variance (Additional file 1,
Supplementary Methods S2 and Table S16, H1b).

Model per scalar feature: y_cpb = mu + a_c + b_p + (ab)_cp + e_cpb. Two estimators:

- :func:`mom` (method of moments, vectorised over features): two-way ANOVA on the complete child x paradigm table
  of cell means, with the residual pooled from within-cell variance. sigma^2_c = (MS_c - MS_r) / P,
  sigma^2_p = (MS_p - MS_r) / C, sigma^2_cxp = MS_r - h sigma^2_e with h = mean(1 / n_cp). Exact for equal n_cp,
  approximate otherwise. Negative estimates are truncated at 0 (counted).
- :func:`reml` (statsmodels MixedLM, one group, three crossed variance components) for the posterior
  region-of-interest features (aperiodic exponent, aperiodic offset, periodic alpha); MoM is used where REML fails.

Running the module computes the variance-components result (``<results>/confirmatory/variance_components*``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config.loader import Config
from stats.inference import bh, percentile_ci
from stats.rng import generator
from utils.base import Analysis, Result
from utils.logging_utils import get_logger

log = get_logger(__name__)
TERMS = ("child", "paradigm", "cxp", "residual")


def mom_stats(
    long: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Per-child sufficient statistics: cell means ``A (C, P, F)``, counts ``n (C, P)``, within-cell sums of
    squares ``ssw (C, F)`` and the child order. Children without every paradigm are dropped here (the estimator
    needs the complete child x paradigm table)."""
    n_par = long["paradigm"].nunique()
    have = long.groupby("child")["paradigm"].nunique()
    long = long[long["child"].isin(have.index[have == n_par])].reset_index(drop=True)
    g = long.groupby(["child", "paradigm"])
    M = g[features].mean()
    cnt = g.size()
    dev = (
        long[features].to_numpy()
        - M.reindex(pd.MultiIndex.from_frame(long[["child", "paradigm"]])).to_numpy()
    )
    ss = pd.DataFrame(dev**2, columns=features).assign(child=long["child"].to_numpy()).groupby("child").sum()
    kids = list(M.index.get_level_values(0).unique())
    P = M.index.get_level_values(1).nunique()
    A = M.to_numpy().reshape(len(kids), P, len(features))
    n = cnt.to_numpy().reshape(len(kids), P).astype(float)
    return A, n, ss.reindex(kids).to_numpy(), kids


def mom_from_stats(A: np.ndarray, n: np.ndarray, ssw: np.ndarray) -> dict[str, np.ndarray]:
    """MoM variances from sufficient statistics (see the module docstring); negative values are kept here."""
    C, P, _ = A.shape
    df_w = float(np.clip(n - 1, 0, None).sum())
    s2e = ssw.sum(axis=0) / max(df_w, 1)
    gm = A.mean(axis=(0, 1))
    rm = A.mean(axis=1)
    cm = A.mean(axis=0)
    ms_c = P * ((rm - gm) ** 2).sum(axis=0) / (C - 1)
    ms_p = C * ((cm - gm) ** 2).sum(axis=0) / (P - 1)
    res = A - rm[:, None, :] - cm[None, :, :] + gm
    ms_r = (res**2).sum(axis=(0, 1)) / ((C - 1) * (P - 1))
    h = float((1.0 / n).mean())
    return {"child": (ms_c - ms_r) / P, "paradigm": (ms_p - ms_r) / C, "cxp": ms_r - h * s2e, "residual": s2e}


def proportions_from_stats(A, n, ssw) -> dict[str, np.ndarray]:
    v = {k: np.clip(x, 0, None) for k, x in mom_from_stats(A, n, ssw).items()}
    tot = sum(v.values())
    with np.errstate(invalid="ignore", divide="ignore"):
        return {f"prop_{k}": x / tot for k, x in v.items()}


def mom(long: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """MoM components for every feature column of ``long`` (rows: child, paradigm, block; complete design)."""
    A, n, ssw, _ = mom_stats(long, features)
    out = pd.DataFrame(mom_from_stats(A, n, ssw), index=features)
    out["n_truncated"] = (out[list(TERMS)] < 0).sum(axis=1)
    out[list(TERMS)] = out[list(TERMS)].clip(lower=0)
    return _proportions(out)


def _proportions(v: pd.DataFrame) -> pd.DataFrame:
    tot = v[list(TERMS)].sum(axis=1)
    for t in TERMS:
        v[f"prop_{t}"] = v[t] / tot.where(tot > 0)
    v["consistency"] = v["child"] / (v["child"] + v["cxp"]).where((v["child"] + v["cxp"]) > 0)
    return v


def reml(long: pd.DataFrame, feature: str) -> dict:
    """REML with crossed random effects (one group) for one feature; returns variances or raises."""
    import statsmodels.api as sm

    d = long[["child", "paradigm", feature]].dropna().copy()
    d["cxp"] = d["child"] + ":" + d["paradigm"]
    d["one"] = 1
    y = d[feature].to_numpy(float)
    scale = y.std() or 1.0
    d["y"] = (y - y.mean()) / scale
    vc = {"child": "0 + C(child)", "paradigm": "0 + C(paradigm)", "cxp": "0 + C(cxp)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sm.MixedLM.from_formula("y ~ 1", groups="one", vc_formula=vc, re_formula="0", data=d)
        r = m.fit(reml=True, method=["lbfgs"])
    names = list(m.exog_vc.names)
    out = {k: float(r.vcomp[names.index(k)]) * scale**2 for k in ("child", "paradigm", "cxp")}
    out["residual"] = float(r.scale) * scale**2
    out["converged"] = bool(r.converged)
    return out


def by_pattern(long: pd.DataFrame, features: list[str]) -> list[list[str]]:
    """Features grouped by identical missing rows, in feature order. Each group is estimated on its own complete
    rows, so a value missing in one channel never removes a child from another channel."""
    miss = long[features].isna().to_numpy()
    groups: dict[bytes, list[str]] = {}
    for j, f in enumerate(features):
        groups.setdefault(miss[:, j].tobytes(), []).append(f)
    return list(groups.values())


def complete_long(wide: pd.DataFrame, children: list[str], paradigms: list[str]) -> pd.DataFrame:
    d = wide[wide.child.isin(children) & wide.paradigm.isin(paradigms)]
    have = d.groupby("child")["paradigm"].nunique()
    keep = have.index[have == len(paradigms)]
    return d[d.child.isin(keep)].reset_index(drop=True)


def bootstrap_props(A, n, ssw, n_boot: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Bootstrap over children (with replacement) of the MoM proportions: ``(n_boot, F)`` per term."""
    C = A.shape[0]
    out: dict[str, list] = {}
    for _ in range(n_boot):
        i = rng.integers(0, C, C)
        for k, v in proportions_from_stats(A[i], n[i], ssw[i]).items():
            out.setdefault(k, []).append(v)
    return {k: np.stack(v) for k, v in out.items()}


class VarianceComponents(Analysis):
    """Components per channel-level feature, region-of-interest tests (H1b) and the group comparison."""

    label = "confirmatory"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.paradigms = list(cfg.dataset.seated_paradigms)

    def h1b(self, scalar: dict[str, pd.DataFrame], n_boot: int | None = None) -> pd.DataFrame:
        """``scalar``: name -> long table (child, paradigm, block, value) of the ROI-averaged H1b features."""
        n_boot = self.cfg.variance.n_bootstrap if n_boot is None else n_boot
        rows = []
        for name, d in scalar.items():
            d = d.rename(columns={"value": name})
            mm = mom(d, [name]).iloc[0]
            try:
                rv = reml(d, name)
                if not rv["converged"]:
                    raise RuntimeError("not converged")
                est = _proportions(pd.DataFrame([rv]))
                est_src = "reml"
            except (
                Exception
            ) as e:  # noqa: BLE001 - REML failure or non-convergence: MoM (logged)
                log.warning("REML failed for %s (%s); MoM used", name, e)
                est, est_src = pd.DataFrame([mm]), "mom"
            rng, seed = generator(self.cfg.project.seed, "h1b", name)
            bp = bootstrap_props(*mom_stats(d, [name])[:3], n_boot, rng)
            draws = (bp["prop_child"] - bp["prop_paradigm"])[:, 0]
            e0 = est.iloc[0]
            lo, hi = percentile_ci(draws, self.cfg.stats.ci_level)
            p = float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))
            rows.append(
                {
                    "feature": name,
                    "estimator": est_src,
                    "n_children": int(d.child.nunique()),
                    **{f"prop_{t}": float(e0[f"prop_{t}"]) for t in TERMS},
                    "mom_prop_child": float(mm["prop_child"]),
                    "mom_prop_paradigm": float(mm["prop_paradigm"]),
                    "child_minus_paradigm": float(e0["prop_child"] - e0["prop_paradigm"]),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p_boot": max(p, 1 / n_boot),
                    "seed": seed,
                }
            )
        out = pd.DataFrame(rows)
        from stats.inference import holm

        out["p_holm"] = holm(out["p_boot"].to_numpy())
        # H1b: lower bound > 0, with Holm across the three features
        out["supported"] = (out["ci_lo"] > 0) & (out["p_holm"] < 1 - self.cfg.stats.ci_level)
        return out

    def maps(self, family: str, wide: pd.DataFrame, children: list[str]) -> pd.DataFrame:
        long = complete_long(wide, children, self.paradigms)
        feats = [c for c in long.columns if c not in ("child", "paradigm", "block")]
        parts = []
        for fs in by_pattern(long, feats):
            sub = long.loc[long[fs[0]].notna(), ["child", "paradigm", "block", *fs]].reset_index(drop=True)
            n_kids = len(mom_stats(sub, fs[:1])[3])
            if n_kids < 3:  # too few complete children for this pattern: not estimated
                continue
            v = mom(sub, fs)
            v["n_children"] = n_kids
            parts.append(v)
        v = pd.concat(parts).reindex(feats)
        v.insert(0, "family", family)
        return v.reset_index(names="feature")

    def group_comparison(
        self, family: str, wide: pd.DataFrame, groups: pd.Series, n_boot: int | None = None
    ) -> pd.DataFrame:
        """ASD - TD in the child x paradigm proportion per feature, bootstrap within groups, BH within family."""
        n_boot = self.cfg.variance.n_bootstrap if n_boot is None else n_boot
        est, draws, ns = {}, {}, {}
        rng, seed = generator(self.cfg.project.seed, "vc_group", family)
        feats = [c for c in wide.columns if c not in ("child", "paradigm", "block")]
        for g in ("ASD", "TD", "SIB"):
            kids = groups.index[groups == g].tolist()
            long = complete_long(wide, kids, self.paradigms)
            e, d, nn = {}, {}, {}
            # each missing-value pattern on its own complete rows; with a single pattern this is the complete-row estimate
            for fs in by_pattern(long, feats):
                sub = long.loc[long[fs[0]].notna(), ["child", "paradigm", "block", *fs]].reset_index(
                    drop=True
                )
                st = mom_stats(sub, fs)
                if len(st[3]) < 3:  # too few complete children for this pattern: not estimated
                    continue
                e.update(mom(sub, fs)["prop_cxp"].to_dict())
                nn.update(dict.fromkeys(fs, len(st[3])))
                if g in ("ASD", "TD"):
                    b = bootstrap_props(*st[:3], n_boot, rng)["prop_cxp"]
                    d.update({f: b[:, j] for j, f in enumerate(fs)})
            est[g] = pd.Series(e).reindex(feats)
            ns[g] = pd.Series(nn).reindex(feats)
            if g in ("ASD", "TD"):
                draws[g] = np.column_stack([d.get(f, np.full(n_boot, np.nan)) for f in feats])
        diff = draws["ASD"] - draws["TD"]
        p = 2 * np.minimum((diff <= 0).mean(0), (diff >= 0).mean(0))
        p = np.maximum(p, 1 / n_boot)
        lo, hi = np.quantile(diff, [0.025, 0.975], axis=0)
        out = pd.DataFrame(
            {
                "family": family,
                "feature": feats,
                "prop_cxp_ASD": est["ASD"].to_numpy(),
                "prop_cxp_TD": est["TD"].to_numpy(),
                "prop_cxp_SIB": est["SIB"].to_numpy(),
                "asd_minus_td": (est["ASD"] - est["TD"]).to_numpy(),
                "n_ASD": ns["ASD"].to_numpy(),
                "n_TD": ns["TD"].to_numpy(),
                "n_SIB": ns["SIB"].to_numpy(),
                "ci_lo": lo,
                "ci_hi": hi,
                "p_boot": p,
                "seed": seed,
            }
        )
        out["q_bh"] = bh(out["p_boot"].to_numpy())
        return out

    def run(self, **inputs) -> Result:
        tables = {}
        if "scalar" in inputs:
            tables["h1b"] = self.h1b(inputs["scalar"])
        maps = [self.maps(f, w, inputs["core_children"]) for f, w in inputs.get("map_families", {}).items()]
        if maps:
            tables["maps"] = pd.concat(maps, ignore_index=True)
        grp = [
            self.group_comparison(f, w, inputs["groups"]) for f, w in inputs.get("group_families", {}).items()
        ]
        if grp:
            tables["group_comparison"] = pd.concat(grp, ignore_index=True)
        summ = {}
        if "h1b" in tables:
            summ["h1b"] = tables["h1b"][
                ["feature", "child_minus_paradigm", "ci_lo", "ci_hi", "supported"]
            ].to_dict("records")
        return Result("variance_components", tables, summ, label="confirmatory")


def main() -> None:
    """Variance components of the core set (posterior ROI features, channel maps, connectivity summaries)."""
    from config.loader import load_config
    from utils.inputs import Inputs
    from utils.store import ResultsStore

    cfg = load_config()
    inp = Inputs(cfg)
    core, meta = inp.core(), inp.meta()
    vc = VarianceComponents(cfg)
    maps = {f: inp.bank.wide(f) for f in ("aperiodic", "periodic", "alpha_cf")}
    maps["band_logpow"] = inp.band_logpow()
    groups = meta.loc[core, "group"]
    res = vc.run(
        scalar=inp.roi_scalar(core),
        map_families=maps,
        core_children=core,
        group_families=maps,
        groups=groups,
    )
    tang = [
        vc.maps(f, inp.bank.wide(f), core)
        for f in (
            "tangent_theta",
            "tangent_alpha",
            "tangent_beta",
            "wpli_theta",
            "wpli_alpha",
            "wpli_beta",
        )
    ]
    res.tables["connectivity_summary"] = pd.concat(tang, ignore_index=True)
    ResultsStore(cfg).write(res, "confirmatory")


if __name__ == "__main__":
    main()
