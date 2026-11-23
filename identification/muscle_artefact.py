"""Muscle artefact and the individual signature (Fig. 2a; Additional file 1, Table S6).

1. EMG index per recording: mean log10 power at 25-34 Hz over temporal channels (FT7, FT8, T7, T8, TP7, TP8) minus
   the same over central channels (C1, Cz, C2, C3, C4), mean over blocks; its between-child share of variance
   (random-intercept model over paradigms), the autistic-TD difference (OLS with HC3 errors, age and sex) and its
   correlation with self-identifiability.
2. Identification from 2-34, 2-20, 2-13 and 21-34 Hz, and from log power after regressing each child's EMG index
   out of every feature within paradigm.
3. Preprocessing variant in which every independent component whose top ICLabel class is muscle is removed
   (``config/muscle_variant.yaml``; preprocessed as variant ``muscle_argmax`` with the primary ICA reused):
   identification of log power, beta covariance, periodic, aperiodic and broadband-covariance features against
   the primary bank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from utils import common as C

TEMP = ["FT7", "FT8", "T7", "T8", "TP7", "TP8"]
CENT = ["C1", "Cz", "C2", "C3", "C4"]
VARIANT_BANK = C.DERIV / "features" / "muscle_argmax" / "bank"  # bank of config/muscle_variant.yaml


def emg_index(bank_dir: Path) -> pd.DataFrame:
    w = pd.read_parquet(bank_dir / "spectral_logpow.parquet")
    hz = range(25, 35)
    t = w[[f"{c}_{h}Hz" for c in TEMP for h in hz]].mean(axis=1)
    c = w[[f"{c}_{h}Hz" for c in CENT for h in hz]].mean(axis=1)
    return w[["child", "paradigm"]].assign(emg=t - c).groupby(["child", "paradigm"]).emg.mean().reset_index()


def family(bank_dir: Path, name: str) -> dict:
    w = pd.read_parquet(bank_dir / f"{name}.parquet")
    feats = [c for c in w.columns if c not in ("child", "paradigm", "block")]
    return {p: d.groupby("child")[feats].mean() for p, d in w.groupby("paradigm")}


def main() -> None:
    out = C.out_dir("muscle_artefact")
    m = C.meta()
    e = emg_index(C.BANK)
    # between-child share (ICC(1)) over paradigms
    piv = e.pivot(index="child", columns="paradigm", values="emg")
    fit = smf.mixedlm("emg ~ paradigm", e, groups=e.child).fit(reml=True)
    icc = float(fit.cov_re.iloc[0, 0] / (fit.cov_re.iloc[0, 0] + fit.scale))
    child = piv.mean(axis=1).rename("emg").to_frame().join(m[["group", "age", "sex_male"]])
    at = child[child.group.isin(["ASD", "TD"])].assign(asd=lambda d: (d.group == "ASD").astype(int)).dropna()
    g = smf.ols("emg ~ asd + age + sex_male", at).fit(cov_type="HC3")
    sd_td = at.loc[at.asd == 0, "emg"].std()
    si = pd.read_parquet(C.OUT / "self_identifiability_correlates/self_identifiability_all_available.parquet")
    s = si[["si"]].join(child.emg).dropna()
    rho = stats.spearmanr(s.si, s.emg)
    summ = {
        "emg_icc_between_child": icc,
        "asd_minus_td_sd": float(g.params["asd"] / sd_td),
        "asd_minus_td_ci_sd": [float(x / sd_td) for x in g.conf_int().loc["asd"]],
        "asd_minus_td_p": float(g.pvalues["asd"]),
        "rho_emg_self_ident": float(rho.statistic),
        "rho_p": float(rho.pvalue),
        "n_children": int(len(s)),
    }
    print(summ, flush=True)

    rep = C.bank_family("spectral_logpow")
    names = list(dict.fromkeys(c.rsplit("_", 1)[0] for c in rep[C.PARADIGMS[0]].columns))
    rows = []
    for label, lo, hi in (("2-34 Hz", 2, 34), ("2-20 Hz", 2, 20), ("2-13 Hz", 2, 13), ("21-34 Hz", 21, 34)):
        cols = [f"{n}_{h}Hz" for n in names for h in range(lo, hi + 1)]
        r = C.identify(rep, columns=cols)
        rows.append(
            {
                "analysis": "primary log power",
                "band": label,
                "accuracy": r["accuracy"],
                "diff_ident": r["diff_ident"],
            }
        )
    reg = {}
    for p, X in rep.items():
        ep = e[e.paradigm == p].set_index("child").emg.reindex(X.index)
        ok = ep.notna()
        Xr = X.astype(float)
        D = np.column_stack([np.ones(ok.sum()), ep[ok].to_numpy()])
        beta, *_ = np.linalg.lstsq(D, X[ok].to_numpy(float), rcond=None)
        Xr.loc[ok] = X[ok].to_numpy(float) - D @ beta
        reg[p] = Xr
    r = C.identify(reg)
    rows.append(
        {
            "analysis": "primary log power, EMG index regressed out",
            "band": "2-34 Hz",
            "accuracy": r["accuracy"],
            "diff_ident": r["diff_ident"],
        }
    )

    if VARIANT_BANK.is_dir():
        ev = emg_index(VARIANT_BANK)
        both = e.merge(ev, on=["child", "paradigm"], suffixes=("_primary", "_muscle_argmax"))
        summ["emg_index_change_median"] = float((both.emg_muscle_argmax - both.emg_primary).median())
        summ["emg_index_primary_median"] = float(both.emg_primary.median())
        summ["emg_index_variant_median"] = float(both.emg_muscle_argmax.median())
        for fam in ("spectral_logpow", "tangent_beta", "periodic", "aperiodic", "broadband_cov"):
            for lab, bdir in (("primary", C.BANK), ("muscle_argmax", VARIANT_BANK)):
                r = C.identify(family(bdir, fam))
                rows.append(
                    {
                        "analysis": f"{fam} ({lab})",
                        "band": "family",
                        "accuracy": r["accuracy"],
                        "diff_ident": r["diff_ident"],
                    }
                )
        vb = family(VARIANT_BANK, "spectral_logpow")
        for label, lo, hi in (("2-20 Hz", 2, 20), ("21-34 Hz", 21, 34)):
            cols = [f"{n}_{h}Hz" for n in names for h in range(lo, hi + 1)]
            r = C.identify(vb, columns=cols)
            rows.append(
                {
                    "analysis": "muscle_argmax log power",
                    "band": label,
                    "accuracy": r["accuracy"],
                    "diff_ident": r["diff_ident"],
                }
            )
        # does removing EMG change the (null) group difference in temporal beta power?
        w = pd.read_parquet(VARIANT_BANK / "spectral_logpow.parquet")
        wp = pd.read_parquet(C.BANK / "spectral_logpow.parquet")
        tb = [f"{c}_{h}Hz" for c in TEMP for h in range(14, 31)]
        for lab, ww in (("primary", wp), ("muscle_argmax", w)):
            v = (
                ww.assign(tb=ww[tb].mean(axis=1))
                .groupby("child")
                .tb.mean()
                .rename("tb")
                .to_frame()
                .join(m[["group", "age", "sex_male"]])
            )
            v = v[v.group.isin(["ASD", "TD"])].assign(asd=lambda d: (d.group == "ASD").astype(int)).dropna()
            gg = smf.ols("tb ~ asd + age + sex_male", v).fit(cov_type="HC3")
            sd = v.loc[v.asd == 0, "tb"].std()
            summ[f"temporal_beta_asd_minus_td_sd_{lab}"] = float(gg.params["asd"] / sd)
            summ[f"temporal_beta_asd_minus_td_p_{lab}"] = float(gg.pvalues["asd"])
    T = pd.DataFrame(rows)
    T.to_parquet(out / "identification.parquet")
    print(T.round(4).to_string(index=False))
    C.write_json(out / "summary.json", summ)
    print(summ)


if __name__ == "__main__":
    main()
