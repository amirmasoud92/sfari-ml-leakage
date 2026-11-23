"""Reliability and diagnostic validity of the candidate markers (Methods, Reliability and validity of candidate
markers; Table 3; Fig. 4 marker panel; Fig. 5; Additional file 1, Supplementary Methods S4).

Seventeen markers: the data authors' published evoked indices (ASSR 40/27-Hz inter-trial phase coherence and
180-250 ms transient response, AVSRT multisensory ERP index and race-model gain, FAST N170 latency and face-inversion
lateralisation, illusory-contour effect, Beep-Flash alpha modulation), AVSRT reaction time and its variability, and
five spectral markers (posterior periodic alpha, alpha centre frequency, aperiodic exponent and offset at rest;
rest-minus-task periodic alpha).

Per marker, in all autistic and TD children with the marker:
- reliability: split-half Pearson r between the independent halves of :mod:`markers.marker_halves`, Spearman-Brown
  corrected; overall, in autistic and in TD children;
- validity: autistic - TD Hedges' g (full recording), and g adjusted for age and sex (OLS coefficient / residual SD,
  with the small-sample correction); single-marker AUC (Mann-Whitney, oriented autistic > TD); HC3 p and
  Benjamini-Hochberg q over the markers;
- error-corrected AUC = Phi(|g| / sqrt(2 r)) under equal-variance normality (1 when r <= 0), with a null from 2,000
  permutations of the group labels (reliability held at its observed value), because |g| is positive under the null;
- one joint child bootstrap (2,000 resamples) for reliability, effect sizes, AUC and error-corrected AUC;
- sample size per group for 80% power at the observed adjusted g (two-sided alpha .05).
Cross-paradigm stability of the spectral markers: ICC(2,1) and ICC(2,k) over the six paradigms (core set).
Marker panel: all 17 markers (missing values imputed with training means) in L2 logistic regression (C = 0.1),
20 x 5 child-level cross-validation stratified on group, within-fold AUC; 200 label permutations (20 x 5 CV each)
with folds re-drawn on the permuted labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stats.rng import generator
from utils import common as C

MARKERS = {
    "ASSR__itpc_40hz": "ASSR 40-Hz ITPC",
    "ASSR__itpc_27hz": "ASSR 27-Hz ITPC",
    "ASSR__broadband_180_250_40hz": "ASSR 40-Hz transient (180-250 ms)",
    "ASSR__broadband_180_250_27hz": "ASSR 27-Hz transient (180-250 ms)",
    "AVSRT__erp_msi_145_165": "Multisensory ERP index",
    "AVSRT__race_model_gain": "Race-model gain",
    "FAST__n170_latency_s": "N170 latency",
    "FAST__fie_lateralisation_uv": "Face-inversion lateralisation",
    "IC__ic_effect_uv": "Illusory-contour effect",
    "Beepflash__alpha_modulation_index": "Cue-locked alpha modulation",
    "AVSRT__rt_mean": "Reaction time",
    "AVSRT__rt_icv": "Reaction-time variability",
    "rest_periodic_alpha": "Resting periodic alpha",
    "rest_alpha_cf": "Resting alpha frequency",
    "rest_aperiodic_exponent": "Resting aperiodic exponent",
    "rest_aperiodic_offset": "Resting aperiodic offset",
    "alpha_rest_minus_task": "Alpha rest - task",
}
NB = 2000
NPERM = 2000


def sb(r):
    return 2 * r / (1 + r)


def hedges(a, b):
    n1, n0 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n0 - 1) * np.var(b, ddof=1)) / (n1 + n0 - 2))
    return (1 - 3 / (4 * (n1 + n0) - 9)) * (np.mean(a) - np.mean(b)) / sp


def adj_g(y, asd, age, sex):
    """Group coefficient / residual SD, with the small-sample correction J = 1 - 3 / (4 df - 1)."""
    X = sm.add_constant(np.column_stack([asd, age, sex]))
    f = sm.OLS(y, X).fit()
    return (1 - 3 / (4 * f.df_resid - 1)) * f.params[1] / np.sqrt(f.scale), f


def ceiling(g, r):
    """Error-corrected AUC Phi(|g| / sqrt(2 r)); 1 when r <= 0 (correction unbounded; no draw is dropped)."""
    return float(stats.norm.cdf(abs(g) / np.sqrt(2 * r))) if r > 0 else 1.0


def icc2(Y: np.ndarray) -> tuple[float, float]:
    """ICC(2,1) and ICC(2,k) (two-way random, absolute agreement) for an n x k matrix."""
    n, k = Y.shape
    gm = Y.mean()
    msr = k * np.sum((Y.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((Y.mean(0) - gm) ** 2) / (k - 1)
    sse = np.sum((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return ((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n), (msr - mse) / (msr + (msc - mse) / n))


def panel_auc(X: np.ndarray, y: np.ndarray, seed: int, repeats: int = 20) -> float:
    aucs = []
    for r in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y):
            mu = np.nanmean(X[tr], 0)
            Xtr = np.where(np.isnan(X[tr]), mu, X[tr])
            Xte = np.where(np.isnan(X[te]), mu, X[te])
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(Xtr, y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs))


def main() -> None:
    out = C.out_dir("reliability_validity")
    v = pd.read_parquet(C.OUT / "marker_halves/marker_values.parquet")
    m = C.meta()
    rng, seed = generator(C.SEED, "v2_b2")
    rows = []
    for mk, label in MARKERS.items():
        d = v[v.marker == mk].pivot_table(index="child", columns="half", values="value", aggfunc="mean")
        d = d.join(m[["group", "age", "sex_male"]]).dropna(subset=["group"])
        d = d[d.group.isin(["ASD", "TD"])]
        for col in (0, 1, -1):
            if col not in d.columns:
                d[col] = np.nan
        h = d.dropna(subset=[0, 1])
        if len(h) < 10:
            print(f"{label}: no split-half values", flush=True)
            continue
        f = d.dropna(subset=[-1, "age", "sex_male"])
        asd = (f.group == "ASD").to_numpy(float)
        y = f[-1].to_numpy(float)
        g = hedges(y[asd == 1], y[asd == 0])
        ga, fit = adj_g(y, asd, f.age.to_numpy(float), f.sex_male.to_numpy(float))
        p = (
            sm.OLS(y, sm.add_constant(np.column_stack([asd, f.age, f.sex_male])))
            .fit(cov_type="HC3")
            .pvalues[1]
        )
        auc = roc_auc_score(asd, y)
        r_all = np.corrcoef(h[0], h[1])[0, 1]
        r_grp = {
            gname: np.corrcoef(h[h.group == gname][0], h[h.group == gname][1])[0, 1]
            for gname in ("ASD", "TD")
        }
        # joint bootstrap over children (one resample of children feeds both reliability and effect size)
        bs = {k: [] for k in ("rsb", "rsb_asd", "rsb_td", "g", "ga", "auc", "ceil", "dr")}
        di_ = d.reset_index()
        has_h = di_[0].notna().to_numpy() & di_[1].notna().to_numpy()
        has_f = (di_[-1].notna() & di_.age.notna() & di_.sex_male.notna()).to_numpy()
        for _ in range(NB):
            idx = rng.integers(0, len(di_), len(di_))
            hb = di_.iloc[idx[has_h[idx]]]
            fb = di_.iloc[idx[has_f[idx]]]
            ab = (fb.group == "ASD").to_numpy(float)
            if ab.sum() < 3 or (1 - ab).sum() < 3:
                continue
            rb = sb(np.corrcoef(hb[0], hb[1])[0, 1])
            ra = sb(np.corrcoef(hb[hb.group == "ASD"][0], hb[hb.group == "ASD"][1])[0, 1])
            rt = sb(np.corrcoef(hb[hb.group == "TD"][0], hb[hb.group == "TD"][1])[0, 1])
            yb = fb[-1].to_numpy(float)
            gb = hedges(yb[ab == 1], yb[ab == 0])
            gab, _ = adj_g(yb, ab, fb.age.to_numpy(float), fb.sex_male.to_numpy(float))
            bs["rsb"].append(rb)
            bs["rsb_asd"].append(ra)
            bs["rsb_td"].append(rt)
            bs["dr"].append(ra - rt)
            bs["g"].append(gb)
            bs["ga"].append(gab)
            bs["auc"].append(roc_auc_score(ab, yb))
            bs["ceil"].append(ceiling(gab, rb))
        q = {k: np.nanpercentile(x, [2.5, 97.5]) for k, x in bs.items()}
        # label-permutation null of the error-corrected AUC (|g| is positive under the null); reliability
        # does not depend on the labels and is held at its observed value
        rs = sb(r_all)
        nul = np.array(
            [
                ceiling(
                    adj_g(y, rng.permutation(asd), f.age.to_numpy(float), f.sex_male.to_numpy(float))[0], rs
                )
                for _ in range(NPERM)
            ]
        )
        n80 = (
            int(np.ceil(2 * (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) ** 2 / ga**2))
            if abs(ga) > 0.01
            else np.nan
        )
        rows.append(
            {
                "marker": mk,
                "label": label,
                "n_halves": len(h),
                "n_asd": int(asd.sum()),
                "n_td": int((1 - asd).sum()),
                "r_split": r_all,
                "r_sb": sb(r_all),
                "r_sb_lo": q["rsb"][0],
                "r_sb_hi": q["rsb"][1],
                "r_sb_asd": sb(r_grp["ASD"]),
                "r_sb_td": sb(r_grp["TD"]),
                "r_sb_diff_lo": q["dr"][0],
                "r_sb_diff_hi": q["dr"][1],
                "g": g,
                "g_lo": q["g"][0],
                "g_hi": q["g"][1],
                "g_adj": ga,
                "g_adj_lo": q["ga"][0],
                "g_adj_hi": q["ga"][1],
                "p": p,
                "auc": auc,
                "auc_lo": q["auc"][0],
                "auc_hi": q["auc"][1],
                "auc_normal": stats.norm.cdf(abs(ga) / np.sqrt(2)),
                "auc_ceiling": ceiling(ga, sb(r_all)),
                "auc_ceiling_lo": q["ceil"][0],
                "auc_ceiling_hi": q["ceil"][1],
                "auc_ceiling_null_median": float(np.median(nul)),
                "auc_ceiling_null_95": float(np.percentile(nul, 95)),
                "p_auc_ceiling": float((1 + (nul >= ceiling(ga, rs)).sum()) / (1 + len(nul))),
                "n_per_group_80": n80,
            }
        )
        print(
            f"{label:35s} rSB {sb(r_all):.2f}  g_adj {ga:+.2f}  AUC {auc:.2f}  ceiling {ceiling(ga, sb(r_all)):.2f}",
            flush=True,
        )
    T = pd.DataFrame(rows)
    p_ = T.p.to_numpy()
    o = np.argsort(p_)
    qv = np.minimum.accumulate((p_[o] * len(p_) / np.arange(1, len(p_) + 1))[::-1])[::-1]
    T.loc[T.index[o], "q_bh"] = np.minimum(qv, 1)
    T.to_parquet(out / "markers.parquet")

    # cross-paradigm stability of spectral markers (core set)
    from utils.inputs import Inputs

    core = Inputs(C.CFG).core()
    sp = pd.read_parquet(C.OUT / "marker_halves/spectral_by_paradigm.parquet")
    icc = []
    for mk, d in sp.groupby("marker"):
        Y = d.pivot(index="child", columns="paradigm", values="v").reindex(core).dropna().to_numpy(float)
        a, k = icc2(Y)
        icc.append({"marker": mk, "n": len(Y), "icc_2_1": a, "icc_2_k": k})
    pd.DataFrame(icc).to_parquet(out / "cross_paradigm_icc.parquet")
    print(pd.DataFrame(icc).round(3).to_string(index=False), flush=True)

    # marker panel
    full = v[v.half == -1].pivot_table(index="child", columns="marker", values="value", aggfunc="mean")[
        list(MARKERS)
    ]
    full = full.join(m.group).query("group in ['ASD', 'TD']")
    X, yy = full[list(MARKERS)].to_numpy(float), (full.group == "ASD").to_numpy(int)
    obs = panel_auc(X, yy, seed % 100000)
    null = Parallel(n_jobs=30)(
        delayed(panel_auc)(X, rng.permutation(yy), int(s) % 100000, 20) for s in rng.integers(0, 2**31, 200)
    )
    null = np.array(null)
    pan = {
        "n": int(len(yy)),
        "n_asd": int(yy.sum()),
        "auc_within_fold": obs,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "p_perm": float((1 + (null >= obs).sum()) / (1 + len(null))),
        "seed": seed,
    }
    C.write_json(out / "panel.json", pan)
    print(pan)


if __name__ == "__main__":
    main()
