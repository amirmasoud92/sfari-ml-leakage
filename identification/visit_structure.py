"""Identification within and across visits (Additional file 1, Table S3, Supplementary Methods S2 and Fig. S2).

The start of the first run of every recording is read from the BDF header. Two recordings of a child are classed
as same-visit if they started on the same date; only this relative classification (a day index within child) and
the hours between recordings within a visit are stored, never a date.

1. Per child and pair type, the fraction of ordered pairs identified; within-child comparison of same-visit and
   other-visit identification in children with both pair types (Wilcoxon test, child-bootstrap interval).
2. Margin of each family over the covariance and external-electrode comparators on same-visit and other-visit
   pairs (child bootstrap, sign-flip p).
3. Self-similarity r_ii against the hours between two recordings within a visit (mixed model, random child
   intercept; Fig. S2c).
4. Mean position of each paradigm within the visit.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from identification import core
from identification import signature_decomposition as SD
from stats.inference import paired_bootstrap_ci, sign_flip_p
from stats.rng import generator
from utils import common as C

DATA = Path(str(C.CFG.paths.data_root))


def bdf_start(path: Path) -> dt.datetime:
    with open(path, "rb") as h:
        b = h.read(184)
    d, t = b[168:176].decode(), b[176:184].decode()
    dd, mm, yy = (int(x) for x in d.split("."))
    hh, mi, ss = (int(x) for x in t.split("."))
    return dt.datetime(2000 + yy if yy < 85 else 1900 + yy, mm, dd, hh, mi, ss)


def recording_times() -> pd.DataFrame:
    """Per kept seated recording: start of its first run, set of dates of its runs (as day indices)."""
    rows = []
    for child, par, _ in C.recordings():
        runs = sorted((DATA / child / "eeg").glob(f"{child}_task-{par}_run-*_eeg.bdf"))
        starts = [bdf_start(r) for r in runs]
        rows.append(
            {"child": child, "paradigm": par, "start": min(starts), "dates": {s.date() for s in starts}}
        )
    df = pd.DataFrame(rows)
    # day index within child (0 = first date); dates themselves are not written out
    out = []
    for c, d in df.groupby("child"):
        days = sorted(set().union(*d.dates))
        idx = {x: i for i, x in enumerate(days)}
        t0 = d.start.min()
        for r in d.itertuples():
            out.append(
                {
                    "child": c,
                    "paradigm": r.paradigm,
                    "day": idx[r.start.date()],
                    "n_days_recording": len(r.dates),
                    "hours_from_first": (r.start - t0).total_seconds() / 3600,
                    "n_days_child": len(days),
                }
            )
    return pd.DataFrame(out)


def build_reps() -> dict:
    """The representations compared here: feature-bank families plus the decomposition of the covariance
    comparator (identification.signature_decomposition)."""
    reps = {
        f: C.bank_family(f)
        for f in (
            "spectral_logpow",
            "broadband_cov",
            "exg_only",
            "aperiodic",
            "periodic",
            "tangent_theta",
            "tangent_alpha",
            "tangent_beta",
            "wpli_alpha",
        )
    }
    recs = C.recordings()

    blocks = [r for x in recs for r in SD.load_one(*x)]
    keys = [{k: b[k] for k in ("child", "paradigm", "block")} for b in blocks]
    names = [str(x) for x in np.load(recs[0][2])["eeg_names"]]
    LP = np.stack([b["logpow"] for b in blocks])
    shape = LP - LP.mean(axis=2, keepdims=True)
    reps["logpow_shape"] = C.block_mean(
        keys, shape.reshape(len(LP), -1), [f"f{i}" for i in range(shape[0].size)]
    )
    S = np.stack([b["S_broadband"] for b in blocks])
    reps["bb_logvar"] = C.block_mean(keys, np.log10(np.einsum("nii->ni", S)), names)
    T = SD.tangent(np.stack([SD.corr_from_cov(s) for s in S]))
    reps["bb_corr_tangent"] = C.block_mean(keys, T, [f"t{i}" for i in range(T.shape[1])])
    L = SD.csd_operator(names)
    Sc = L @ S @ L.T
    Tc = SD.tangent(np.stack([SD.corr_from_cov(s) for s in Sc]))
    reps["bb_corr_tangent_csd"] = C.block_mean(keys, Tc, [f"t{i}" for i in range(Tc.shape[1])])
    return reps


def self_similarity(rep: dict, src: str, tgt: str) -> pd.Series:
    kids = sorted(set(rep[src].index) & set(rep[tgt].index))
    A = core.zscore_columns(rep[src].loc[kids].to_numpy(float))
    B = core.zscore_columns(rep[tgt].loc[kids].to_numpy(float))
    return pd.Series(np.diag(core.corr_matrix(A, B)), index=kids)


def main() -> None:
    out = C.out_dir("visit_structure")
    times = recording_times()
    times.to_parquet(out / "recording_times_relative.parquet")
    per_child = times.groupby("child").agg(
        n_days=("n_days_child", "first"), span_h=("hours_from_first", "max")
    )
    same_day_children = per_child[per_child.n_days == 1]
    order = (
        times[times.n_days_child == 1]
        .sort_values(["child", "hours_from_first"])
        .assign(position=lambda d: d.groupby("child").cumcount() + 1)
        .groupby("paradigm")
        .position.agg(["mean", "std", "count"])
    )
    print(
        per_child.n_days.value_counts().to_dict(),
        "single-day span h median",
        round(same_day_children.span_h.median(), 2),
        flush=True,
    )
    print(order.round(2), flush=True)

    tix = times.set_index(["child", "paradigm"])
    reps = build_reps()
    rng, seed = generator(C.SEED, "committee_a4")
    corr = {}
    for name, rep in reps.items():
        corr[name] = C.identify(rep)["correct"]
    # pair type per (child, ordered pair)
    pc = corr["spectral_logpow"]
    ptype = pd.DataFrame(index=pc.index, columns=pc.columns, dtype=object)
    gap = pd.DataFrame(index=pc.index, columns=pc.columns, dtype=float)
    for col in pc.columns:
        s, t = col.split(">")
        for k in pc.index[pc[col].notna()]:
            a, b = tix.loc[(k, s)], tix.loc[(k, t)]
            ptype.loc[k, col] = "same_day" if a.day == b.day else "cross_day"
            gap.loc[k, col] = abs(a.hours_from_first - b.hours_from_first)

    rows, both_rows = [], []
    for name, cm in corr.items():
        cm = cm.reindex(index=pc.index, columns=pc.columns)
        for pt in ("same_day", "cross_day"):
            m = cm.where(ptype == pt)
            rate = m.astype(float).mean(axis=1).dropna()
            rows.append(
                {
                    "representation": name,
                    "pair_type": pt,
                    "n_children": len(rate),
                    "n_pairs": int(m.notna().sum().sum()),
                    "accuracy_pairs": float(np.nanmean(m.astype(float))),
                    "mean_child_rate": rate.mean(),
                }
            )
        sd = cm.where(ptype == "same_day").astype(float).mean(axis=1)
        cd = cm.where(ptype == "cross_day").astype(float).mean(axis=1)
        ok = sd.notna() & cd.notna()
        d = (sd[ok] - cd[ok]).to_numpy()
        lo, hi = paired_bootstrap_ci(d, 2000, rng, 0.95)
        both_rows.append(
            {
                "representation": name,
                "n_children_both": int(ok.sum()),
                "same_day": sd[ok].mean(),
                "cross_day": cd[ok].mean(),
                "same_minus_cross": d.mean(),
                "ci_lo": lo,
                "ci_hi": hi,
                "wilcoxon_p": stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0,
            }
        )
    acc = pd.DataFrame(rows)
    within = pd.DataFrame(both_rows)
    acc.to_parquet(out / "accuracy_by_pair_type.parquet")
    within.to_parquet(out / "within_child_same_vs_cross.parquet")
    print(acc.round(3).to_string(index=False), flush=True)
    print(within.round(3).to_string(index=False), flush=True)

    # comparator margins by pair type (per child, all children with that pair type)
    marg = []
    for fam in ("spectral_logpow", "logpow_shape", "tangent_theta", "tangent_alpha", "aperiodic", "periodic"):
        for floor in ("broadband_cov", "bb_corr_tangent", "exg_only"):
            for pt in ("same_day", "cross_day"):
                a = corr[fam].reindex(index=pc.index, columns=pc.columns).where(ptype == pt).astype(float)
                b = corr[floor].reindex(index=pc.index, columns=pc.columns).where(ptype == pt).astype(float)
                ok = a.notna() & b.notna()
                da = a.where(ok).mean(axis=1) - b.where(ok).mean(axis=1)
                da = da.dropna().to_numpy()
                lo, hi = paired_bootstrap_ci(da, 2000, rng, 0.95)
                marg.append(
                    {
                        "family": fam,
                        "floor": floor,
                        "pair_type": pt,
                        "n_children": len(da),
                        "margin": da.mean(),
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "p_signflip": sign_flip_p(da, 10000, rng),
                    }
                )
    M = pd.DataFrame(marg)
    M.to_parquet(out / "floor_margin_by_pair_type.parquet")
    print(M.round(3).to_string(index=False), flush=True)

    # self-similarity against hours between recordings (same-day pairs, unordered)
    import itertools

    import statsmodels.formula.api as smf

    dec = []
    for name in (
        "spectral_logpow",
        "broadband_cov",
        "bb_corr_tangent",
        "logpow_shape",
        "aperiodic",
        "periodic",
    ):
        recs = []
        for s, t in itertools.combinations(C.PARADIGMS, 2):
            r = self_similarity(reps[name], s, t)
            for k, v in r.items():
                a, b = tix.loc[(k, s)], tix.loc[(k, t)]
                if a.day == b.day:
                    recs.append(
                        {
                            "child": k,
                            "pair": f"{s}|{t}",
                            "r": v,
                            "hours": abs(a.hours_from_first - b.hours_from_first),
                        }
                    )
        D = pd.DataFrame(recs)
        fit = smf.mixedlm("r ~ hours + pair", D, groups=D.child).fit(reml=True)
        dec.append(
            {
                "representation": name,
                "n_obs": len(D),
                "slope_r_per_hour": fit.params["hours"],
                "ci_lo": fit.conf_int().loc["hours", 0],
                "ci_hi": fit.conf_int().loc["hours", 1],
                "p": fit.pvalues["hours"],
                "median_gap_h": D.hours.median(),
            }
        )
    T = pd.DataFrame(dec)
    T.to_parquet(out / "self_similarity_decay.parquet")
    print(T.round(4).to_string(index=False), flush=True)
    order.to_parquet(out / "paradigm_order.parquet")
    C.write_json(
        out / "summary.json",
        {
            "n_children": int(len(per_child)),
            "children_by_n_days": per_child.n_days.value_counts().to_dict(),
            "single_day_span_h_median": float(same_day_children.span_h.median()),
            "recordings_spanning_two_dates": int((times.n_days_recording > 1).sum()),
            "seed": seed,
        },
    )


if __name__ == "__main__":
    main()
