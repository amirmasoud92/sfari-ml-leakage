"""Published group effects recomputed with the data authors' definitions and without tuning (Additional file 1,
Supplementary Methods S4 and Table S15). Per-child measures are computed first and joined with the group labels
afterwards.

| check | published (direction compared) |
|---|---|
| ASSR | 180-250 ms broadband response less negative in ASD: ASD - TD = +1.49 uV (40 Hz), +1.21 uV (27 Hz) |
| AVSRT | ERP MSI AV - (A+V) at 145-165 ms present (> 0) in TD, absent in ASD and SIB; group F = 6.28 |
| FAST | face-inversion effect right-lateralised in TD (R - L < 0), bilateral in ASD (ASD - TD > 0) |
| Rest | periodic alpha ordinal trend ASD < SIB < TD, beta = +0.057 |
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from config.loader import Config
from features.signals import CleanRecording, fit_specparam, largest_peak, welch_from_segments
from utils.base import Analysis, Result
from utils.dataset import BidsDataset, Participants
from utils.logging_utils import get_logger

log = get_logger(__name__)

PUBLISHED = {
    "assr_40": {"estimate": 1.491, "text": "ASD - TD = +1.49 uV (TD - ASD beta = -1.491, p = .019)"},
    "assr_27": {"estimate": 1.208, "text": "ASD - TD = +1.21 uV (TD - ASD beta = -1.208, p = .032)"},
    "avsrt_msi": {
        "estimate": None,
        "text": "MSI > 0 in TD only; group F = 6.28, p = .003; TD > ASD, TD > SIB",
    },
    "fast_fie_lat": {
        "estimate": None,
        "text": "FIE right-lateralised in TD, bilateral in ASD (ASD - TD > 0 in R - L)",
    },
    "rest_periodic_alpha": {"estimate": 0.057, "text": "ordinal beta = 0.057, p = .04 (ASD < SIB < TD)"},
}


def _assr(cfg: Config, child: str) -> list[dict]:
    r = CleanRecording(cfg, child, "ASSR")
    if not r.usable:
        return []
    e = cfg.features.evoked["assr"]
    chans, ref = list(e["channels"]), list(e["reference"])
    out = []
    for freq in ("27", "40"):
        tr = r.trials(f"^{freq}_Hz_Standard$")
        E, t = r.epochs(tr["sample"].to_numpy(), -0.2, 0.8, chans + ref)
        if len(E) < 20:
            continue
        E = E[:, : len(chans)] - E[:, len(chans) :].mean(axis=1, keepdims=True)
        bm = (t >= e["baseline_s"][0]) & (t <= e["baseline_s"][1])
        E = E - E[:, :, bm].mean(axis=2, keepdims=True)
        w = (t >= e["broadband_window_s"][0]) & (t <= e["broadband_window_s"][1])
        amp = E.mean(axis=0)[:, w].mean(axis=1) * 1e6
        out += [
            {"child": child, "freq": freq, "channel": c, "amp_uv": float(a), "n_trials": len(E)}
            for c, a in zip(chans, amp, strict=True)
        ]
    return out


def _responded(r: CleanRecording, onsets: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Published AVSRT inclusion: first response 100-1500 ms after onset and <= 2 responses before the next trial."""
    resp = np.sort(r.events.loc[r.events["trial_type"] == "Response_button", "sample"].to_numpy(np.int64))
    allon = np.sort(r.trials(artefact_free=False)["sample"].to_numpy(np.int64))
    sf = r.sfreq
    ok = np.zeros(onsets.size, bool)
    for i, s in enumerate(onsets):
        nxt = allon[allon > s]
        stop = nxt[0] if nxt.size else s + int(hi * sf)
        rs = resp[(resp > s) & (resp < stop)]
        ok[i] = rs.size > 0 and lo * sf <= rs[0] - s <= hi * sf and rs.size <= 2
    return ok


def _avsrt(cfg: Config, child: str) -> list[dict]:
    r = CleanRecording(cfg, child, "AVSRT")
    if not r.usable:
        return []
    e = cfg.features.evoked["avsrt_msi"]
    lo, hi = cfg.features.evoked["rt_window_s"]
    erp = {}
    n = {}
    for cond, tt in (("A", "Auditory_stim"), ("V", "Visual_stim"), ("AV", "Audiovisual_stim")):
        tr = r.trials(f"^{tt}$")
        s = tr["sample"].to_numpy(np.int64)
        s = s[_responded(r, s, lo, hi)]
        E, t = r.epochs(s, -0.1, 0.4, list(e["channels"]), tuple(e["baseline_s"]))
        if len(E) < 20:
            return []
        w = (t >= e["window_s"][0]) & (t <= e["window_s"][1])
        erp[cond] = float(E.mean(axis=0)[:, w].mean() * 1e6)
        n[cond] = len(E)
    return [
        {
            "child": child,
            "msi_uv": erp["AV"] - (erp["A"] + erp["V"]),
            **{f"erp_{k}": v for k, v in erp.items()},
            **{f"n_{k}": v for k, v in n.items()},
        }
    ]


def _fast(cfg: Config, child: str) -> list[dict]:
    r = CleanRecording(cfg, child, "FAST")
    if not r.usable:
        return []
    e = cfg.features.evoked["fast_n170"]
    chans = [e["channels"]["left"][0], e["channels"]["right"][0]]
    amps, n = {}, {}
    for cond in ("upright", "inverted"):
        tr = r.trials(f"^Face_{cond}$")
        E, t = r.epochs(tr["sample"].to_numpy(), -0.2, 0.5, chans, tuple(e["baseline_s"]))
        if len(E) < 20:
            return []
        w = (t >= e["amplitude_window_s"][0]) & (t <= e["amplitude_window_s"][1])
        amps[cond] = E.mean(axis=0)[:, w].mean(axis=1) * 1e6
        n[cond] = len(E)
    fie = amps["inverted"] - amps["upright"]
    return [
        {
            "child": child,
            "fie_left_uv": float(fie[0]),
            "fie_right_uv": float(fie[1]),
            "fie_lat_uv": float(fie[1] - fie[0]),
            "n_upright": n["upright"],
            "n_inverted": n["inverted"],
        }
    ]


def _rest(cfg: Config, child: str) -> list[dict]:
    r = CleanRecording(cfg, child, "Restingstate")
    if not r.usable:
        return []
    chans = list(cfg.reproduction.rest_channels)
    out = []
    blocks = r.blocks.sort_values("block")
    picked = 0
    for _, b in blocks.iterrows():
        S = r.segments(b, chans, seg_s=2.0, step_s=1.0)
        if len(S) < 25:  # at least ~26 s of clean 1-min block
            continue
        f, P = welch_from_segments(S, r.sfreq)
        fits = fit_specparam(f, P, cfg, "knee")
        for c, ft in zip(chans, fits, strict=True):
            if ft is None or ft.r2 < cfg.features.spectral.specparam.min_r2:
                continue
            cf, pw = largest_peak(ft, tuple(cfg.features.spectral.alpha_band))
            out.append(
                {
                    "child": child,
                    "block": int(b["block"]),
                    "order": picked,
                    "channel": c,
                    "periodic_alpha_pw": pw,
                    "alpha_cf": cf,
                    "r2": ft.r2,
                }
            )
        picked += 1
    return out


EXTRACTORS = {"assr": _assr, "avsrt": _avsrt, "fast": _fast, "rest": _rest}


def _ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(x.size))
    q = stats.t.ppf(0.5 + level / 2, x.size - 1)
    p = float(stats.ttest_1samp(x, 0).pvalue)
    return m, m - q * se, m + q * se, p


def _mixed(formula: str, data: pd.DataFrame, term: str) -> dict:
    import statsmodels.formula.api as smf

    try:
        m = smf.mixedlm(formula, data, groups=data["child"]).fit(reml=True, method=["lbfgs", "powell"])
        ci = m.conf_int().loc[term]
    except (np.linalg.LinAlgError, ValueError, KeyError) as e:  # recorded as a failed fit, never silent
        log.warning("mixed model failed (%s): %s", formula, e)
        return {
            "estimate": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "p": np.nan,
            "converged": False,
            "error": f"{type(e).__name__}: {e}",
        }
    return {
        "estimate": float(m.params[term]),
        "ci_lo": float(ci[0]),
        "ci_hi": float(ci[1]),
        "p": float(m.pvalues[term]),
        "converged": bool(m.converged),
    }


class ReproductionChecks(Analysis):
    """The four published-effect checks (module docstring)."""

    label = "confirmatory"
    needs_labels = True

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.ds = BidsDataset.from_config(cfg)
        self.parts = Participants.load(self.ds, cfg)

    def extract(self, children: list[str]) -> dict[str, pd.DataFrame]:
        out = {}
        for name, fn in EXTRACTORS.items():
            rows = Parallel(n_jobs=self.cfg.runtime.n_jobs)(delayed(fn)(self.cfg, c) for c in children)
            out[name] = pd.DataFrame([x for r in rows for x in r])
            log.info(
                "reproduction %s: %d children", name, out[name]["child"].nunique() if len(out[name]) else 0
            )
        return out

    def run(self, **inputs) -> Result:
        children = inputs.get("children") or self.parts.usable_ids
        per = self.extract(children)
        lab = self.parts.labels().reset_index()
        cov = self.parts.covariates().reset_index()[["participant_id", "age", "sex"]]
        meta = lab.merge(cov, on="participant_id").rename(
            columns={"participant_id": "child", "group_code": "group"}
        )
        meta = meta[["child", "group", "age", "sex"]]
        rows = []
        tables = {}

        # ASSR: LMM amp ~ group * channel (sum-coded) + age, random intercept; ASD - TD averaged over channels
        a = per["assr"].merge(meta, on="child")
        tables["assr_per_child"] = a
        for freq in ("40", "27"):
            d = a[a["freq"] == freq].dropna(subset=["age"])
            res = _mixed(
                "amp_uv ~ C(group, Treatment('TD')) * C(channel, Sum) + age",
                d,
                "C(group, Treatment('TD'))[T.ASD]",
            )
            rows.append(
                {
                    "check": f"assr_{freq}",
                    "contrast": "ASD - TD, 180-250 ms (uV)",
                    **res,
                    **_ns(d),
                    "published": PUBLISHED[f"assr_{freq}"]["text"],
                    "direction_match": res["estimate"] > 0,
                }
            )

        # AVSRT: MSI per group (one-sample) and one-way ANOVA
        v = per["avsrt"].merge(meta, on="child")
        tables["avsrt_per_child"] = v
        for g in ("TD", "ASD", "SIB"):
            m, lo, hi, p = _ci(v.loc[v.group == g, "msi_uv"])
            rows.append(
                {
                    "check": "avsrt_msi",
                    "contrast": f"MSI in {g} (uV)",
                    "estimate": m,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p": p,
                    f"n_{g}": int((v.group == g).sum()),
                    "published": PUBLISHED["avsrt_msi"]["text"],
                    "direction_match": (m > 0 and lo > 0) if g == "TD" else None,
                }
            )
        F, pF = stats.f_oneway(*[v.loc[v.group == g, "msi_uv"] for g in ("TD", "ASD", "SIB")])
        diff = v.loc[v.group == "TD", "msi_uv"].mean() - v.loc[v.group == "ASD", "msi_uv"].mean()
        rows.append(
            {
                "check": "avsrt_msi",
                "contrast": "group F (TD, ASD, SIB)",
                "estimate": float(F),
                "p": float(pF),
                **_ns(v),
                "published": PUBLISHED["avsrt_msi"]["text"],
                "direction_match": bool(diff > 0),
            }
        )

        # FAST: lateralisation R - L of FIE; ASD - TD with age (OLS), and per-group means
        f = per["fast"].merge(meta, on="child")
        tables["fast_per_child"] = f
        import statsmodels.formula.api as smf

        d = f[f.group.isin(["ASD", "TD"])].dropna(subset=["age"])
        ols = smf.ols("fie_lat_uv ~ C(group, Treatment('TD')) + age", d).fit()
        term = "C(group, Treatment('TD'))[T.ASD]"
        ci = ols.conf_int().loc[term]
        rows.append(
            {
                "check": "fast_fie_lat",
                "contrast": "ASD - TD in FIE lateralisation R - L (uV)",
                "estimate": float(ols.params[term]),
                "ci_lo": float(ci[0]),
                "ci_hi": float(ci[1]),
                "p": float(ols.pvalues[term]),
                **_ns(f),
                "published": PUBLISHED["fast_fie_lat"]["text"],
                "direction_match": float(ols.params[term]) > 0,
            }
        )
        for g in ("TD", "ASD", "SIB"):
            m, lo, hi, p = _ci(f.loc[f.group == g, "fie_lat_uv"])
            rows.append(
                {
                    "check": "fast_fie_lat",
                    "contrast": f"R - L in {g} (uV)",
                    "estimate": m,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p": p,
                    "published": PUBLISHED["fast_fie_lat"]["text"],
                    "direction_match": (m < 0) if g == "TD" else None,
                }
            )

        # Rest: ordinal LMM on periodic alpha (first valid block; all blocks secondary)
        rr = per["rest"].merge(meta, on="child")
        rr["ordinal"] = rr["group"].map(self.cfg.reproduction.ordinal_codes)
        tables["rest_per_child"] = rr
        for which, d in (
            ("first block", rr[rr["order"] == 0]),
            (
                "all blocks",
                rr.groupby(["child", "channel", "group", "age", "sex", "ordinal"], as_index=False)[
                    "periodic_alpha_pw"
                ].mean(),
            ),
        ):
            d = d.dropna(subset=["age"])
            res = _mixed("periodic_alpha_pw ~ ordinal + age + C(sex) + C(channel)", d, "ordinal")
            rows.append(
                {
                    "check": "rest_periodic_alpha",
                    "contrast": f"ordinal trend ASD<SIB<TD ({which})",
                    **res,
                    **_ns(d),
                    "published": PUBLISHED["rest_periodic_alpha"]["text"],
                    "direction_match": res["estimate"] > 0,
                }
            )
        summary = pd.DataFrame(rows)
        summary["direction_match"] = summary["direction_match"].astype(object)
        summary.loc[~np.isfinite(summary["estimate"].astype(float)), "direction_match"] = None  # failed fit
        tables["summary"] = summary
        return Result(
            "reproduction",
            tables,
            {
                "n_checks": int(len(summary)),
                "direction_matches": summary["direction_match"].dropna().astype(bool).tolist(),
            },
            label="confirmatory",
        )


def _ns(d: pd.DataFrame) -> dict:
    g = d.drop_duplicates("child")["group"].value_counts()
    return {f"n_{k}": int(g.get(k, 0)) for k in ("ASD", "TD", "SIB")}


if __name__ == "__main__":
    from config.loader import load_config
    from utils.store import ResultsStore

    cfg = load_config()
    ResultsStore(cfg).write(ReproductionChecks(cfg).run(), "reproduction")
