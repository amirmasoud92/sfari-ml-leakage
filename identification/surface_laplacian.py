"""Identification after a surface Laplacian (Fig. 2a, b, d; Additional file 1, Table S3 and Supplementary
Methods S2).

The spectral pass of :mod:`features.spectra` is repeated with every clean 2-s segment multiplied by the
spherical-spline current source density operator L (MNE ``compute_current_source_density`` defaults on the
BioSemi-64 template positions: lambda^2 1e-5, stiffness 4, 50 Legendre terms; obtained by applying it to the identity,
so that CSD(x) = Lx) before the spectral estimates. Identification from CSD log power, CSD spectral shape and CSD
debiased wPLI is compared with the average-reference families, overall and for same-visit and other-visit pairs.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from features.signals import CleanRecording, bin_log_power, welch_from_segments
from features.spectra import cross_spectral
from identification import signature_decomposition as SD
from utils import common as C

CACHE = C.DERIV / "features" / "csd_spectra"


def one(child: str, paradigm: str, L: np.ndarray) -> str | None:
    out = CACHE / f"{child}_task-{paradigm}.npz"
    if out.is_file():
        return str(out)
    cfg = C.CFG
    fc = cfg.features
    r = CleanRecording(cfg, child, paradigm, "primary", "primary")
    if not r.paths["qc"].is_file() or not r.usable:
        return None
    bands = {k: tuple(v) for k, v in fc.bands.items()}
    psd, wpli, blocks = [], {k: [] for k in bands}, []
    for _, b in r.blocks.sort_values("block").iterrows():
        S1 = r.segments(
            b,
            None,
            seg_s=fc.spectral.welch_window_s,
            step_s=fc.spectral.welch_window_s * (1 - fc.spectral.welch_overlap),
        )
        S2 = r.segments(b, None, seg_s=fc.wpli.segment_s, step_s=fc.wpli.segment_s)
        if len(S2) < 5:
            continue
        S1, S2 = np.einsum("ij,njt->nit", L, S1), np.einsum("ij,njt->nit", L, S2)
        f, P = welch_from_segments(S1, r.sfreq)
        sel = (f >= 0.5) & (f <= 45.0)
        psd.append(P[:, sel].astype(np.float32))
        cs = cross_spectral(S2, r.sfreq, bands, fc.covariance.shrinkage_floor)
        for k in bands:
            wpli[k].append(np.asarray(cs[f"wpli_{k}"], np.float32))
        blocks.append(int(b["block"]))
    if not psd:
        return None
    np.savez_compressed(
        out,
        freqs=f[sel],
        psd=np.stack(psd),
        block=np.array(blocks),
        eeg_names=np.array(r.eeg_names),
        **{f"wpli_{k}": np.stack(v) for k, v in wpli.items()},
    )
    return str(out)


def reps_from_cache(recs) -> dict:
    sp = C.CFG.features.spectral
    rows, lp, wp = [], [], {b: [] for b in ("theta", "alpha", "beta")}
    names = None
    for child, par, _ in recs:
        f = CACHE / f"{child}_task-{par}.npz"
        if not f.is_file():
            continue
        z = np.load(f)
        names = [str(x) for x in z["eeg_names"]]
        for i, blk in enumerate(z["block"]):
            centres, v = bin_log_power(
                z["freqs"], z["psd"][i].astype(float), sp.fmin, sp.fmax - 0.5, sp.logpow_bin_hz
            )
            rows.append({"child": child, "paradigm": par, "block": int(blk)})
            lp.append(v)
            for b in wp:
                wp[b].append(z[f"wpli_{b}"][i])
    LP = np.stack(lp)
    cols = [f"{ch}_{int(c)}Hz" for ch in names for c in centres]
    reps = {"csd_logpow": C.block_mean(rows, LP.reshape(len(LP), -1), cols)}
    sh = LP - LP.mean(axis=2, keepdims=True)
    reps["csd_logpow_shape"] = C.block_mean(rows, sh.reshape(len(sh), -1), cols)
    for b, v in wp.items():
        W = np.stack(v)
        reps[f"csd_wpli_{b}"] = C.block_mean(rows, W, [f"p{i}" for i in range(W.shape[1])])
    return reps, names, [int(c) for c in centres]


def main() -> None:
    t0 = time.time()
    out = C.out_dir("surface_laplacian")
    CACHE.mkdir(parents=True, exist_ok=True)
    recs = C.recordings()
    names = [str(x) for x in np.load(recs[0][2])["eeg_names"]]
    L = SD.csd_operator(names)
    done = Parallel(n_jobs=10, backend="loky")(delayed(one)(c, p, L) for c, p, _ in recs)
    print(
        f"CSD spectra: {sum(d is not None for d in done)} of {len(recs)} recordings, {time.time() - t0:.0f} s",
        flush=True,
    )
    reps, names2, centres = reps_from_cache(recs)
    assert names2 == names
    for fam in ("spectral_logpow", "wpli_theta", "wpli_alpha", "wpli_beta"):
        reps[f"avgref_{fam}"] = C.bank_family(fam)
    rows, res = [], {}
    for k, rep in reps.items():
        r = C.identify(rep)
        res[k] = r
        rows.append(
            {
                "representation": k,
                "accuracy": r["accuracy"],
                "diff_ident": r["diff_ident"],
                "chance": r["chance"],
            }
        )
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_parquet(out / "identification.parquet")
    cmp = [
        C.compare(res[a], res[b], a, b)
        for a, b in (
            ("csd_logpow", "avgref_spectral_logpow"),
            ("csd_logpow_shape", "csd_logpow"),
            ("csd_wpli_alpha", "avgref_wpli_alpha"),
            ("csd_wpli_theta", "avgref_wpli_theta"),
            ("csd_wpli_beta", "avgref_wpli_beta"),
        )
    ]
    pd.DataFrame(cmp).to_parquet(out / "comparisons.parquet")
    # frequency- and channel-resolved CSD log power
    fr, ch = [], []
    for c in centres:
        r = C.identify(reps["csd_logpow"], columns=[f"{n}_{c}Hz" for n in names])
        fr.append({"freq_hz": c, "accuracy": r["accuracy"]})
    for n in names:
        r = C.identify(reps["csd_logpow"], columns=[f"{n}_{c}Hz" for c in centres])
        ch.append({"channel": n, "accuracy": r["accuracy"]})
    pd.DataFrame(fr).to_parquet(out / "by_frequency.parquet")
    pd.DataFrame(ch).to_parquet(out / "by_channel.parquet")
    # same-visit vs other-visit pairs (relative timing from identification.visit_structure)
    t = pd.read_parquet(C.OUT / "visit_structure" / "recording_times_relative.parquet").set_index(
        ["child", "paradigm"]
    )
    pc = res["csd_logpow"]["correct"]
    day_rows = []
    for k in (
        "csd_logpow",
        "csd_logpow_shape",
        "avgref_spectral_logpow",
        "csd_wpli_alpha",
        "avgref_wpli_alpha",
    ):
        cm = res[k]["correct"].reindex(index=pc.index, columns=pc.columns)
        same, cross = [], []
        for col in cm.columns:
            s_, t_ = col.split(">")
            for kid in cm.index[cm[col].notna()]:
                v = float(cm.loc[kid, col])
                (same if t.loc[(kid, s_), "day"] == t.loc[(kid, t_), "day"] else cross).append((kid, v))
        sd = pd.DataFrame(same, columns=["child", "v"]).groupby("child").v.mean()
        cd = pd.DataFrame(cross, columns=["child", "v"]).groupby("child").v.mean()
        both = sd.index.intersection(cd.index)
        day_rows.append(
            {
                "representation": k,
                "n_both": len(both),
                "same_day": sd[both].mean(),
                "cross_day": cd[both].mean(),
            }
        )
    pd.DataFrame(day_rows).to_parquet(out / "by_day.parquet")
    print(pd.DataFrame(day_rows).round(3).to_string(index=False))
    C.write_json(
        out / "summary.json",
        {
            "seconds": time.time() - t0,
            "n_recordings": len(recs),
            "csd": "MNE compute_current_source_density defaults, biosemi64",
        },
    )


if __name__ == "__main__":
    main()
