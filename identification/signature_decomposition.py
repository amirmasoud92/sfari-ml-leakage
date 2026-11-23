"""What carries the cross-paradigm individual signature (Fig. 2a, b, d; Additional file 1, Table S3).

Representations, each identified with the statistic of :mod:`utils.common`:

1. Log power (2-34 Hz, 1-Hz bins, 64 channels) and *spectral shape*, in which each channel's mean log power over
   frequency is subtracted. In the quasi-static regime the lead field does not depend on frequency, so skull,
   scalp, electrode and amplifier gain scale all frequencies of a channel by the same factor, an additive
   per-channel constant in log power; spectral shape removes it. Removing each frequency's mean over channels
   removes a global gain (*spatial pattern*).
2. The broadband sensor-covariance comparator split into a power part (log channel variance) and a
   correlation-structure part (tangent space of the correlation matrices).
3. Band power and correlation structure after a spherical-spline surface Laplacian (CSD; MNE defaults on the
   BioSemi-64 montage: lambda^2 1e-5, stiffness 4, 50 Legendre terms), applied as the linear operator L S L^T.
4. Identification from each single 1-Hz bin of all 64 channels and from the 33-bin spectrum of each single
   channel (Fig. 2b, d).
5. Identification with every child's competitors restricted to same-sex children within one year of age.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from features.signals import bin_log_power
from utils import common as C

BANDS = ("theta", "alpha", "beta", "broadband")
SP = C.CFG.features.spectral


def load_one(child: str, paradigm: str, path) -> list[dict]:
    z = np.load(path, allow_pickle=True)
    rows = []
    for i, blk in enumerate(z["block"]):
        centres, lp = bin_log_power(
            z["freqs"], z["psd"][i].astype(float), SP.fmin, SP.fmax - 0.5, SP.logpow_bin_hz
        )
        r = {"child": child, "paradigm": paradigm, "block": int(blk), "logpow": lp, "centres": centres}
        for b in BANDS:
            r[f"S_{b}"] = C.unshrink(z[f"cov_{b}"][i].astype(float), float(z[f"shrinkage_{b}"][i]))
        rows.append(r)
    return rows


def csd_operator(names: list[str]) -> np.ndarray:
    """Linear CSD operator of MNE's spherical-spline Laplacian on the BioSemi-64 montage (applied to identity)."""
    import mne

    info = mne.create_info(names, 256.0, "eeg")
    raw = mne.io.RawArray(np.eye(len(names)), info, verbose="error")
    raw.set_montage("biosemi64", verbose="error")
    out = mne.preprocessing.compute_current_source_density(raw, verbose="error")
    return out.get_data()  # column j = CSD of a unit impulse on channel j: CSD(X) = L @ X


def corr_from_cov(S: np.ndarray, shrink: float = 0.05) -> np.ndarray:
    """Correlation matrix of S shrunk towards the identity (SPD even for rank-deficient average-reference S)."""
    d = np.sqrt(np.clip(np.diag(S), 1e-30, None))
    R = S / np.outer(d, d)
    return (1 - shrink) * R + shrink * np.eye(len(R))


def tangent(mats: np.ndarray) -> np.ndarray:
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.tangentspace import tangent_space

    return tangent_space(mats, mean_riemann(mats))


def main() -> None:
    t0 = time.time()
    out = C.out_dir("signature_decomposition")
    recs = C.recordings()
    blocks = [r for rows in Parallel(n_jobs=30)(delayed(load_one)(*x) for x in recs) for r in rows]
    names = [str(x) for x in np.load(recs[0][2])["eeg_names"]]
    centres = [int(c) for c in blocks[0]["centres"]]
    keys = [{k: b[k] for k in ("child", "paradigm", "block")} for b in blocks]
    print(f"{len(recs)} recordings, {len(blocks)} blocks, {time.time() - t0:.0f} s", flush=True)

    L = csd_operator(names)
    LP = np.stack([b["logpow"] for b in blocks])  # (n, 64, 33) log10 power
    lp_cols = [f"{ch}_{c}Hz" for ch in names for c in centres]
    reps = {}
    reps["logpow"] = C.block_mean(keys, LP.reshape(len(LP), -1), lp_cols)
    shape = LP - LP.mean(axis=2, keepdims=True)  # per-channel gain removed
    reps["logpow_shape"] = C.block_mean(keys, shape.reshape(len(LP), -1), lp_cols)
    glob_ = LP - LP.mean(axis=1, keepdims=True)  # per-frequency global gain removed
    reps["logpow_spatial"] = C.block_mean(keys, glob_.reshape(len(LP), -1), lp_cols)
    both = shape - shape.mean(axis=1, keepdims=True)
    reps["logpow_shape_spatial"] = C.block_mean(keys, both.reshape(len(LP), -1), lp_cols)

    iu = np.triu_indices(len(names), 1)
    for space, op in (("avgref", None), ("csd", L)):
        bp = []
        for b in BANDS:
            S = np.stack([x[f"S_{b}"] for x in blocks])
            if op is not None:
                S = op @ S @ op.T
            bp.append(np.log10(np.clip(np.einsum("nii->ni", S), 1e-30, None)))
            if b == "broadband":
                Rm = np.stack([corr_from_cov(s) for s in S])
                T = tangent(Rm)
                reps[f"bb_corr_tangent_{space}"] = C.block_mean(keys, T, [f"t{i}" for i in range(T.shape[1])])
                fz = np.arctanh(np.clip(Rm[:, iu[0], iu[1]] / 0.95, -0.999, 0.999))
                reps[f"bb_corr_fisher_{space}"] = C.block_mean(
                    keys, fz, [f"r{i}" for i in range(fz.shape[1])]
                )
                reps[f"bb_logvar_{space}"] = C.block_mean(keys, bp[-1], names)
                print(f"{space} broadband done, {time.time() - t0:.0f} s", flush=True)
        BP = np.stack(bp, axis=2)  # (n, 64, 4)
        cols = [f"{ch}_{b}" for ch in names for b in BANDS]
        reps[f"bandpow4_{space}"] = C.block_mean(keys, BP.reshape(len(BP), -1), cols)
        sh = BP[:, :, :3] - BP[:, :, :3].mean(axis=2, keepdims=True)
        reps[f"bandpow3_shape_{space}"] = C.block_mean(
            keys, sh.reshape(len(sh), -1), [f"{ch}_{b}" for ch in names for b in BANDS[:3]]
        )
    print(f"representations built, {time.time() - t0:.0f} s", flush=True)

    # feature-bank families for reference (reproduce identification.analysis)
    for fam in (
        "spectral_logpow",
        "broadband_cov",
        "exg_only",
        "tangent_alpha",
        "aperiodic",
        "periodic",
        "alpha_cf",
    ):
        reps[f"bank_{fam}"] = C.bank_family(fam)

    m = C.meta()
    res, cmp_rows = {}, []
    rows = []
    for name, rep in reps.items():
        r = C.identify(rep)
        rm = C.identify(rep, candidates=m[["age", "sex"]])
        res[name] = r
        rows.append(
            {
                "representation": name,
                "n_features": rep[C.PARADIGMS[0]].shape[1],
                "accuracy": r["accuracy"],
                "chance": r["chance"],
                "diff_ident": r["diff_ident"],
                "accuracy_matched": rm["accuracy"],
                "chance_matched": rm["chance"],
            }
        )
        print(
            f"{name:28s} acc {r['accuracy']:.3f} Idiff {r['diff_ident']:.1f} matched {rm['accuracy']:.3f}",
            flush=True,
        )
    summary = pd.DataFrame(rows)
    summary.to_parquet(out / "representations.parquet")

    pairs = [
        ("logpow", "bank_broadband_cov"),
        ("logpow", "bb_logvar_avgref"),
        ("logpow", "bb_corr_tangent_avgref"),
        ("bank_broadband_cov", "bb_logvar_avgref"),
        ("bank_broadband_cov", "bb_corr_tangent_avgref"),
        ("logpow_shape", "logpow"),
        ("logpow_shape", "bank_broadband_cov"),
        ("logpow_shape", "bb_corr_tangent_avgref"),
        ("logpow_shape_spatial", "bb_corr_tangent_avgref"),
        ("bandpow4_csd", "bandpow4_avgref"),
        ("bandpow4_csd", "bb_corr_tangent_csd"),
        ("bandpow3_shape_csd", "bb_corr_tangent_csd"),
        ("logpow", "bank_exg_only"),
    ]
    for a, b in pairs:
        cmp_rows.append(C.compare(res[a], res[b], a, b))
    pd.DataFrame(cmp_rows).to_parquet(out / "comparisons.parquet")

    # frequency- and channel-resolved identification (log power and spectral shape)
    fr, ch = [], []
    for repname in ("logpow", "logpow_shape"):
        rep = reps[repname]
        for c in centres:
            cols = [f"{n}_{c}Hz" for n in names]
            r = C.identify(rep, columns=cols)
            fr.append(
                {
                    "representation": repname,
                    "freq_hz": c,
                    "accuracy": r["accuracy"],
                    "diff_ident": r["diff_ident"],
                }
            )
        for n in names:
            cols = [f"{n}_{c}Hz" for c in centres]
            r = C.identify(rep, columns=cols)
            ch.append(
                {
                    "representation": repname,
                    "channel": n,
                    "accuracy": r["accuracy"],
                    "diff_ident": r["diff_ident"],
                }
            )
    for space in ("avgref", "csd"):
        rep = reps[f"bandpow4_{space}"]
        for b in BANDS:
            r = C.identify(rep, columns=[f"{n}_{b}" for n in names])
            fr.append(
                {
                    "representation": f"bandpow_{space}",
                    "freq_hz": b,
                    "accuracy": r["accuracy"],
                    "diff_ident": r["diff_ident"],
                }
            )
    pd.DataFrame(fr).astype({"freq_hz": str}).to_parquet(out / "by_frequency.parquet")
    pd.DataFrame(ch).to_parquet(out / "by_channel.parquet")

    # per-child rates for later group analyses
    pd.DataFrame({k: v["child_rate"] for k, v in res.items()}).to_parquet(out / "child_rates.parquet")
    np.save(out / "csd_operator.npy", L)
    C.write_json(
        out / "summary.json",
        {
            "n_recordings": len(recs),
            "n_blocks": len(blocks),
            "seconds": time.time() - t0,
            "csd": "mne.preprocessing.compute_current_source_density defaults "
            "(lambda2 1e-5, stiffness 4, n_legendre 50), biosemi64",
            "matched_candidates": "same sex, |age difference| <= 1 year",
        },
    )
    print(f"done, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
