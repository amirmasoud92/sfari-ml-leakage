"""Child-bootstrap 95% intervals of the identification accuracies reported in the Results (Fig. 2a, b, d).

The accuracies of :mod:`identification.signature_decomposition`, :mod:`identification.surface_laplacian`,
:mod:`identification.muscle_artefact` and :mod:`identification.cross_system` are recomputed and given a
percentile interval from 2,000 resamples of children: in every resample, each ordered paradigm pair's accuracy is
the mean correctness of the resampled children who have that pair, and the statistic is the mean over pairs.
Competitor sets are held fixed, so the interval reflects which children were sampled.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from identification import core
from identification import cross_system as XS
from identification import signature_decomposition as SD
from identification import surface_laplacian as SL
from stats.rng import generator
from utils import common as C


def boot(correct: pd.DataFrame, rng, n_boot: int = 2000) -> tuple[float, float, float]:
    X = correct.to_numpy(float)
    M = np.isfinite(X)
    X0 = np.where(M, X, 0.0)
    W = rng.multinomial(len(X), np.full(len(X), 1 / len(X)), size=n_boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.nanmean((W @ X0) / (W @ M), axis=1)
    est = float(np.nanmean(np.nansum(X0, 0) / M.sum(0)))
    lo, hi = np.percentile(acc, [2.5, 97.5])
    return est, float(lo), float(hi)


def correct_pairs(rep: dict, pairs, columns=None) -> pd.DataFrame:
    per = []
    for s, t in pairs:
        kids = sorted(set(rep[s].index) & set(rep[t].index))
        A, B = rep[s].loc[kids], rep[t].loc[kids]
        if columns is not None:
            A, B = A[columns], B[columns]
        R = core.corr_matrix(core.zscore_columns(A.to_numpy(float)), core.zscore_columns(B.to_numpy(float)))
        per.append(pd.Series(core.correct_rows(R), index=kids, name=f"{s}>{t}"))
    return pd.concat(per, axis=1)


def main() -> None:
    out = C.out_dir("identification_intervals")
    rng, seed = generator(C.SEED, "v2_b6")
    recs = C.recordings()
    blocks = [r for rows in Parallel(n_jobs=30)(delayed(SD.load_one)(*x) for x in recs) for r in rows]
    names = [str(x) for x in np.load(recs[0][2])["eeg_names"]]
    centres = [int(c) for c in blocks[0]["centres"]]
    keys = [{k: b[k] for k in ("child", "paradigm", "block")} for b in blocks]
    LP = np.stack([b["logpow"] for b in blocks])
    cols = [f"{ch}_{c}Hz" for ch in names for c in centres]
    reps = {"logpow": C.block_mean(keys, LP.reshape(len(LP), -1), cols)}
    reps["logpow_shape"] = C.block_mean(
        keys, (LP - LP.mean(axis=2, keepdims=True)).reshape(len(LP), -1), cols
    )
    reps["exg_only"] = C.bank_family("exg_only")
    m = C.meta()
    rows = []

    def add(name, r, ref=None):
        est, lo, hi = boot(r["correct"], rng)
        rows.append(
            {"what": name, "accuracy": r["accuracy"], "boot_est": est, "lo": lo, "hi": hi, "reference": ref}
        )
        print(f"{name:40s} {r['accuracy']:.4f} ({lo:.4f}-{hi:.4f}) reference {ref}", flush=True)

    ref = pd.read_parquet(C.OUT / "signature_decomposition/representations.parquet").set_index("representation")
    add("logpow", C.identify(reps["logpow"]), ref.loc["logpow", "accuracy"])
    add("logpow_shape", C.identify(reps["logpow_shape"]), ref.loc["logpow_shape", "accuracy"])
    add("exg_only", C.identify(reps["exg_only"]), ref.loc["bank_exg_only", "accuracy"])
    add(
        "logpow_matched",
        C.identify(reps["logpow"], candidates=m[["age", "sex"]]),
        ref.loc["logpow", "accuracy_matched"],
    )
    fr = pd.read_parquet(C.OUT / "signature_decomposition/by_frequency.parquet")
    fr = fr[(fr.representation == "logpow") & fr.freq_hz.str.isdigit()]
    peak = fr.loc[fr.accuracy.idxmax()]
    add(
        f"logpow_{peak.freq_hz}Hz",
        C.identify(reps["logpow"], columns=[f"{n}_{peak.freq_hz}Hz" for n in names]),
        peak.accuracy,
    )
    ch = pd.read_parquet(C.OUT / "signature_decomposition/by_channel.parquet")
    ch = ch[ch.representation == "logpow"]
    best = ch.loc[ch.accuracy.idxmax()]
    add(
        f"logpow_channel_{best.channel}",
        C.identify(reps["logpow"], columns=[f"{best.channel}_{c}Hz" for c in centres]),
        best.accuracy,
    )
    emg = pd.read_parquet(C.OUT / "muscle_artefact/identification.parquet")
    add(
        "logpow_2_20Hz",
        C.identify(reps["logpow"], columns=[f"{n}_{c}Hz" for n in names for c in centres if 2 <= c <= 20]),
        emg.set_index(["analysis", "band"]).loc[("primary log power", "2-20 Hz"), "accuracy"],
    )
    # surface Laplacian
    creps, _, _ = SL.reps_from_cache(recs)
    cref = pd.read_parquet(C.OUT / "surface_laplacian/identification.parquet").set_index("representation")
    add("csd_logpow", C.identify(creps["csd_logpow"]), cref.loc["csd_logpow", "accuracy"])
    add("csd_logpow_shape", C.identify(creps["csd_logpow_shape"]), cref.loc["csd_logpow_shape", "accuracy"])
    # cross-system (seated BioSemi <-> standing Brain Products)
    mrecs = [
        (f.stem.split("_task-")[0], XS.MOTOR, f) for f in sorted(XS.MSPEC.glob("sub-*_task-motor.npz"))
    ]
    mblocks = [r for x in mrecs for r in SD.load_one(*x)]
    allb = blocks + mblocks
    k2 = [{k: b[k] for k in ("child", "paradigm", "block")} for b in allb]
    LP2 = np.stack([b["logpow"] for b in allb])
    c2 = [f"f{i}" for i in range(LP2[0].size)]
    xreps = {
        "logpow": C.block_mean(k2, LP2.reshape(len(LP2), -1), c2),
        "logpow_shape": C.block_mean(k2, (LP2 - LP2.mean(axis=2, keepdims=True)).reshape(len(LP2), -1), c2),
    }
    pairs = [(a, b) for a, b in itertools.permutations([*C.PARADIGMS, XS.MOTOR], 2) if XS.MOTOR in (a, b)]
    xref = pd.read_parquet(C.OUT / "cross_system/identification.parquet").set_index("representation")
    for k in ("logpow", "logpow_shape"):
        cp = correct_pairs(xreps[k], pairs)
        acc = float(np.mean(cp.mean(axis=0)))
        add(f"cross_system_{k}", {"correct": cp, "accuracy": acc}, xref.loc[k, "cross_system_accuracy"])
    T = pd.DataFrame(rows).assign(seed=seed)
    T["reference"] = T.reference.astype(str)
    T.to_parquet(out / "intervals.parquet")


if __name__ == "__main__":
    main()
