"""Identification across amplifier and electrode systems (Fig. 2a; Additional file 1, Table S3).

Each child's standing motor recordings (mobile Brain Products system) are matched with its seated recordings
(BioSemi) over the 12 ordered pairs motor <-> each seated paradigm, for log power, spectral shape, broadband
channel power, broadband correlation structure and the broadband sensor covariance (shrinkage 0.05), with the
tangent space at one Riemannian mean of all seated and motor blocks so that the two systems share a reference.
Paired per-child margins of log power and spectral shape over the covariance comparators use a child bootstrap and
a sign-flip test.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from identification import core
from identification import signature_decomposition as SD
from stats.inference import paired_bootstrap_ci, sign_flip_p
from stats.rng import generator
from utils import common as C

MOTOR = C.CFG.dataset.motor_paradigm
MSPEC = C.DERIV / "features" / "primary__standing" / "spectra"


def shrink(S, g=0.05):
    mu = np.trace(S) / S.shape[0]
    return (1 - g) * S + g * mu * np.eye(S.shape[0])


def identify_pairs(rep: dict, pairs: list[tuple[str, str]]) -> dict:
    accs, per = [], []
    for s, t in pairs:
        kids = sorted(set(rep[s].index) & set(rep[t].index))
        R = core.corr_matrix(
            core.zscore_columns(rep[s].loc[kids].to_numpy(float)),
            core.zscore_columns(rep[t].loc[kids].to_numpy(float)),
        )
        ok = core.correct_rows(R)
        accs.append(ok.mean())
        per.append(pd.Series(ok, index=kids, name=f"{s}>{t}"))
    pc = pd.concat(per, axis=1)
    return {
        "accuracy": float(np.mean(accs)),
        "pair_accuracy": accs,
        "child_rate": pc.mean(axis=1),
        "n": [len(p.dropna()) for p in per],
    }


def main() -> None:
    out = C.out_dir("cross_system")
    recs = C.recordings() + [
        (f.stem.split("_task-")[0], MOTOR, f) for f in sorted(MSPEC.glob("sub-*_task-motor.npz"))
    ]
    blocks = [r for x in recs for r in SD.load_one(*x)]
    keys = [{k: b[k] for k in ("child", "paradigm", "block")} for b in blocks]
    names = [str(x) for x in np.load(recs[0][2])["eeg_names"]]
    LP = np.stack([b["logpow"] for b in blocks])
    cols = [f"f{i}" for i in range(LP[0].size)]
    reps = {"logpow": C.block_mean(keys, LP.reshape(len(LP), -1), cols)}
    sh = LP - LP.mean(axis=2, keepdims=True)
    reps["logpow_shape"] = C.block_mean(keys, sh.reshape(len(sh), -1), cols)
    S = np.stack([b["S_broadband"] for b in blocks])
    reps["bb_logvar"] = C.block_mean(keys, np.log10(np.einsum("nii->ni", S)), names)
    T = SD.tangent(np.stack([SD.corr_from_cov(s) for s in S]))
    reps["bb_corr_tangent"] = C.block_mean(keys, T, [f"t{i}" for i in range(T.shape[1])])
    T2 = SD.tangent(np.stack([shrink(s) for s in S]))
    reps["broadband_cov"] = C.block_mean(keys, T2, [f"t{i}" for i in range(T2.shape[1])])
    pairs = [(a, b) for a, b in itertools.permutations([*C.PARADIGMS, MOTOR], 2) if MOTOR in (a, b)]
    seated_pairs = list(itertools.permutations(C.PARADIGMS, 2))
    res, rows = {}, []
    for k, rep in reps.items():
        r = identify_pairs(rep, pairs)
        rs = identify_pairs(rep, seated_pairs)
        res[k] = r
        rows.append(
            {
                "representation": k,
                "cross_system_accuracy": r["accuracy"],
                "cross_system_min_pair": float(np.min(r["pair_accuracy"])),
                "cross_system_max_pair": float(np.max(r["pair_accuracy"])),
                "n_range": f"{min(r['n'])}-{max(r['n'])}",
                "seated_accuracy": rs["accuracy"],
            }
        )
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_parquet(out / "identification.parquet")
    rng, seed = generator(C.SEED, "f07")
    cmp = []
    for a in ("logpow", "logpow_shape"):
        for b in ("broadband_cov", "bb_corr_tangent", "bb_logvar"):
            common = res[a]["child_rate"].index.intersection(res[b]["child_rate"].index)
            d = (res[a]["child_rate"][common] - res[b]["child_rate"][common]).to_numpy()
            lo, hi = paired_bootstrap_ci(d, 2000, rng, 0.95)
            cmp.append(
                {
                    "a": a,
                    "b": b,
                    "n": len(d),
                    "difference": d.mean(),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p_signflip": sign_flip_p(d, 10000, rng),
                }
            )
    pd.DataFrame(cmp).to_parquet(out / "margins.parquet")
    print(pd.DataFrame(cmp).round(3).to_string(index=False))
    C.write_json(
        out / "summary.json", {"seed": seed, "n_motor_recordings": len(list(MSPEC.glob("*.npz")))}
    )


if __name__ == "__main__":
    main()
