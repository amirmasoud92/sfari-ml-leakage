"""Variance partition of block-level log power into a between-child and a between-group share (Methods,
Individual identification; Fig. 2c, e; Additional file 1, Supplementary Methods S2).

Data: block-level log power (64 channels x 33 1-Hz bins, 2-34 Hz) of the 70 autistic and TD core children. For every
feature the paradigm mean is removed, and the total sum of squares over blocks is split exactly into

    child share     = SS between children / SS total
    diagnosis share = SS between the two groups / SS total

The diagnosis share is nested in the child share, so both are on one scale. Features: each 1-Hz bin, and theta
(4-7 Hz), alpha (8-12 Hz) and beta (13-30 Hz) band means of log power per electrode. Hedges' g (autistic - TD, child
means) is stored for the band maps. Null: 1,000 permutations of diagnosis between children; for each band the
maximum diagnosis share over electrodes gives a family-wise 95% threshold, and for each bin the mean over electrodes
gives a 95% band.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from stats.rng import generator
from utils import common as C
from utils.inputs import Inputs

N_PERM = 1000
BANDS = {"theta": (4, 7), "alpha": (8, 12), "beta": (13, 30)}


def data():
    inp = Inputs(C.CFG)
    meta, core = inp.meta(), inp.core()
    kids = [k for k in core if meta.loc[k, "group"] in ("ASD", "TD")]
    w = pd.read_parquet(C.BANK / "spectral_logpow.parquet")
    w = w[w.child.isin(kids)].reset_index(drop=True)
    feats = [c for c in w.columns if c not in ("child", "paradigm", "block")]
    X = w[feats].to_numpy(float)
    X = X - w.groupby("paradigm")[feats].transform("mean").to_numpy(float)
    y = (meta.loc[w.child, "group"] == "ASD").astype(int).to_numpy()
    return X, feats, y, w.child.to_numpy()


def band_matrix(X, feats):
    ch = [f.rsplit("_", 1)[0] for f in feats]
    hz = np.array([int(f.rsplit("_", 1)[1][:-2]) for f in feats])
    chans = list(dict.fromkeys(ch))
    out, names = [], []
    for b, (lo, hi) in BANDS.items():
        for c in chans:
            idx = [i for i, (cc, h) in enumerate(zip(ch, hz, strict=True)) if cc == c and lo <= h <= hi]
            out.append(X[:, idx].mean(1))
            names.append((b, c))
    return np.column_stack(out), names


def shares(X, codes, n_child, ych):
    """Child and diagnosis shares of the (paradigm-centred) total sum of squares, per column of X."""
    gm = X.mean(0)
    sst = ((X - gm) ** 2).sum(0)
    cnt = np.bincount(codes, minlength=n_child).astype(float)
    cm = np.zeros((n_child, X.shape[1]))
    np.add.at(cm, codes, X)
    cm /= cnt[:, None]
    ssc = (cnt[:, None] * (cm - gm) ** 2).sum(0)
    yb = ych[codes]
    ssg = sum(((yb == g).sum() * (X[yb == g].mean(0) - gm) ** 2) for g in (0, 1))
    return ssc / sst, ssg / sst, cm


def hedges_g(cm, ych):
    a, t = cm[ych == 1], cm[ych == 0]
    n1, n0 = len(a), len(t)
    sp = np.sqrt(((n1 - 1) * a.var(0, ddof=1) + (n0 - 1) * t.var(0, ddof=1)) / (n1 + n0 - 2))
    return (a.mean(0) - t.mean(0)) / sp * (1 - 3 / (4 * (n1 + n0) - 9))


def main() -> None:
    out = C.out_dir("spatial_variance_partition")
    t0 = time.time()
    X, feats, y, g = data()
    kids, codes = np.unique(g, return_inverse=True)
    ych = pd.Series(y, index=g).groupby(level=0).first().reindex(kids).to_numpy()
    B, bnames = band_matrix(X, feats)
    rng, seed = generator(C.SEED, "v3_b10")

    cs_f, ds_f, _ = shares(X, codes, len(kids), ych)
    cs_b, ds_b, cm_b = shares(B, codes, len(kids), ych)
    gb = hedges_g(cm_b, ych)

    hz = np.array([int(f.rsplit("_", 1)[1][:-2]) for f in feats])
    bins = np.unique(hz)
    band_of = np.array([b for b, _ in bnames])
    null_bin, null_max = [], []
    for _ in range(N_PERM):
        yp = rng.permutation(ych)
        _, d_f, _ = shares(X, codes, len(kids), yp)
        _, d_b, _ = shares(B, codes, len(kids), yp)
        null_bin.append([d_f[hz == h].mean() for h in bins])
        null_max.append([d_b[band_of == b].max() for b in BANDS])
    null_bin, null_max = np.array(null_bin), np.array(null_max)

    pd.DataFrame(
        {
            "band": band_of,
            "channel": [c for _, c in bnames],
            "child_share": cs_b,
            "diagnosis_share": ds_b,
            "hedges_g": gb,
        }
    ).to_parquet(out / "band_maps.parquet")
    fr = pd.DataFrame({"freq_hz": hz, "child_share": cs_f, "diagnosis_share": ds_f}).groupby("freq_hz").mean()
    fr["null_lo"], fr["null_med"], fr["null_hi"] = np.percentile(null_bin, [2.5, 50, 97.5], axis=0)
    fr.reset_index().to_parquet(out / "by_frequency.parquet")
    thr = {b: float(np.percentile(null_max[:, i], 95)) for i, b in enumerate(BANDS)}
    summ = {
        "seed": seed,
        "n_perm": N_PERM,
        "n_children": len(kids),
        "n_asd": int(ych.sum()),
        "n_blocks": len(g),
        "bands": BANDS,
        "fwe_threshold_diagnosis_share": thr,
        "n_channels_above_threshold": {b: int((ds_b[band_of == b] > thr[b]).sum()) for b in BANDS},
        "child_share_median": {b: float(np.median(cs_b[band_of == b])) for b in BANDS},
        "child_share_range": {
            b: [float(cs_b[band_of == b].min()), float(cs_b[band_of == b].max())] for b in BANDS
        },
        "diagnosis_share_median": {b: float(np.median(ds_b[band_of == b])) for b in BANDS},
        "diagnosis_share_max": {b: float(ds_b[band_of == b].max()) for b in BANDS},
        "abs_g_max": {b: float(np.abs(gb[band_of == b]).max()) for b in BANDS},
        "ratio_median": {b: float(np.median(ds_b[band_of == b] / cs_b[band_of == b])) for b in BANDS},
        "minutes": (time.time() - t0) / 60,
    }
    C.write_json(out / "summary.json", summ)
    print(summ)


if __name__ == "__main__":
    main()
