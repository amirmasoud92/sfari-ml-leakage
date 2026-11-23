"""Shared helpers of the child-level analysis scripts.

Paths to the feature bank and the per-recording spectra, the output folder of each analysis, and the
cross-paradigm identification statistic used throughout: child x feature matrices (mean over blocks) are z-scored
per feature across children within each paradigm, each child's vector in one paradigm is correlated (Pearson) with
the vectors of all children in the other, and a child is identified if its own vector has the highest correlation.
Accuracy is averaged over the 30 ordered pairs of the six seated paradigms; each pair uses every child with both
paradigms. :func:`identify` optionally restricts the competitors to same-sex children within one year of age
(Fig. 2a). :func:`compare` gives the paired per-child difference in identification rate between two
representations with a child-bootstrap interval and a sign-flip p-value.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import PROJECT_ROOT, load_config
from identification import core
from stats.inference import paired_bootstrap_ci, sign_flip_p
from stats.rng import generator

ROOT = PROJECT_ROOT
CFG = load_config()  # config/analysis.yaml
PARADIGMS = list(CFG.dataset.seated_paradigms)
DERIV = CFG.path("derivatives")
RESULTS = CFG.path("results")
SPECTRA = DERIV / "features" / "primary" / "spectra"  # per-recording spectra (features.spectra)
BANK = DERIV / "features" / "primary" / "bank"  # feature bank (features.bank)
OUT = RESULTS / "analyses"  # one sub-folder per analysis script
SEED = CFG.project.seed


def out_dir(name: str) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))


def meta() -> pd.DataFrame:
    from utils.inputs import Inputs

    return Inputs(CFG).meta()


def recordings() -> list[tuple[str, str, Path]]:
    """(child, paradigm, npz path) of every kept seated recording."""
    out = []
    for f in sorted(SPECTRA.glob("sub-*_task-*.npz")):
        child, task = f.stem.split("_task-")
        if task in PARADIGMS:
            out.append((child, task, f))
    return out


def unshrink(C: np.ndarray, g: float) -> np.ndarray:
    """Undo ``(1 - g) S + g mu I`` with ``mu = tr(S) / n`` (the trace is preserved, so mu = tr(C) / n)."""
    mu = np.trace(C) / C.shape[0]
    return (C - g * mu * np.eye(C.shape[0])) / (1 - g)


def block_mean(rows: list[dict], feats: np.ndarray, names: list[str]) -> dict[str, pd.DataFrame]:
    """Block-level feature rows -> {paradigm: child x feature} (mean over blocks)."""
    keys = pd.DataFrame(rows)
    df = pd.concat([keys, pd.DataFrame(feats, columns=names)], axis=1)
    return {p: d.drop(columns=["paradigm", "block"]).groupby("child").mean() for p, d in df.groupby("paradigm")}


def bank_family(family: str) -> dict[str, pd.DataFrame]:
    """A feature-bank family as {paradigm: child x feature} (mean over blocks)."""
    w = pd.read_parquet(BANK / f"{family}.parquet")
    feats = [c for c in w.columns if c not in ("child", "paradigm", "block")]
    return {p: d.groupby("child")[feats].mean() for p, d in w.groupby("paradigm")}


def identify(
    rep: dict[str, pd.DataFrame],
    columns: list[str] | None = None,
    candidates: pd.DataFrame | None = None,
    children: list[str] | None = None,
) -> dict:
    """Mean accuracy over the 30 ordered pairs, mean differential identifiability, per-child correct rates.

    ``columns`` restricts the features. ``candidates`` (index child; columns age, sex) restricts every source
    child's competitors to target children of the same sex within 1 year of age (matched-candidate accuracy).
    """
    accs, chances, idiffs, per_child = [], [], [], []
    for src, tgt in itertools.permutations(PARADIGMS, 2):
        kids = sorted(set(rep[src].index) & set(rep[tgt].index))
        if children is not None:
            kids = [k for k in kids if k in set(children)]
        A = rep[src].loc[kids]
        B = rep[tgt].loc[kids]
        if columns is not None:
            A, B = A[columns], B[columns]
        A, B = core.zscore_columns(A.to_numpy(float)), core.zscore_columns(B.to_numpy(float))
        R = core.corr_matrix(A, B)
        if candidates is None:
            ok = core.correct_rows(R)
            chance = 1 / len(kids)
        else:
            c = candidates.reindex(kids)
            same = (c.sex.to_numpy()[:, None] == c.sex.to_numpy()[None, :]) & (
                np.abs(c.age.to_numpy()[:, None] - c.age.to_numpy()[None, :]) <= 1.0
            )
            Rm = np.where(same, R, -np.inf)
            np.fill_diagonal(Rm, np.diag(R))
            ok = core.correct_rows(Rm)
            chance = float(np.mean(1 / same.sum(axis=1)))
        accs.append(ok.mean())
        chances.append(chance)
        idiffs.append(core.differential_identifiability(R))
        per_child.append(pd.Series(ok, index=kids, name=f"{src}>{tgt}"))
    pc = pd.concat(per_child, axis=1)
    return {
        "accuracy": float(np.mean(accs)),
        "chance": float(np.mean(chances)),
        "diff_ident": float(np.mean(idiffs)),
        "child_rate": pc.mean(axis=1),
        "correct": pc,  # child x ordered pair ("src>tgt"), NaN where the child lacks the pair
        "pair_accuracy": accs,
    }


def compare(a: dict, b: dict, name_a: str, name_b: str, n_boot: int = 2000) -> dict:
    """Per-child (a - b) correct-rate difference: mean, paired bootstrap CI, sign-flip p."""
    common = a["child_rate"].index.intersection(b["child_rate"].index)
    d = (a["child_rate"][common] - b["child_rate"][common]).to_numpy()
    rng, seed = generator(SEED, "committee_compare", name_a, name_b)
    lo, hi = paired_bootstrap_ci(d, n_boot, rng, 0.95)
    return {
        "a": name_a,
        "b": name_b,
        "n": int(common.size),
        "rate_a": float(a["child_rate"][common].mean()),
        "rate_b": float(b["child_rate"][common].mean()),
        "difference": float(d.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "p_signflip": sign_flip_p(d, 10000, rng),
        "seed": seed,
    }
