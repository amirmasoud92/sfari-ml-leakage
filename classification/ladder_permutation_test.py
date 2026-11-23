"""Label-permutation test of the model-ladder rungs without MOFA+ (Additional file 1, Table S14).

For rungs (a), (d), (e) and (f) and every target, 1,000 label permutations in 5 x 5 cross-validation: (i) the null
with the outer folds re-stratified on each permuted label vector (``ModelLadder.permutation_test_restratified``;
the reported p-values) and (ii) for comparison the null with folds kept stratified on the true labels (200
permutations, same permutation seeds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, roc_auc_score

from classification.model_ladder import ModelLadder
from config.loader import load_config
from stats.rng import generator
from utils.base import Result
from utils.inputs import Inputs, make_targets
from utils.store import ResultsStore


def fixed_fold_null(lad: ModelLadder, t, y: pd.Series, n_perm: int, rungs: list[str]) -> pd.DataFrame:
    """The null with the outer folds stratified on the true labels (same permutation seeds) for the given rungs."""
    cfg = lad.cfg
    reps = cfg.prediction.permutation_repeats
    key = "auc" if t.kind == "classification" else "r2"
    fn = roc_auc_score if key == "auc" else r2_score
    yv = y.to_numpy()
    lad.cache.prefill([(f.train, f.test) for f in lad._folds(t, y, reps)])
    light = lad._light([(f.train, f.test) for f in lad._folds(t, y, reps)])
    rng, _ = generator(cfg.project.seed, "ladder_perm", t.name)  # the permutation seeds of the model ladder
    seeds = rng.integers(0, 2**31, n_perm)

    def one(s):
        yp = pd.Series(np.random.default_rng(int(s)).permutation(yv), index=y.index)
        P = light.cv_predictions(t, yp, reps, rungs, parallel=False)
        return {r: fn(yp.to_numpy(), np.nanmean(P[r], 0)) for r in rungs}

    null = Parallel(n_jobs=cfg.runtime.n_jobs)(delayed(one)(s) for s in seeds)
    return pd.DataFrame(
        [
            {
                "target": t.name,
                "rung": r,
                "null_mean_fixed_folds": float(np.mean([d[r] for d in null])),
                "null_q95_fixed_folds": float(np.quantile([d[r] for d in null], 0.95)),
            }
            for r in rungs
        ]
    )


def main(n_perm: int | None = None, n_perm_fixed: int = 200) -> None:
    cfg = load_config()
    n_perm = n_perm or cfg.prediction.n_permutations
    inp = Inputs(cfg)
    core, meta = inp.core(), inp.meta()
    lad = ModelLadder(cfg, inp.child_matrix(core), meta, use_mofa=False)
    out = []
    for t in make_targets(meta, core, cfg):
        y = t.y.dropna()
        rungs = list(lad.rungs)
        res = lad.permutation_test_restratified(t, y, n_perm, rungs)
        fix = fixed_fold_null(lad, t, y, n_perm_fixed, rungs)
        res = res.merge(fix, on=["target", "rung"])
        out.append(res)
        print(res.drop(columns=["perm_seed"]).round(3).to_string(index=False), flush=True)
    df = pd.concat(out, ignore_index=True)
    ResultsStore(cfg).write(
        Result("model_ladder_perm_restratified", {"perm": df}, {"n_perm": n_perm}, label="estimation"),
        "estimation",
    )


if __name__ == "__main__":
    main()
