"""Pooled versus within-fold AUC (Additional file 1, Supplementary Methods S3).

The ladder averages each child's out-of-fold predictions over repeats and computes one AUC over all children
(pooled). If predictions are nearly constant, the residual differences between children come from each fold's
training composition: a child's own label is missing from its training set, so its prediction shifts away from its
label and the pooled AUC falls below 0.5 without information in the features. The within-fold AUC (computed inside
each test fold and averaged over the 100 folds) is not affected by between-fold offsets.

Rungs compared on the 70 autistic and TD core children: (a) all covariates, (a) sex only, (a) medication only,
(d), (e) and (f); same folds, seeds and settings as the ladder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from classification import covariate_baselines as CB
from classification.model_ladder import ModelLadder, Target
from utils import common as C
from utils.inputs import Inputs


def main() -> None:
    out = C.out_dir("auc_estimators")
    cfg = C.CFG.model_copy(update={"runtime": C.CFG.runtime.model_copy(update={"n_jobs": 30})})
    inp = Inputs(cfg)
    core, meta = inp.core(), inp.meta()
    kids = [k for k in core if meta.loc[k, "group"] in ("ASD", "TD")]
    m = meta.loc[kids]
    y = (m.group == "ASD").astype(int)
    t = Target("asd_vs_td", y, "classification", m.group)
    lad = ModelLadder(cfg, inp.child_matrix(core).loc[kids], meta, use_mofa=False)
    reps = cfg.prediction.repeats
    folds = lad._folds(t, y, reps)
    pos = {k: i for i, k in enumerate(kids)}
    preds = lad.cv_predictions(t, y, reps, ["d_best_paradigm", "e_all"], parallel=True)
    for cs in ("all_covariates", "sex_only", "medication_only"):
        lad.cfg = CB.with_covariates(cfg, CB.COVSETS[cs])
        P = lad.cv_predictions(t, y, reps, ["a_covariates", "f_all_plus_covariates"], parallel=True)
        preds[f"a_covariates[{cs}]"] = P["a_covariates"]
        if cs == "all_covariates":
            preds["f_all_plus_covariates[all_covariates]"] = P["f_all_plus_covariates"]
    yv = y.to_numpy()
    rows = []
    for rung, P in preds.items():
        pooled = roc_auc_score(yv, np.nanmean(P, axis=0))
        per_repeat = [roc_auc_score(yv, P[r]) for r in range(reps)]
        fw = []
        for f in folds:
            idx = [pos[k] for k in f.test]
            if len(np.unique(yv[idx])) == 2:
                fw.append(roc_auc_score(yv[idx], P[f.repeat, idx]))
        rows.append(
            {
                "rung": rung,
                "auc_pooled_over_repeats": pooled,
                "auc_per_repeat_mean": np.mean(per_repeat),
                "auc_foldwise_mean": np.mean(fw),
                "auc_foldwise_se": np.std(fw, ddof=1) / np.sqrt(len(fw)),
                "n_folds": len(fw),
                "pred_sd_median": float(np.median(np.nanstd(P, axis=1))),
            }
        )
    R = pd.DataFrame(rows)
    R.to_parquet(out / "auc_pooled_vs_foldwise.parquet")
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
