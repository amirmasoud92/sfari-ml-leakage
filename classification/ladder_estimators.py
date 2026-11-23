"""Every rung of the model ladder, including the MOFA+ factor rungs, scored three ways (Fig. 4; Additional file 1,
Table S13).

The full ladder is refitted (MOFA+ in every outer training fold; same folds, seeds and grids) and every repeat's
predictions are kept. For every rung and target the script reports

    pooled (the ladder's estimator),
    mean over repeats of the per-repeat AUC / R^2,
    mean over the 100 outer folds of the within-fold AUC (classification only),

and, for the contrast AUC(c) - AUC(b), the same three versions with a fold-level bootstrap for the within-fold one.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

from classification.model_ladder import ModelLadder
from stats.rng import generator
from utils import common as C
from utils.inputs import Inputs, make_targets


def main() -> None:
    t0 = time.time()
    out = C.out_dir("ladder_estimators")
    cfg = C.CFG.model_copy(update={"runtime": C.CFG.runtime.model_copy(update={"n_jobs": 24})})
    inp = Inputs(cfg)
    core, meta = inp.core(), inp.meta()
    lad = ModelLadder(cfg, inp.child_matrix(core), meta, use_mofa=True)
    stored = pd.read_parquet(C.RESULTS / "estimation" / "model_ladder__ladder.parquet")  # classification.model_ladder
    rng, seed = generator(C.SEED, "f01")
    rows, h3 = [], []
    for t in make_targets(meta, core, cfg):
        y = t.y.dropna()
        yv = y.to_numpy()
        kids = list(y.index)
        pos = {k: i for i, k in enumerate(kids)}
        reps = cfg.prediction.repeats
        folds = lad._folds(t, y, reps)
        preds = lad.cv_predictions(t, y, reps, lad.rungs, parallel=True)
        clf = t.kind == "classification"
        fn = roc_auc_score if clf else r2_score
        key = "auc" if clf else "r2"
        for rung, P in preds.items():
            pooled = fn(yv, np.nanmean(P, axis=0))
            per_rep = [fn(yv, P[r]) for r in range(reps)]
            fw = []
            if clf:
                for f in folds:
                    idx = [pos[k] for k in f.test]
                    if len(np.unique(yv[idx])) == 2:
                        fw.append(roc_auc_score(yv[idx], P[f.repeat, idx]))
            st = stored[(stored.target == t.name) & (stored.rung == rung)]
            rows.append(
                {
                    "target": t.name,
                    "rung": rung,
                    "metric": key,
                    "pooled": pooled,
                    "stored_pooled": float(st[key].iloc[0]) if len(st) else np.nan,
                    "mean_of_repeats": np.mean(per_rep),
                    "repeat_sd": np.std(per_rep),
                    "within_fold_mean": np.mean(fw) if fw else np.nan,
                    "within_fold_se": np.std(fw, ddof=1) / np.sqrt(len(fw)) if fw else np.nan,
                    "pred_sd_median": float(np.median(np.nanstd(P, axis=1))),
                }
            )
        if "c_specific_task" in preds:
            b, c = preds["b_shared"], preds["c_specific_task"]
            d_rep = [fn(yv, c[r]) - fn(yv, b[r]) for r in range(reps)]
            row = {
                "target": t.name,
                "metric": key,
                "pooled": fn(yv, np.nanmean(c, 0)) - fn(yv, np.nanmean(b, 0)),
                "mean_of_repeats": np.mean(d_rep),
            }
            if clf:
                dfw = []
                for f in folds:
                    idx = [pos[k] for k in f.test]
                    if len(np.unique(yv[idx])) == 2:
                        dfw.append(
                            roc_auc_score(yv[idx], c[f.repeat, idx])
                            - roc_auc_score(yv[idx], b[f.repeat, idx])
                        )
                dfw = np.array(dfw)
                bs = [rng.choice(dfw, len(dfw)).mean() for _ in range(2000)]
                row.update(
                    {
                        "within_fold_mean": dfw.mean(),
                        "within_fold_ci_lo": np.percentile(bs, 2.5),
                        "within_fold_ci_hi": np.percentile(bs, 97.5),
                    }
                )
            h3.append(row)
        print(t.name, f"{time.time() - t0:.0f} s", flush=True)
        print(pd.DataFrame(rows).query("target == @t.name").round(3).to_string(index=False), flush=True)
    R = pd.DataFrame(rows)
    R.to_parquet(out / "rungs.parquet")
    pd.DataFrame(h3).to_parquet(out / "h3_contrast.parquet")
    print(pd.DataFrame(h3).round(3).to_string(index=False))
    bad = R[np.abs(R.pooled - R.stored_pooled) > 1e-9]
    C.write_json(
        out / "summary.json",
        {"seed": seed, "seconds": time.time() - t0, "n_rungs_not_reproduced": int(len(bad))},
    )
    print("rungs not reproducing the stored pooled value:", len(bad))


if __name__ == "__main__":
    main()
