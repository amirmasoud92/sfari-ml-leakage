"""Covariate baselines and the contribution of EEG with children kept whole (Results; Fig. 4).

The non-factor rungs of the model ladder are refitted with the ladder's code, folds and settings (20 x 5 nested
cross-validation, inner-CV penalty grid, child bootstrap) in three analysis sets:

- core: the 70 autistic and TD core children; rungs (a) and (f) with six covariate sets (all covariates, without
  medication, demographic variables, sex only, medication only, data quality only);
- medication-free core children;
- male core children.

For each set: AUC of (a), (d), (e) and (f), and the paired bootstrap difference AUC(f) - AUC(a). The out-of-fold
predictions feed the likelihood-ratio test of :mod:`classification.eeg_increment_test`.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from classification.model_ladder import ModelLadder, Target, bootstrap_metrics
from stats.inference import percentile_ci
from stats.rng import generator
from utils import common as C
from utils.inputs import Inputs

QC = ["retained_data", "qc_bad_channels", "qc_asr_fraction"]
COVSETS = {
    "all_covariates": ["age", "sex", "fsiq", "medication_class", *QC],
    "no_medication": ["age", "sex", "fsiq", *QC],
    "age_sex_fsiq": ["age", "sex", "fsiq"],
    "sex_only": ["sex"],
    "medication_only": ["medication_class"],
    "data_quality_only": QC,
}


def with_covariates(cfg, cov: list[str]):
    return cfg.model_copy(update={"prediction": cfg.prediction.model_copy(update={"covariates": cov})})


def run_set(
    name: str, kids: list[str], meta: pd.DataFrame, X: pd.DataFrame, covsets: dict, n_jobs: int
) -> tuple:
    cfg = C.CFG.model_copy(update={"runtime": C.CFG.runtime.model_copy(update={"n_jobs": n_jobs})})
    m = meta.loc[kids]
    y = (m.group == "ASD").astype(int)
    t = Target("asd_vs_td", y, "classification", m.group)
    lad = ModelLadder(cfg, X.loc[kids], meta, use_mofa=False)
    reps = cfg.prediction.repeats
    t0 = time.time()
    preds = lad.cv_predictions(t, y, reps, ["d_best_paradigm", "e_all"], parallel=True)
    oof = {r: np.nanmean(P, axis=0) for r, P in preds.items()}
    for cs, cols in covsets.items():
        lad.cfg = with_covariates(cfg, cols)
        P = lad.cv_predictions(t, y, reps, ["a_covariates", "f_all_plus_covariates"], parallel=True)
        oof[f"a_covariates[{cs}]"] = np.nanmean(P["a_covariates"], axis=0)
        oof[f"f_all_plus_covariates[{cs}]"] = np.nanmean(P["f_all_plus_covariates"], axis=0)
    yv = y.to_numpy()
    rng, seed = generator(C.SEED, "committee_a2", name)
    rows = []
    for rung, p in oof.items():
        b = bootstrap_metrics(
            yv, p, "classification", cfg.prediction.n_bootstrap, rng, 0.95, float(yv.mean())
        )
        rows.append({"set": name, "rung": rung, "n": len(yv), "n_asd": int(yv.sum()), **b, "boot_seed": seed})
    diffs = []
    for cs in covsets:
        fa, ff = oof[f"a_covariates[{cs}]"], oof[f"f_all_plus_covariates[{cs}]"]
        d0 = roc_auc_score(yv, ff) - roc_auc_score(yv, fa)
        ds = []
        for _ in range(cfg.prediction.n_bootstrap):
            i = rng.integers(0, len(yv), len(yv))
            if len(np.unique(yv[i])) < 2:
                continue
            ds.append(roc_auc_score(yv[i], ff[i]) - roc_auc_score(yv[i], fa[i]))
        lo, hi = percentile_ci(np.array(ds), 0.95)
        diffs.append({"set": name, "covariates": cs, "auc_f_minus_a": d0, "ci_lo": lo, "ci_hi": hi})
    print(f"{name}: n = {len(yv)} ({int(yv.sum())} ASD), {time.time() - t0:.0f} s", flush=True)
    oof_df = pd.DataFrame({"child": kids, "y": yv, **oof}).assign(set=name)
    return pd.DataFrame(rows), pd.DataFrame(diffs), oof_df


def main() -> None:
    out = C.out_dir("covariate_baselines")
    inp = Inputs(C.CFG)
    core, meta = inp.core(), inp.meta()
    X = inp.child_matrix(core)
    at = [k for k in core if meta.loc[k, "group"] in ("ASD", "TD")]
    med = meta.loc[at, "medication_class"].fillna("none")
    sets = {
        "core": (at, COVSETS),
        "medication_free": (
            [k for k in at if med[k] == "none"],
            {
                k: [c for c in v if c != "medication_class"]
                for k, v in COVSETS.items()
                if k != "medication_only"
            },
        ),
        "male": (
            [k for k in at if meta.loc[k, "sex_male"] == 1],
            {k: [c for c in v if c != "sex"] for k, v in COVSETS.items() if k not in ("sex_only",)},
        ),
    }
    print(
        {k: (len(v[0]), int((meta.loc[v[0], "group"] == "ASD").sum())) for k, v in sets.items()}, flush=True
    )
    print(med.value_counts().to_dict(), flush=True)
    R, D, O = [], [], []
    for name, (kids, covs) in sets.items():
        r, d, o = run_set(name, kids, meta, X, covs, 30)
        R.append(r)
        D.append(d)
        O.append(o)
    R, D = pd.concat(R, ignore_index=True), pd.concat(D, ignore_index=True)
    R.to_parquet(out / "auc.parquet")
    D.to_parquet(out / "eeg_added_value.parquet")
    pd.concat(O, ignore_index=True).to_parquet(out / "oof.parquet")
    print(R[["set", "rung", "n", "auc", "auc_lo", "auc_hi"]].round(3).to_string(index=False))
    print(D.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
