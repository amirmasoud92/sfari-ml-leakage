"""The model ladder in 20 x 5 repeated nested cross-validation (Methods, Diagnostic classification with children
kept whole; Fig. 4; Additional file 1, Supplementary Methods S3 and Tables S13, S14 and S16).

Rungs: (a) covariates; (b) shared and (b') partially shared MOFA+ factors; (c) task-specific and (c') rest-specific
factors; (d) best single paradigm; (e) all features; (f) all features plus covariates. Per outer fold, the
unsupervised steps (``ViewReducer``, ``MultiViewFactorModel``) are fitted on the outer training children only and
cached (:class:`FoldCache`); they are label-free, so rungs, targets and permutations reuse them. The supervised step
(z-scoring + L2 logistic regression for diagnosis, ridge regression for SRS-2, MABC-2 and FSIQ) is tuned by inner
cross-validation on the training representation and evaluated on the outer test children. Out-of-fold predictions
are averaged over repeats (the pooled estimator). The label-permutation test with folds re-stratified on every
permuted label vector is :meth:`ModelLadder.permutation_test_restratified`.

Running the module computes the ladder for all targets (``<results>/estimation/model_ladder*``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification.factor_model import MultiViewFactorModel, ViewReducer
from classification.splits import SplitFactory
from config.loader import Config
from stats.inference import percentile_ci
from stats.rng import generator
from utils.base import Analysis, Result
from utils.logging_utils import get_logger

log = get_logger(__name__)
FACTOR_RUNGS = {
    "b_shared": "shared",
    "b2_partial": "partial",
    "c_specific_task": "specific_task",
    "c2_specific_rest": "specific_rest",
}


@dataclass
class FoldRep:
    """Representations of one outer fold (train / test), keyed by rung source."""

    train: dict[str, np.ndarray] = field(default_factory=dict)
    test: dict[str, np.ndarray] = field(default_factory=dict)
    views_train: dict[str, np.ndarray] = field(default_factory=dict)
    views_test: dict[str, np.ndarray] = field(default_factory=dict)
    factor_labels: list[str] = field(default_factory=list)


class FoldCache:
    """Unsupervised representations per training set (reused across rungs, targets and permutations)."""

    def __init__(
        self, cfg: Config, X: pd.DataFrame, use_mofa: bool = True, missing_views: str = "impute"
    ) -> None:
        self.cfg = cfg
        self.X = X
        self.use_mofa = use_mofa
        self.missing_views = missing_views
        self._cache: dict[tuple, FoldRep] = {}

    def get(self, train: tuple[str, ...], test: tuple[str, ...]) -> FoldRep:
        key = (train, test)
        if key not in self._cache:
            self._cache[key] = self._compute(train, test)
        return self._cache[key]

    def prefill(self, keys: list[tuple[tuple[str, ...], tuple[str, ...]]]) -> None:
        """Compute the missing representations in parallel (one fold per worker, single-threaded BLAS); each fold
        is computed exactly as :meth:`get` would, so the result does not depend on the number of workers."""
        todo = list(dict.fromkeys(k for k in keys if k not in self._cache))
        if len(todo) < 2:
            for k in todo:  # one missing fold: computed here, so that worker copies always find it
                self.get(*k)
            return

        def one(tr, te):
            from threadpoolctl import threadpool_limits

            with threadpool_limits(1):
                return self._compute(tr, te)

        reps = Parallel(n_jobs=self.cfg.runtime.n_jobs)(delayed(one)(tr, te) for tr, te in todo)
        self._cache.update(dict(zip(todo, reps, strict=True)))

    def _compute(self, train: tuple[str, ...], test: tuple[str, ...]) -> FoldRep:
        lc = self.cfg.latent
        Xtr, Xte = self.X.loc[list(train)], self.X.loc[list(test)]
        red = ViewReducer(lc.view_pca_components, missing_views=self.missing_views).fit(Xtr)
        Rtr, Rte = red.transform(Xtr), red.transform(Xte)
        rep = FoldRep()
        # direct rungs: a missing block is set to the training mean (0 after centring); MOFA+ sees it as missing
        Ftr, Fte = Rtr.fillna(0.0), Rte.fillna(0.0)
        rep.train["e_all"], rep.test["e_all"] = Ftr.to_numpy(), Fte.to_numpy()
        for v in Rtr.columns.get_level_values(0).unique():
            rep.views_train[v], rep.views_test[v] = Ftr.loc[:, v].to_numpy(), Fte.loc[:, v].to_numpy()
        if self.use_mofa:
            m = MultiViewFactorModel(
                lc.max_factors,
                lc.drop_factor_threshold,
                lc.active_r2,
                self.cfg.project.seed,
                lc.convergence_mode,
                method=lc.method,
            ).fit(Rtr)
            Ztr, Zte = m.transform(Rtr), m.transform(Rte)
            for rung, kind in FACTOR_RUNGS.items():
                idx = m.factor_index(kind)
                rep.train[rung], rep.test[rung] = Ztr[:, idx], Zte[:, idx]
            rep.factor_labels = list(m.labels_)
        return rep


def _estimator(kind: str, cfg: Config):
    p = cfg.prediction
    if kind == "classification":
        return LogisticRegression(max_iter=2000), {"model__C": list(p.logistic_c)}
    return Ridge(), {"model__alpha": list(p.ridge_alpha)}


def fit_predict(Xtr, ytr, Xte, kind: str, cfg: Config, inner: list, scoring: str):
    """Inner-CV-tuned z-score + linear model; returns (test predictions, train decision scores, model)."""
    if Xtr.shape[1] == 0:  # no feature for this rung in this fold: chance-level prediction (logged by caller)
        base = float(np.mean(ytr))
        return np.full(len(Xte), base), np.full(len(Xtr), base), None
    est, grid = _estimator(kind, cfg)
    pipe = Pipeline([("scale", StandardScaler()), ("model", est)])
    fold_id = np.full(len(ytr), -1)
    for k, (_, te) in enumerate(inner):
        fold_id[te] = k
    gs = GridSearchCV(pipe, grid, cv=PredefinedSplit(fold_id), scoring=scoring, refit=True, n_jobs=1)
    gs.fit(Xtr, ytr)
    if kind == "classification":
        return gs.predict_proba(Xte)[:, 1], gs.decision_function(Xtr), gs.best_estimator_
    return gs.predict(Xte), gs.predict(Xtr), gs.best_estimator_


def metrics(y: np.ndarray, pred: np.ndarray, kind: str, prevalence: float | None = None) -> dict:
    if kind == "classification":
        thr = 0.5 if prevalence is None else prevalence
        return {
            "auc": roc_auc_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred >= thr),
            "brier": brier_score_loss(y, pred),
        }
    return {"r2": r2_score(y, pred), "pearson_r": float(np.corrcoef(y, pred)[0, 1])}


def bootstrap_metrics(y, pred, kind, n, rng, level, prevalence=None) -> dict:
    est = metrics(y, pred, kind, prevalence)
    draws: dict[str, list] = {k: [] for k in est}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if kind == "classification" and len(np.unique(y[i])) < 2:
            continue
        for k, v in metrics(y[i], pred[i], kind, prevalence).items():
            draws[k].append(v)
    out = {}
    for k, v in est.items():
        lo, hi = percentile_ci(np.array(draws[k]), level)
        out.update({k: float(v), f"{k}_lo": lo, f"{k}_hi": hi})
    return out


def calibration_curve(y, p, bins: int) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return pd.DataFrame(
        {
            "bin": range(bins),
            "mean_pred": [p[b == k].mean() if (b == k).any() else np.nan for k in range(bins)],
            "observed": [y[b == k].mean() if (b == k).any() else np.nan for k in range(bins)],
            "n": [int((b == k).sum()) for k in range(bins)],
        }
    )


def haufe_pattern(X: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Activation pattern of a linear model with one output: cov(x_j, s) / var(s) (Haufe et al. 2014)."""
    Xc = X - X.mean(0)
    s = score - score.mean()
    return (Xc.T @ s) / (len(s) - 1) / s.var(ddof=1)


@dataclass
class Target:
    name: str
    y: pd.Series  # index child (training population)
    kind: str  # classification | regression
    strata: pd.Series  # for stratified folds
    heldout: pd.Series | None = None  # siblings (prediction only)


class ModelLadder(Analysis):
    """Model ladder in repeated nested cross-validation (module docstring)."""

    label = "estimation"
    needs_labels = True

    def __init__(
        self,
        cfg: Config,
        X: pd.DataFrame,
        covariates: pd.DataFrame,
        use_mofa: bool = True,
        missing_views: str = "impute",
    ) -> None:
        super().__init__(cfg)
        self.X = X
        self.cov = covariates
        self.cache = FoldCache(cfg, X, use_mofa, missing_views)
        self.rungs = [
            "a_covariates",
            *(FACTOR_RUNGS if use_mofa else {}),
            "d_best_paradigm",
            "e_all",
            "f_all_plus_covariates",
        ]

    def _rep(self, rung: str, rep: FoldRep, train, test, cov_cols: list[str]):
        if rung == "a_covariates":
            C = self.cov[cov_cols]
            mu = C.loc[list(train)].mean()
            return C.loc[list(train)].fillna(mu).to_numpy(), C.loc[list(test)].fillna(mu).to_numpy()
        if rung == "f_all_plus_covariates":
            a_tr, a_te = self._rep("a_covariates", rep, train, test, cov_cols)
            return np.hstack([rep.train["e_all"], a_tr]), np.hstack([rep.test["e_all"], a_te])
        return rep.train[rung], rep.test[rung]

    def _cov_cols(self, target: str) -> list[str]:
        """Numeric covariate columns for rung (a): ``sex`` -> ``sex_male``; ``medication_class`` -> its one-hot
        columns (reference: none); the target itself is never a covariate."""
        out = []
        for c in self.cfg.prediction.covariates:
            if c == target:
                continue
            if c == "sex" and "sex_male" in self.cov.columns:
                out.append("sex_male")
            elif c in self.cov.columns and pd.api.types.is_numeric_dtype(self.cov[c]):
                out.append(c)
            else:
                out += [k for k in self.cov.columns if k.startswith(f"{c}=")]
        return out

    def cv_predictions(
        self, t: Target, y: pd.Series, repeats: int, rungs: list[str], parallel: bool = False
    ) -> dict[str, np.ndarray]:
        """Out-of-fold predictions per rung and repeat. Folds run in parallel (``parallel``) or in turn; both give
        identical predictions."""
        cfg = self.cfg
        kids = list(y.index)
        sf = SplitFactory(kids, cfg.project.seed, strata=t.strata.reindex(kids).to_numpy())
        scoring = "roc_auc" if t.kind == "classification" else "neg_mean_squared_error"
        preds = {r: np.full((repeats, len(kids)), np.nan) for r in rungs}
        pos = {k: i for i, k in enumerate(kids)}
        folds = list(sf.outer(cfg.prediction.outer_folds, repeats))
        self.cache.prefill([(f.train, f.test) for f in folds])

        def one(lad, fold):
            rep = lad.cache.get(fold.train, fold.test)
            ytr = y.loc[list(fold.train)].to_numpy()
            # inner splits are a pure function of the training children (seeded), not of the call order. For a
            # classification they are stratified on the labels being fitted, so that under permuted labels no
            # inner fold is left with one class; the outer folds (and the fold cache) stay fixed.
            inner_strata = (
                ytr if t.kind == "classification" else t.strata.reindex(list(fold.train)).to_numpy()
            )
            inner = sf.inner(fold.train, cfg.prediction.inner_folds, strata=inner_strata)
            out = {}
            for rung in rungs:
                if rung == "d_best_paradigm":
                    out[rung] = lad._best_view(rep, ytr, t.kind, inner, scoring)
                else:
                    Xtr, Xte = lad._rep(rung, rep, fold.train, fold.test, lad._cov_cols(t.name))
                    out[rung] = fit_predict(Xtr, ytr, Xte, t.kind, cfg, inner, scoring)[0]
            return out

        if parallel and len(folds) > 1 and cfg.runtime.n_jobs != 1:
            fold_preds = Parallel(n_jobs=cfg.runtime.n_jobs)(
                delayed(one)(self._light([(f.train, f.test)]), f) for f in folds
            )
        else:
            fold_preds = [one(self, f) for f in folds]
        for fold, out in zip(folds, fold_preds, strict=True):
            te_idx = [pos[k] for k in fold.test]
            for rung, p in out.items():
                preds[rung][fold.repeat, te_idx] = p
        return preds

    def _folds(self, t: Target, y: pd.Series, repeats: int) -> list:
        kids = list(y.index)
        sf = SplitFactory(kids, self.cfg.project.seed, strata=t.strata.reindex(kids).to_numpy())
        return list(sf.outer(self.cfg.prediction.outer_folds, repeats))

    def _light(self, keys: list[tuple]) -> ModelLadder:
        """A copy for worker processes: only the cached folds in ``keys``, without the raw feature matrix."""
        light = copy.copy(self)
        light.X = None
        light.cache = copy.copy(self.cache)
        light.cache.X = None
        light.cache._cache = {k: self.cache._cache[k] for k in keys}
        return light

    def _best_view(self, rep: FoldRep, ytr, kind, inner, scoring):
        """Rung (d): the paradigm with the best inner-CV score. An undefined inner score (NaN) never wins; if
        every view's score is undefined, the first view is used."""
        best, best_score, model = None, -np.inf, None
        for v, Xv in rep.views_train.items():
            est, grid = _estimator(kind, self.cfg)
            fold_id = np.full(len(ytr), -1)
            for k, (_, te) in enumerate(inner):
                fold_id[te] = k
            gs = GridSearchCV(
                Pipeline([("scale", StandardScaler()), ("model", est)]),
                grid,
                cv=PredefinedSplit(fold_id),
                scoring=scoring,
            ).fit(Xv, ytr)
            score = gs.best_score_ if np.isfinite(gs.best_score_) else -np.inf
            if model is None or score > best_score:
                best, best_score, model = v, score, gs.best_estimator_
        Xte = rep.views_test[best]
        return model.predict_proba(Xte)[:, 1] if kind == "classification" else model.predict(Xte)

    def run_target(self, t: Target, n_perm: int | None = None) -> dict[str, pd.DataFrame]:
        cfg = self.cfg
        p = cfg.prediction
        y = t.y.dropna()
        preds = self.cv_predictions(t, y, p.repeats, self.rungs, parallel=True)
        rng, seed = generator(cfg.project.seed, "ladder_boot", t.name)
        prev = float(y.mean()) if t.kind == "classification" else None
        rows, oof = [], {}
        yv = y.to_numpy()
        for rung, P in preds.items():
            m = np.nanmean(P, axis=0)
            oof[rung] = m
            per_rep = [metrics(yv, P[r], t.kind, prev) for r in range(P.shape[0])]
            key = "auc" if t.kind == "classification" else "r2"
            rows.append(
                {
                    "target": t.name,
                    "rung": rung,
                    "n": len(yv),
                    **bootstrap_metrics(yv, m, t.kind, p.n_bootstrap, rng, cfg.stats.ci_level, prev),
                    f"{key}_repeat_sd": float(np.std([d[key] for d in per_rep])),
                    "boot_seed": seed,
                }
            )
        res = pd.DataFrame(rows)
        # H3 contrast: c - b, paired bootstrap over children
        contrast = None
        if "c_specific_task" in oof and "b_shared" in oof:
            key = "auc" if t.kind == "classification" else "r2"
            fn = (lambda yy, pp: roc_auc_score(yy, pp)) if key == "auc" else r2_score
            d0 = fn(yv, oof["c_specific_task"]) - fn(yv, oof["b_shared"])
            ds = []
            for _ in range(p.n_bootstrap):
                i = rng.integers(0, len(yv), len(yv))
                if key == "auc" and len(np.unique(yv[i])) < 2:
                    continue
                ds.append(fn(yv[i], oof["c_specific_task"][i]) - fn(yv[i], oof["b_shared"][i]))
            lo, hi = percentile_ci(np.array(ds), cfg.stats.ci_level)
            verdict = "supported" if lo > 0 else ("contradicted" if hi < 0 else "inconclusive")
            contrast = pd.DataFrame(
                [
                    {
                        "target": t.name,
                        "metric": key,
                        "specific_minus_shared": d0,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "h3_direction": verdict,
                    }
                ]
            )
        # permutations (5 x 5 outer CV, cached unsupervised steps)
        n_perm = p.n_permutations if n_perm is None else n_perm
        perm = self.permutation_test(t, y, n_perm) if n_perm else pd.DataFrame()
        if len(perm):
            res = res.merge(perm, on=["target", "rung"], how="left")
        oof_df = pd.DataFrame({"child": y.index, "y": yv, **{f"pred_{k}": v for k, v in oof.items()}})
        out = {"ladder": res, "oof": oof_df.assign(target=t.name)}
        if contrast is not None:
            out["h3"] = contrast
        if t.kind == "classification":
            out["calibration"] = pd.concat(
                [
                    calibration_curve(yv, v, p.calibration_bins).assign(rung=k, target=t.name)
                    for k, v in oof.items()
                ]
            )
        if t.heldout is not None:
            out["heldout"] = self.heldout(t, y)
        return out

    def permutation_test(self, t: Target, y: pd.Series, n_perm: int) -> pd.DataFrame:
        cfg = self.cfg
        reps = cfg.prediction.permutation_repeats
        obs = self.cv_predictions(t, y, reps, self.rungs, parallel=True)
        key = "auc" if t.kind == "classification" else "r2"
        fn = (lambda yy, pp: roc_auc_score(yy, pp)) if key == "auc" else r2_score
        yv = y.to_numpy()
        obs_stat = {r: fn(yv, np.nanmean(P, 0)) for r, P in obs.items()}
        rng, seed = generator(cfg.project.seed, "ladder_perm", t.name)
        seeds = rng.integers(0, 2**31, n_perm)
        # workers only need the folds of the permutation CV: ship those, without the raw feature matrix
        light = self._light([(f.train, f.test) for f in self._folds(t, y, reps)])

        def one(s):
            yp = pd.Series(np.random.default_rng(s).permutation(yv), index=y.index)
            P = light.cv_predictions(t, yp, reps, light.rungs, parallel=False)
            return {r: fn(yp.to_numpy(), np.nanmean(v, 0)) for r, v in P.items()}

        # the fold cache is already filled by the observed 5 x 5 pass, so workers only refit the classifier
        null = Parallel(n_jobs=cfg.runtime.n_jobs)(delayed(one)(int(s)) for s in seeds)
        rows = []
        for r in self.rungs:
            nd = np.array([d[r] for d in null])
            rows.append(
                {
                    "target": t.name,
                    "rung": r,
                    f"perm_{key}_5x5": obs_stat[r],
                    "p_perm": float((1 + (nd >= obs_stat[r]).sum()) / (1 + n_perm)),
                    "perm_seed": seed,
                    "n_perm": n_perm,
                }
            )
        return pd.DataFrame(rows)

    def permutation_test_restratified(
        self, t: Target, y: pd.Series, n_perm: int, rungs: list[str] | None = None
    ) -> pd.DataFrame:
        """Label permutation with the outer folds re-stratified on each permuted label vector.

        :meth:`permutation_test` keeps the folds stratified on the true labels. Under permuted labels those folds
        are no longer balanced on the labels being fitted, so a weak or empty rung's training prevalence (or mean)
        moves against each test child's label and the null falls below chance. Here every permutation draws its own
        folds stratified on the permuted labels, exactly as the observed statistic was computed. The unsupervised
        step is refitted per fold, which is affordable only without MOFA+, so the rungs are (a), (d), (e), (f).
        """
        cfg = self.cfg
        reps = cfg.prediction.permutation_repeats
        rungs = rungs or [r for r in self.rungs if r not in FACTOR_RUNGS]
        key = "auc" if t.kind == "classification" else "r2"
        fn = (lambda yy, pp: roc_auc_score(yy, pp)) if key == "auc" else r2_score
        yv = y.to_numpy()

        def strata(v: np.ndarray) -> pd.Series:
            if t.kind == "classification":
                return pd.Series(v, index=y.index)
            return pd.Series(
                pd.qcut(pd.Series(v).rank(method="first"), 5, labels=False).to_numpy(), index=y.index
            )

        def stat(v: np.ndarray) -> dict[str, float]:
            from threadpoolctl import threadpool_limits

            lad = copy.copy(self)
            lad.cache = FoldCache(cfg, self.X, use_mofa=False, missing_views=self.cache.missing_views)
            tp = Target(t.name, pd.Series(v, index=y.index), t.kind, strata(v))
            with threadpool_limits(1):
                P = lad.cv_predictions(tp, tp.y, reps, rungs, parallel=False)
            return {r: fn(v, np.nanmean(P[r], 0)) for r in rungs}

        obs = stat(yv)
        rng, seed = generator(cfg.project.seed, "ladder_perm_restratified", t.name)
        seeds = rng.integers(0, 2**31, n_perm)
        null = Parallel(n_jobs=cfg.runtime.n_jobs)(
            delayed(stat)(np.random.default_rng(int(s)).permutation(yv)) for s in seeds
        )
        rows = []
        for r in rungs:
            nd = np.array([d[r] for d in null])
            rows.append(
                {
                    "target": t.name,
                    "rung": r,
                    f"perm_{key}_5x5": obs[r],
                    "null_mean": float(nd.mean()),
                    "null_q95": float(np.quantile(nd, 0.95)),
                    "p_perm_restratified": float((1 + (nd >= obs[r]).sum()) / (1 + n_perm)),
                    "perm_seed": seed,
                    "n_perm": n_perm,
                }
            )
        return pd.DataFrame(rows)

    def heldout(self, t: Target, y: pd.Series) -> pd.DataFrame:
        """Train on every training child, predict the held-out set (siblings)."""
        cfg = self.cfg
        train = tuple(y.index)
        test = tuple(t.heldout.index)
        rep = self.cache.get(train, test)
        sf = SplitFactory(list(train), cfg.project.seed, strata=t.strata.reindex(list(train)).to_numpy())
        ytr = y.to_numpy()
        inner_strata = ytr if t.kind == "classification" else t.strata.reindex(list(train)).to_numpy()
        inner = sf.inner(train, cfg.prediction.inner_folds, strata=inner_strata)
        scoring = "roc_auc" if t.kind == "classification" else "neg_mean_squared_error"
        rows = []
        for rung in [r for r in self.rungs if r != "d_best_paradigm"]:
            Xtr, Xte = self._rep(rung, rep, train, test, self._cov_cols(t.name))
            p, _, _ = fit_predict(Xtr, y.to_numpy(), Xte, t.kind, cfg, inner, scoring)
            rows += [
                {
                    "target": t.name,
                    "rung": rung,
                    "child": c,
                    "pred": float(v),
                    "y": float(t.heldout.get(c, np.nan)),
                }
                for c, v in zip(test, p, strict=True)
            ]
        return pd.DataFrame(rows)

    def run(self, **inputs) -> Result:
        tables: dict[str, list] = {}
        for t in inputs["targets"]:
            log.info("ladder target %s (n=%d)", t.name, t.y.notna().sum())
            for k, v in self.run_target(t, inputs.get("n_perm")).items():
                tables.setdefault(k, []).append(v)
        out = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
        summ = {}
        if "h3" in out:
            summ["h3"] = out["h3"].to_dict("records")
        return Result("model_ladder", out, summ, label="estimation")


def main() -> None:
    """Fit the model ladder for all targets on the core set and write it to ``<results>/estimation/``."""
    from config.loader import load_config
    from utils.inputs import Inputs, make_targets
    from utils.store import ResultsStore

    cfg = load_config()
    inp = Inputs(cfg)
    core, meta = inp.core(), inp.meta()
    X = inp.child_matrix(core)
    lad = ModelLadder(cfg, X, meta)
    res = lad.run(targets=make_targets(meta, core, cfg))
    ResultsStore(cfg).write(res, "estimation")


if __name__ == "__main__":
    main()
