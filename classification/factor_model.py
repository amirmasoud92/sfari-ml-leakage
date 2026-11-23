"""Multi-view factor model of the model ladder (Additional file 1, Supplementary Methods S3 and Table S13).

``MultiViewFactorModel``: MOFA+ with one view per paradigm, factor labels (shared, partially shared, task-specific,
rest-specific) from the per-view variance explained, out-of-sample posterior-mean scores, bootstrap stability, and a
concatenated-PCA fallback.

``ViewReducer`` turns the child-level feature table (columns ``(view, family, feature)``) into the factor-model input:
every feature z-scored on the training children (missing values imputed with training means), then per view and
family a PCA to at most ``n_components`` components (scores centred, not whitened); paradigm-specific columns
(evoked, behaviour) kept as z-scores. Both are fitted on training children only. With ``missing_views="nan"``
(all-available children) a child without any value in a (view, family) block gets NaN for that block instead of the
training mean, is left out of that block's PCA fit, and MOFA+ treats the block as missing.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

from config.loader import Config
from stats.rng import generator
from utils.base import Analysis, Result
from utils.logging_utils import get_logger

log = get_logger(__name__)
RAW_FAMILIES = ("evoked", "behaviour")


class ViewReducer:
    """Per (view, family) PCA reduction fitted on training children (see the module docstring)."""

    def __init__(
        self,
        n_components: int = 10,
        views: list[str] | None = None,
        families: list[str] | None = None,
        missing_views: str = "impute",
    ):
        if missing_views not in ("impute", "nan"):
            raise ValueError(f"missing_views must be 'impute' or 'nan', not {missing_views!r}")
        self.n_components = n_components
        self.views = views
        self.families = families
        self.missing_views = missing_views

    def _blocks(self, X: pd.DataFrame):
        cols = X.columns
        for v in cols.get_level_values(0).unique():
            if self.views is not None and v not in self.views:
                continue
            for f in cols[cols.get_level_values(0) == v].get_level_values(1).unique():
                if self.families is not None and f not in self.families:
                    continue
                yield v, f, X.loc[:, (v, f)]

    def fit(self, X: pd.DataFrame) -> ViewReducer:
        self.steps_ = []
        for v, f, B in self._blocks(X):
            A = B.to_numpy(float)
            rows = self._present(A)
            if rows.sum() < 3:
                continue
            A = A[rows]
            keep = np.isfinite(A).mean(axis=0) >= 0.5
            mu = np.nanmean(A[:, keep], axis=0)
            Ai = np.where(np.isfinite(A[:, keep]), A[:, keep], mu)
            sd = Ai.std(axis=0)
            sd[sd == 0] = 1
            Z = (Ai - mu) / sd
            if f in RAW_FAMILIES or Z.shape[1] <= self.n_components:
                pca = None
                out = Z
            else:
                pca = PCA(n_components=min(self.n_components, Z.shape[0] - 1), random_state=0).fit(Z)
                out = pca.transform(Z)
            # PC scores keep their variance (whitening them would give noise PCs the weight of signal PCs; MOFA+
            # has a noise precision per feature, so scale differences between families are absorbed)
            osd = np.ones(out.shape[1])
            self.steps_.append((v, f, B.columns, keep, mu, sd, pca, out.mean(axis=0), osd))
        return self

    def _present(self, A: np.ndarray) -> np.ndarray:
        """Rows that take part in a block: all rows (``impute``) or rows with at least one value (``nan``)."""
        if self.missing_views == "impute":
            return np.ones(len(A), bool)
        return np.isfinite(A).any(axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        parts, names = [], []
        blocks = set(zip(X.columns.get_level_values(0), X.columns.get_level_values(1), strict=True))
        for v, f, cols, keep, mu, sd, pca, om, osd in self.steps_:
            if (v, f) in blocks:
                A = X.loc[:, (v, f)].reindex(columns=cols).to_numpy(float)
            else:
                A = np.full((len(X), len(cols)), np.nan)
            rows = self._present(A)
            A = A[:, keep]
            Z = (np.where(np.isfinite(A), A, mu) - mu) / sd
            out = (pca.transform(Z) if pca is not None else Z) - om
            out[~rows] = np.nan
            parts.append(out / osd)
            k = out.shape[1]
            names += [
                (v, f, f"pc{i}" if pca is not None else str(c))
                for i, c in zip(range(k), np.asarray(cols)[keep], strict=False)
            ]
        return pd.DataFrame(np.hstack(parts), index=X.index, columns=pd.MultiIndex.from_tuples(names))

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class MultiViewFactorModel:
    """MOFA+ (mofapy2) on views given as the first column level of a DataFrame."""

    def __init__(
        self,
        n_factors: int = 15,
        drop_r2: float = 0.01,
        active_r2: float = 0.01,
        seed: int = 0,
        convergence_mode: str = "medium",
        max_iter: int = 1000,
        method: str = "mofa",
        scale_views: bool = False,
        n_restarts: int = 5,
    ):
        self.n_factors = n_factors
        self.drop_r2 = drop_r2
        self.active_r2 = active_r2
        self.seed = seed
        self.convergence_mode = convergence_mode
        self.max_iter = max_iter
        self.method = method
        self.scale_views = scale_views
        self.n_restarts = n_restarts

    def fit(self, X: pd.DataFrame) -> MultiViewFactorModel:
        self.views_ = list(X.columns.get_level_values(0).unique())
        self.columns_ = {v: X.loc[:, v].columns for v in self.views_}
        data = [X.loc[:, v].to_numpy(float) for v in self.views_]
        if self.method == "mofa":
            try:
                self._fit_mofa(data)
            except Exception as e:  # noqa: BLE001 - logged; fallback: concatenated PCA
                log.warning("MOFA+ failed (%s): concatenated PCA fallback", e)
                self._fit_pca(data)
        else:
            self._fit_pca(data)
        self._label()
        return self

    def _fit_once(self, data: list[np.ndarray], seed: int):
        from mofapy2.run.entry_point import entry_point

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ent = entry_point()
            ent.set_data_options(scale_views=self.scale_views, scale_groups=False)
            ent.set_data_matrix([[d] for d in data])
            ent.set_model_options(factors=self.n_factors, spikeslab_weights=True, ard_weights=True)
            ent.set_train_options(
                iter=self.max_iter,
                convergence_mode=self.convergence_mode,
                dropR2=self.drop_r2,
                seed=seed,
                verbose=False,
                quiet=True,
            )
            ent.build()
            ent.run()
        m = ent.model
        stats = m.getTrainingStats() if hasattr(m, "getTrainingStats") else {}
        elbo = np.asarray(stats.get("elbo", [np.nan]) if isinstance(stats, dict) else [np.nan], float)
        fin = elbo[np.isfinite(elbo)]
        return m, (float(fin[-1]) if fin.size else -np.inf), int(fin.size)

    def _fit_mofa(self, data: list[np.ndarray]) -> None:
        """Best of ``n_restarts`` seeded fits by final ELBO (a single random start can miss a strong view-specific
        factor at fold sizes of about 60 children)."""
        rng = np.random.default_rng(self.seed)
        fits = [self._fit_once(data, int(s)) for s in rng.integers(1, 2**31 - 1, self.n_restarts)]
        m, elbo, n_iter = max(fits, key=lambda f: f[1])
        ex = m.getExpectations()
        self.Z_train_ = np.asarray(ex["Z"]["E"])
        self.W_ = [np.asarray(w["E"]) for w in ex["W"]]
        self.tau_ = [np.nanmean(np.asarray(t["E"]), axis=0) for t in ex["Tau"]]
        self.r2_ = np.array(m.calculate_variance_explained()[0])  # (views, factors)
        self.method_used_ = "mofa"
        self.elbo_ = elbo
        self.restart_elbos_ = [f[1] for f in fits]
        self.converged_ = bool(n_iter < self.max_iter)
        self.n_iter_ = n_iter

    def _fit_pca(self, data: list[np.ndarray]) -> None:
        Xc = np.hstack([np.nan_to_num(d) for d in data])
        k = min(self.n_factors, Xc.shape[0] - 1)
        pca = PCA(n_components=k, random_state=self.seed).fit(Xc)
        sizes = np.cumsum([0] + [d.shape[1] for d in data])
        comps = pca.components_.T  # (D, k)
        Zt = pca.transform(Xc)
        self.Z_train_ = Zt
        self.W_ = [
            comps[sizes[i] : sizes[i + 1]] * np.sqrt(pca.explained_variance_) for i in range(len(data))
        ]
        self.tau_ = [np.ones(d.shape[1]) for d in data]
        r2 = []
        for i, d in enumerate(data):
            dv = np.nan_to_num(d)
            tot = (dv**2).sum()
            r2.append(
                [(np.outer(Zt[:, j], comps[sizes[i] : sizes[i + 1], j]) ** 2).sum() / tot for j in range(k)]
            )
        self.r2_ = np.array(r2)
        self.method_used_ = "concat_pca"
        self.converged_ = True
        self.n_iter_ = 0

    def _label(self) -> None:
        act = self.r2_ >= self.active_r2
        n_act = act.sum(axis=0)
        nv = len(self.views_)
        self.labels_ = np.where(
            n_act == nv,
            "shared",
            np.where(n_act == 1, "specific", np.where(n_act == 0, "inactive", "partial")),
        )
        self.active_views_ = [
            [self.views_[i] for i in np.flatnonzero(act[:, j])] for j in range(act.shape[1])
        ]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Posterior-mean factor scores under the training weights (missing entries skipped)."""
        K = self.W_[0].shape[1]
        out = np.zeros((len(X), K))
        for n in range(len(X)):
            prec = np.eye(K)
            rhs = np.zeros(K)
            for v, W, tau in zip(self.views_, self.W_, self.tau_, strict=True):
                y = X.loc[:, v].reindex(columns=self.columns_[v]).to_numpy(float)[n]
                ok = np.isfinite(y)
                Wo = W[ok] * tau[ok, None]
                prec += W[ok].T @ Wo
                rhs += Wo.T @ y[ok]
            out[n] = np.linalg.solve(prec, rhs)
        return out

    def factor_index(self, kind: str, rest_view: str = "Restingstate") -> np.ndarray:
        if kind == "all":
            return np.flatnonzero(self.labels_ != "inactive")
        if kind == "specific_task":
            return np.array(
                [
                    j
                    for j, (lab, av) in enumerate(zip(self.labels_, self.active_views_, strict=True))
                    if lab == "specific" and av[0] != rest_view
                ],
                int,
            )
        if kind == "specific_rest":
            return np.array(
                [
                    j
                    for j, (lab, av) in enumerate(zip(self.labels_, self.active_views_, strict=True))
                    if lab == "specific" and av[0] == rest_view
                ],
                int,
            )
        return np.flatnonzero(self.labels_ == kind)

    def summary(self) -> pd.DataFrame:
        rows = []
        for j in range(self.r2_.shape[1]):
            rows.append(
                {
                    "factor": j,
                    "label": self.labels_[j],
                    "active_views": ",".join(self.active_views_[j]),
                    **{f"r2_{v}": float(self.r2_[i, j]) for i, v in enumerate(self.views_)},
                }
            )
        return pd.DataFrame(rows)


def match_factors(W_ref: list[np.ndarray], W_boot: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian matching of bootstrap to reference factors by |corr| of concatenated loadings."""
    A = np.vstack(W_ref)
    B = np.vstack(W_boot)
    A = (A - A.mean(0)) / np.where(A.std(0) > 0, A.std(0), 1)
    B = (B - B.mean(0)) / np.where(B.std(0) > 0, B.std(0), 1)
    C = np.abs(A.T @ B) / A.shape[0]
    r, c = linear_sum_assignment(-C)
    out = np.full(A.shape[1], np.nan)
    out[r] = C[r, c]
    return out, c


class FactorModelAnalysis(Analysis):
    """Full-sample MOFA+ on the core children with threshold sensitivity and bootstrap stability (estimation)."""

    label = "estimation"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    def model(self, seed: int | None = None, active_r2: float | None = None) -> MultiViewFactorModel:
        lc = self.cfg.latent
        return MultiViewFactorModel(
            n_factors=lc.max_factors,
            drop_r2=lc.drop_factor_threshold,
            active_r2=lc.active_r2 if active_r2 is None else active_r2,
            seed=self.cfg.project.seed if seed is None else seed,
            convergence_mode=lc.convergence_mode,
            method=lc.method,
        )

    def run(self, **inputs) -> Result:
        X = inputs["X"]
        lc = self.cfg.latent
        red = ViewReducer(lc.view_pca_components, missing_views=inputs.get("missing_views", "impute")).fit(X)
        R = red.transform(X)
        m = self.model().fit(R)
        summ = m.summary()
        sens = []
        for thr in lc.active_r2_sensitivity:
            mm = self.model(active_r2=thr)
            mm.views_, mm.r2_ = m.views_, m.r2_
            mm._label()
            sens.append(pd.Series(mm.labels_).value_counts().rename(f"thr_{thr}"))
        rng, seed = generator(self.cfg.project.seed, "mofa_bootstrap")
        n_boot = int(inputs.get("n_boot", lc.n_bootstrap))
        draws = [(rng.integers(0, len(R), len(R)), int(rng.integers(1e9))) for _ in range(n_boot)]
        from joblib import Parallel, delayed

        def one(idx, s):
            mb = self.model(seed=s).fit(R.iloc[idx])
            # a refit fails if MOFA+ raised (PCA fallback) or did not converge
            bad = (mb.method_used_ != m.method_used_) or not mb.converged_
            if not mb.W_[0].shape[1]:
                return np.full(len(m.labels_), np.nan), bad
            return match_factors(m.W_, mb.W_)[0], bad

        out = Parallel(n_jobs=self.cfg.runtime.n_jobs)(delayed(one)(i, s) for i, s in draws)
        failed = int(sum(o[1] for o in out))
        if (
            m.method_used_ == "mofa"
            and failed / max(n_boot, 1) > lc.max_failed_fraction
            and not inputs.get("_fallback")
        ):
            log.warning("%d of %d bootstrap refits failed: concatenated PCA", failed, n_boot)
            lat = lc.model_copy(update={"method": "concat_pca"})
            alt = FactorModelAnalysis(self.cfg.model_copy(update={"latent": lat}))
            res = alt.run(**{**inputs, "_fallback": True})
            res.summary["mofa_bootstrap_failed_fraction"] = failed / n_boot
            return res
        # a reference factor that a refit does not reproduce counts as r = 0, not as missing
        stab = np.nan_to_num(np.array([o[0] for o in out]), nan=0.0)
        summ["stability_median_r"] = np.median(stab, axis=0)
        summ["recovered_fraction"] = np.mean(stab > lc.stability_match_r, axis=0)
        loadings = []
        for v, W in zip(m.views_, m.W_, strict=True):
            cols = R.loc[:, v].columns
            for i, c in enumerate(cols):
                loadings.append(
                    {
                        "view": v,
                        "family": c[0],
                        "component": c[1],
                        **{f"w{j}": float(W[i, j]) for j in range(W.shape[1])},
                    }
                )
        info = {
            "method_used": m.method_used_,
            "converged": m.converged_,
            "n_factors": int(len(m.labels_)),
            "label_counts": pd.Series(m.labels_).value_counts().to_dict(),
            "bootstrap_failed_fraction": failed / max(n_boot, 1),
            "bootstrap_seed": seed,
        }
        scores = pd.DataFrame(
            m.Z_train_, index=R.index, columns=[f"F{j + 1}" for j in range(m.Z_train_.shape[1])]
        )
        return Result(
            "factor_model",
            {
                "factors": summ,
                "threshold_sensitivity": pd.concat(sens, axis=1).reset_index(names="label"),
                "loadings": pd.DataFrame(loadings),
                "structure": structure_coefficients(m.Z_train_, X),
                "scores": scores.reset_index(names="child"),
            },
            info,
            label="estimation",
        )


def structure_coefficients(
    Z: np.ndarray, X: pd.DataFrame, families: tuple[str, ...] = ("aperiodic", "periodic", "alpha_cf")
) -> pd.DataFrame:
    """Correlation of each factor's scores with each channel-level feature (``<ch>_<kind>``) of the given
    families in every view: the factor 'loading topographies' of Figure 3."""
    Zc = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    cols = [c for c in X.columns if c[1] in families]
    A = X[cols].to_numpy(float)
    mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
    A = np.where(np.isfinite(A), (A - mu) / np.where(sd > 0, sd, 1), 0.0)
    R = Zc.T @ A / len(A)
    rows = []
    for j in range(R.shape[0]):
        for k, (v, fam, feat) in enumerate(cols):
            ch, kind = str(feat).rsplit("_", 1)
            rows.append(
                {"factor": j, "view": v, "family": fam, "kind": kind, "channel": ch, "r": float(R[j, k])}
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Fit the factor model on the child-level matrix of the core set and write it to ``<results>/estimation/``."""
    from config.loader import load_config
    from utils.inputs import Inputs
    from utils.store import ResultsStore

    cfg = load_config()
    inp = Inputs(cfg)
    res = FactorModelAnalysis(cfg).run(X=inp.child_matrix(inp.core()))
    ResultsStore(cfg).write(res, "estimation")


if __name__ == "__main__":
    main()
