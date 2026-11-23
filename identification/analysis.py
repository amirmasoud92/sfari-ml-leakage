"""Cross-paradigm identification per feature family (Methods, Individual identification; Additional file 1,
Table S16, H1a; Fig. S1).

For every ordered pair of the six seated paradigms, child x feature matrices (mean over blocks) are z-scored within
paradigm and correlated between children; a child is identified if its own vector has the highest correlation.
The module gives accuracy, differential identifiability, a permutation null (2,000 relabellings of the target
paradigm's children), within-paradigm odd/even split-half identification (diagonal of Fig. S1a), per-child
self-identifiability, and the paired comparison of log power with the sensor-covariance and external-electrode
comparators (child bootstrap and sign-flip test). Running the module computes the identification result of all
families (``<results>/confirmatory/identification*``).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from config.loader import Config
from identification import core
from stats.inference import holm, paired_bootstrap_ci, sign_flip_p
from stats.rng import generator
from utils.base import Analysis, Result
from utils.logging_utils import get_logger

log = get_logger(__name__)


def _children_with(wide: pd.DataFrame, paradigm: str) -> set[str]:
    return set(wide.loc[wide.paradigm == paradigm, "child"])


class IdentificationAnalysis(Analysis):
    """Identification accuracy, differential identifiability, self-identifiability, nulls, ceilings and floors."""

    label = "confirmatory"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.paradigms = list(cfg.dataset.seated_paradigms)

    def family(
        self,
        name: str,
        wide: pd.DataFrame,
        children: list[str] | None = None,
        covariates: pd.DataFrame | None = None,
        n_perm: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """All ordered pairs of one family. ``children`` restricts the set (core); otherwise each pair uses every
        child with both paradigms (pairwise set). ``covariates`` (index child) are regressed out first."""
        cfg = self.cfg
        n_perm = cfg.identification.n_permutations if n_perm is None else n_perm
        rows, child_rows, si_rows = [], [], []
        null_by_pair = {}
        for src, tgt in itertools.permutations(self.paradigms, 2):
            kids = sorted(_children_with(wide, src) & _children_with(wide, tgt))
            if children is not None:
                kids = [k for k in kids if k in set(children)]
            if len(kids) < 5:
                continue
            A = core.paradigm_matrix(wide, src, kids)
            B = core.paradigm_matrix(wide, tgt, kids)
            if covariates is not None:
                C = covariates.reindex(kids).to_numpy(float)
                A, B = core.residualise(A, C), core.residualise(B, C)
            if cfg.identification.standardise == "zscore_within_paradigm":
                A, B = core.zscore_columns(A), core.zscore_columns(B)
            R = core.corr_matrix(A, B)
            rng, seed = generator(cfg.project.seed, "ident", name, src, tgt)
            perms = (
                np.array([rng.permutation(len(kids)) for _ in range(n_perm)]) if n_perm else np.zeros((0, 0))
            )
            null = core.permutation_accuracies(R, perms) if n_perm else np.zeros(0)
            acc = core.accuracy(R)
            null_by_pair[(src, tgt)] = null
            rows.append(
                {
                    "family": name,
                    "source": src,
                    "target": tgt,
                    "n": len(kids),
                    "accuracy": acc,
                    "chance": 1 / len(kids),
                    "diff_ident": core.differential_identifiability(R),
                    "null_mean": float(null.mean()) if null.size else np.nan,
                    "p_perm": float((1 + (null >= acc).sum()) / (1 + null.size)) if null.size else np.nan,
                    "seed": seed,
                }
            )
            ok = core.correct_rows(R)
            child_rows += [
                {"family": name, "child": k, "source": src, "target": tgt, "correct": bool(o)}
                for k, o in zip(kids, ok, strict=True)
            ]
            if src < tgt:  # unordered pair: self-identifiability z (mean of row and column views)
                z = core.self_identifiability_z(R)
                si_rows += [
                    {"family": name, "child": k, "pair": f"{src}|{tgt}", "si_z": float(v)}
                    for k, v in zip(kids, z, strict=True)
                ]
        pairs = pd.DataFrame(rows)
        if len(pairs) and n_perm:
            null_mean = np.mean(
                np.stack([null_by_pair[(r.source, r.target)] for r in pairs.itertuples()]), axis=0
            )
            obs = pairs["accuracy"].mean()
            fam = {
                "family": name,
                "mean_accuracy": obs,
                "mean_chance": pairs["chance"].mean(),
                "mean_diff_ident": pairs["diff_ident"].mean(),
                "null_mean": float(null_mean.mean()),
                "null_q95": float(np.quantile(null_mean, 0.95)),
                "p_perm": float((1 + (null_mean >= obs).sum()) / (1 + null_mean.size)),
                "n_pairs": len(pairs),
            }
        else:
            fam = {"family": name, "mean_accuracy": pairs["accuracy"].mean() if len(pairs) else np.nan}
        return {
            "pairs": pairs,
            "per_child": pd.DataFrame(child_rows),
            "self_ident": pd.DataFrame(si_rows),
            "family": pd.DataFrame([fam]),
        }

    def split_half(self, name: str, wide: pd.DataFrame, children: list[str] | None = None) -> pd.DataFrame:
        """Within-paradigm odd vs even block identification (reliability ceiling)."""
        rows = []
        for p in self.paradigms:
            d = wide[wide.paradigm == p]
            nb = d.groupby("child")["block"].nunique()
            kids = sorted(nb.index[nb >= 2])
            if children is not None:
                kids = [k for k in kids if k in set(children)]
            if len(kids) < 5:
                continue
            A = core.zscore_columns(core.paradigm_matrix(wide, p, kids, "odd"))
            B = core.zscore_columns(core.paradigm_matrix(wide, p, kids, "even"))
            R = core.corr_matrix(A, B)
            rows.append(
                {
                    "family": name,
                    "paradigm": p,
                    "n": len(kids),
                    "accuracy": core.accuracy(R),
                    "accuracy_reverse": core.accuracy(R.T),
                    "diff_ident": core.differential_identifiability(R),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def child_rates(per_child: pd.DataFrame) -> pd.Series:
        return per_child.groupby("child")["correct"].mean()

    def floor_comparison(self, fam: dict, floor: dict, fam_name: str, floor_name: str) -> dict:
        a = self.child_rates(fam["per_child"])
        b = self.child_rates(floor["per_child"])
        common = a.index.intersection(b.index)
        d = (a[common] - b[common]).to_numpy()
        rng, seed = generator(self.cfg.project.seed, "floor", fam_name, floor_name)
        lo, hi = paired_bootstrap_ci(d, self.cfg.identification.floor_bootstrap, rng, self.cfg.stats.ci_level)
        p = sign_flip_p(d, self.cfg.identification.floor_signflip, rng)
        return {
            "family": fam_name,
            "floor": floor_name,
            "n": int(common.size),
            "mean_rate_family": float(a[common].mean()),
            "mean_rate_floor": float(b[common].mean()),
            "difference": float(d.mean()),
            "ci_lo": lo,
            "ci_hi": hi,
            "p_signflip": p,
            "seed": seed,
        }

    def run(self, **inputs) -> Result:
        """``inputs``: ``bank`` (FeatureBank), ``families`` (names), ``floors``, ``core_children``, ``covariates``."""
        bank = inputs["bank"]
        families = inputs.get("families", [])
        floors = inputs.get("floors", ["broadband_cov", "exg_only"])
        controls = inputs.get("controls", [])
        core_kids = inputs.get("core_children")
        cov = inputs.get("covariates")
        out: dict[str, list[pd.DataFrame]] = {
            k: [] for k in ("pairs", "per_child", "self_ident", "family", "split_half", "floors", "regressed")
        }
        res = {}
        for name in [*families, *floors, *controls]:
            w = bank.wide(name)
            log.info("identification %s", name)
            r = self.family(name, w)
            res[name] = r
            for k in ("pairs", "per_child", "family"):
                out[k].append(r[k])
            out["split_half"].append(self.split_half(name, w))
            if core_kids is not None:
                rc = self.family(name, w, children=core_kids, n_perm=0)
                out["self_ident"].append(rc["self_ident"].assign(set="core"))
            if cov is not None and name in families:
                rr = self.family(name, w, covariates=cov, n_perm=0)
                out["regressed"].append(rr["pairs"].assign(variant="age_sex_regressed"))
        for name in families:
            for fl in floors:
                out["floors"].append(pd.DataFrame([self.floor_comparison(res[name], res[fl], name, fl)]))
        tables = {k: pd.concat(v, ignore_index=True) for k, v in out.items() if v}
        fam = tables["family"]
        shared = fam[fam.family.isin(families)].copy()
        shared["p_holm"] = holm(shared["p_perm"].to_numpy())
        tables["family"] = fam.merge(shared[["family", "p_holm"]], on="family", how="left")
        prim = self.cfg.identification.primary_family
        summary = {"primary_family": prim, "families": families, "floors": floors}
        if prim in res:
            f = tables["family"].set_index("family").loc[prim]
            fl = tables["floors"][tables["floors"].family == prim]
            summary["h1a_primary"] = {
                "mean_accuracy": float(f["mean_accuracy"]),
                "p_perm": float(f["p_perm"]),
                "floor_lower_bounds": fl.set_index("floor")["ci_lo"].to_dict(),
                "supported": bool(f["p_perm"] < 0.05 and (fl["ci_lo"] > 0).all()),
            }
        return Result("identification", tables, summary, label="confirmatory")


def main() -> None:
    """Identification of every feature family, with the comparators and the coherence controls, and the core-set
    self-identifiability (``<results>/confirmatory/identification*``)."""
    from config.loader import load_config
    from features.bank import SHARED_FAMILIES
    from utils.inputs import Inputs
    from utils.store import ResultsStore

    cfg = load_config()
    inp = Inputs(cfg)
    core, meta = inp.core(), inp.meta()
    res = IdentificationAnalysis(cfg).run(
        bank=inp.bank,
        families=list(SHARED_FAMILIES),
        floors=["broadband_cov", "exg_only"],
        controls=["coherence_theta", "coherence_alpha", "coherence_beta"],
        core_children=core,
        covariates=meta[["age", "sex_male"]],
    )
    ResultsStore(cfg).write(res, "confirmatory")


if __name__ == "__main__":
    main()
