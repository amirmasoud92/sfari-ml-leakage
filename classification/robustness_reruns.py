"""Robustness reruns of identification, self-identifiability, variance components and the model ladder
(Additional file 1, Table S8).

Every rerun uses the main analysis classes with modified inputs: preprocessing without artefact subspace
reconstruction (feature bank of the ``no_asr`` variant), alternative equal-duration blocks, sex-matched (age within
sex) and IQ-matched subsets (greedy 1:1 nearest-neighbour matching of TD to autistic children, caliper 0.5 SD),
medication-free children, all children with at least two paradigms (MOFA+ with missing views), and three
alternative all-feature models (spectral, tangent-space and per-child re-centred tangent-space logistic regression).
Further reruns that drop one feature family or one paradigm at a time and the two convolutional networks and the
subject-contrastive encoder (:mod:`deep_learning`) are run through the same entry point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from classification.model_ladder import ModelLadder
from config.loader import Config
from features.bank import SHARED_FAMILIES, FeatureBank
from identification.analysis import IdentificationAnalysis
from identification.self_identifiability import PRIMARY_COVARIATES, contrast, si_table
from utils.base import Result
from utils.inputs import Inputs, make_targets
from utils.logging_utils import get_logger
from utils.store import ResultsStore
from variance_partition.components import VarianceComponents

log = get_logger(__name__)
LADDER_RUNGS = ["a_covariates", "b_shared", "c_specific_task", "e_all"]


def _ladder(
    cfg: Config,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    kids: list[str],
    tag: str,
    rungs: list[str] | None = None,
    use_mofa: bool = True,
    missing_views: str = "impute",
) -> pd.DataFrame:
    t = make_targets(meta, kids, cfg)[0]
    t.heldout = None
    lad = ModelLadder(
        cfg,
        X.loc[[k for k in X.index if k in set(kids)]],
        meta,
        use_mofa=use_mofa,
        missing_views=missing_views,
    )
    if rungs is not None:
        lad.rungs = [r for r in rungs if r in lad.rungs or r == "e_all"]
    out = lad.run_target(t, n_perm=0)
    res = out["ladder"].assign(variant=tag)
    if "h3" in out:
        res = res.merge(
            out["h3"][["target", "specific_minus_shared", "ci_lo", "ci_hi", "h3_direction"]].rename(
                columns={"ci_lo": "h3_ci_lo", "ci_hi": "h3_ci_hi"}
            ),
            on="target",
            how="left",
        )
    return res


def _h2(
    cfg: Config,
    bank: FeatureBank,
    kids: list[str],
    meta: pd.DataFrame,
    tag: str,
    cols: list[str] | None = None,
    gallery: list[str] | None = None,
) -> dict:
    """Self-identifiability contrast on ``kids``. Self-identifiability is computed against ``gallery`` (default:
    ``kids``); for subsets of the core set the gallery is the core set, so that the comparison set is the one of the
    main analysis.
    """
    fam = cfg.identification.primary_family
    r = IdentificationAnalysis(cfg).family(fam, bank.wide(fam), children=gallery or kids, n_perm=0)
    si = si_table(r["self_ident"].assign(set="core"), fam)
    df = meta.loc[kids].join(si[si.index.isin(kids)].rename("si"), how="inner")
    return {
        "variant": tag,
        **contrast(df, cols or PRIMARY_COVARIATES, cfg, n_perm=0, n_boot=1000, key=f"s6_{tag}"),
    }


def _ident(cfg: Config, bank: FeatureBank, tag: str, paradigms: list[str] | None = None) -> pd.DataFrame:
    ia = IdentificationAnalysis(cfg)
    if paradigms is not None:
        ia.paradigms = paradigms
    rows = []
    for fam in SHARED_FAMILIES:
        r = ia.family(fam, bank.wide(fam), n_perm=200)
        rows.append(r["family"].assign(variant=tag))
    return pd.concat(rows, ignore_index=True)


def matched(
    meta: pd.DataFrame,
    kids: list[str],
    on: str,
    within: str | None = None,
    caliper: float = 0.5,
    seed: int = 0,
) -> list[str]:
    """Greedy 1:1 nearest-neighbour matching of TD to ASD on ``on`` (within ``within`` strata), caliper in SD."""
    m = meta.loc[kids].dropna(subset=[on])
    rng = np.random.default_rng(seed)
    sd = m[on].std()
    out = []
    strata = [None] if within is None else m[within].dropna().unique()
    for s in strata:
        mm = m if s is None else m[m[within] == s]
        asd = mm[mm.group == "ASD"].sample(frac=1, random_state=int(rng.integers(1e9)))
        td = mm[mm.group == "TD"].copy()
        for c, row in asd.iterrows():
            if td.empty:
                break
            d = (td[on] - row[on]).abs()
            j = d.idxmin()
            if d[j] <= caliper * sd:
                out += [c, j]
                td = td.drop(j)
    return sorted(out)


def run_reruns(cfg: Config, steps: tuple[str, ...]) -> dict:
    """Run the robustness reruns named in ``steps`` and write their tables to ``<results>/controls/``."""
    store = ResultsStore(cfg)
    inp = Inputs(cfg)
    core = inp.core()
    meta = inp.meta()
    X = inp.child_matrix(core)
    asd_td = [k for k in core if meta.loc[k, "group"] in ("ASD", "TD")]
    tables: dict[str, list] = {}

    def add(key, df):
        tables.setdefault(key, []).append(df)

    if "family_ablation" in steps:
        for fam in SHARED_FAMILIES:
            keep = [c for c in X.columns if c[1] != fam]
            add("ladder", _ladder(cfg, X[keep], meta, core, f"without {fam}", LADDER_RUNGS))
    if "paradigm_ablation" in steps:
        for p in inp.paradigms:
            keep = [c for c in X.columns if c[0] != p]
            add("ladder", _ladder(cfg, X[keep], meta, core, f"without {p}", LADDER_RUNGS))
            add("identification", _ident(cfg, inp.bank, f"without {p}", [q for q in inp.paradigms if q != p]))
    for variant, block_mode, key in (
        ("no_asr", "primary", "no_asr"),
        ("primary", "alternative", "alt_blocks"),
    ):
        if key not in steps:
            continue
        bank = FeatureBank(cfg, variant, block_mode)
        if not (bank.root / "bank_summary.json").is_file():
            log.warning("%s: feature bank %s missing; build it first", key, bank.root)
            continue
        add("identification", _ident(cfg, bank, key))
        add("h2", pd.DataFrame([_h2(cfg, bank, core, meta, key)]))
        vin = Inputs(cfg, bank)
        vc = VarianceComponents(cfg)
        add("h1b", vc.h1b(vin.roi_scalar(core), n_boot=200).assign(variant=key))
        add("ladder", _ladder(cfg, vin.child_matrix(core), meta, core, key, LADDER_RUNGS))
    if "matched" in steps:
        sexm = matched(meta, asd_td, "age", within="sex_male", seed=cfg.project.seed)
        iqm = matched(meta, asd_td, "fsiq", seed=cfg.project.seed)
        for tag, kids in (("sex-matched (age within sex)", sexm), ("IQ-matched", iqm)):
            sib = [k for k in core if meta.loc[k, "group"] == "SIB"]
            add("h2", pd.DataFrame([_h2(cfg, inp.bank, kids + sib, meta, tag, gallery=core)]))
            add("ladder", _ladder(cfg, X, meta, kids, tag, LADDER_RUNGS))
            add("samples", pd.DataFrame({"variant": tag, "child": kids}))
    if "medication" in steps:
        none = [k for k in core if meta.loc[k, "medication_class"] == "none"]
        add(
            "h2",
            pd.DataFrame(
                [
                    _h2(
                        cfg,
                        inp.bank,
                        core,
                        meta,
                        "medication covariate",
                        PRIMARY_COVARIATES + ["med_stimulant_or_ssri", "med_other", "med_unknown"],
                    )
                ]
            ),
        )
        add("h2", pd.DataFrame([_h2(cfg, inp.bank, none, meta, "medication none only", gallery=core)]))
        add("ladder", _ladder(cfg, X, meta, none, "medication none only", LADDER_RUNGS))
    if "baselines" in steps:
        for fam_set, tag in (
            (["spectral_logpow"], "spectral LR"),
            (["tangent_theta", "tangent_alpha", "tangent_beta"], "tangent-space LR"),
        ):
            keep = [c for c in X.columns if c[1] in fam_set]
            add("ladder", _ladder(cfg, X[keep], meta, core, tag, ["e_all"], use_mofa=False))
    if "all_available" in steps:
        for k, df in all_available(cfg, inp, meta).items():
            add(k, df)
    if "deep" in steps:
        from deep_learning.convnets import run_deep

        lab = (meta.loc[asd_td, "group"] == "ASD").astype(int)
        add("deep", run_deep(cfg, lab, inp.paradigms))
    if "recentred" in steps:
        R = recentred_tangent_matrix(cfg, inp, core)
        keep = [c for c in X.columns if not c[1].startswith("tangent")]
        add(
            "ladder",
            _ladder(
                cfg,
                pd.concat([X[keep], R], axis=1),
                meta,
                core,
                "Riemannian re-centred tangent",
                ["e_all"],
                use_mofa=False,
            ),
        )
    # the subject-contrastive encoder writes its own result
    separate = {}
    if "encoder" in steps:
        from deep_learning.contrastive_encoder import run_encoder

        separate["encoder"] = run_encoder(cfg, inp, meta, core, make_targets(meta, core, cfg))
    for r in separate.values():
        store.write(r, "controls")
    out = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
    if out:
        # a run of the convolutional networks alone writes its own result, so it never replaces the main tables
        own = {"encoder"}
        name = "convnets" if set(steps) - own == {"deep"} else "robustness_reruns"
        store.write(Result(name, out, {"steps": list(steps)}, label="estimation"), "controls")
    return {**{k: len(v) for k, v in out.items()}, **{k: r.summary for k, r in separate.items()}}


def all_available(cfg: Config, inp: Inputs, meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All children with at least two kept seated paradigms: MOFA+ with missing views, the self-identifiability
    contrast averaged over each child's available pairs (number of pairs as an extra covariate) and the
    autistic-versus-TD ladder with missing views."""
    fam = cfg.identification.primary_family
    wide = inp.bank.wide(fam)
    n_par = wide.groupby("child").paradigm.nunique()
    kids = sorted(c for c in n_par.index[n_par >= 2] if c in meta.index and pd.notna(meta.loc[c, "group"]))
    X = inp.child_matrix(kids)
    log.info(
        "all-available: %d children (%s)", len(kids), meta.loc[kids, "group"].value_counts().to_dict()
    )
    from classification.factor_model import FactorModelAnalysis

    fm = FactorModelAnalysis(cfg).run(X=X, missing_views="nan", n_boot=min(100, cfg.latent.n_bootstrap))
    ResultsStore(cfg).write(
        Result("factor_model_all_available", fm.tables, fm.summary, label="estimation"), "controls"
    )
    si = IdentificationAnalysis(cfg).family(fam, wide, children=kids, n_perm=0)["self_ident"]
    per = si.groupby("child").agg(si=("si_z", "mean"), n_pairs=("pair", "nunique"))
    df = meta.loc[kids].join(per, how="inner")
    h2 = contrast(df, PRIMARY_COVARIATES + ["n_pairs"], cfg, n_perm=0, n_boot=1000, key="s6_all_available")
    factors = fm.tables["factors"].assign(variant="all available")
    return {
        "h2": pd.DataFrame([{"variant": "all available", **h2}]),
        "ladder": _ladder(cfg, X, meta, kids, "all available", LADDER_RUNGS, missing_views="nan"),
        "factor_labels": factors,
        "samples": pd.DataFrame(
            {"variant": "all available", "child": kids, "n_paradigms": n_par[kids].values}
        ),
    }


def recentred_tangent_matrix(cfg: Config, inp: Inputs, kids: list[str]) -> pd.DataFrame:
    """Per-child Riemannian re-centring (each child's block covariances whitened by that child's own Riemannian
    mean over all paradigms), tangent space at the identity, block means per paradigm (ladder ablation only).
    """
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.tangentspace import tangent_space

    from features.spectra import spectra_path

    parts = []
    for band in ("theta", "alpha", "beta"):
        rows = {}
        for c in kids:
            covs, pars = [], []
            for p in inp.paradigms:
                f = spectra_path(cfg, c, p)
                if not f.is_file():
                    continue
                with np.load(f) as z:
                    C = z[f"cov_{band}"].astype(float)
                covs.append(C)
                pars += [p] * len(C)
            if not covs:
                continue
            C = np.concatenate(covs)
            W = invsqrtm(mean_riemann(C))
            T = tangent_space(W @ C @ W, np.eye(C.shape[1]))
            df = pd.DataFrame(T)
            df["paradigm"] = pars
            for p, g in df.groupby("paradigm"):
                rows[(c, p)] = g.drop(columns="paradigm").mean().to_numpy()
        for p in inp.paradigms:
            m = {c: v for (c, q), v in rows.items() if q == p}
            if not m:
                continue
            block = pd.DataFrame(m).T.reindex(kids)
            block.columns = pd.MultiIndex.from_tuples(
                [(p, f"tangent_{band}_recentred", f"t{i}") for i in range(block.shape[1])]
            )
            parts.append(block)
    return pd.concat(parts, axis=1)


if __name__ == "__main__":
    from config.loader import load_config

    # "deep" (convolutional networks) and "encoder" (subject-contrastive encoder) are run the same way
    run_reruns(
        load_config(),
        (
            "no_asr",
            "alt_blocks",
            "matched",
            "medication",
            "baselines",
            "all_available",
            "recentred",
            "family_ablation",
            "paradigm_ablation",
        ),
    )
