"""Inputs shared by the child-level analyses.

:class:`Inputs` builds, from the feature bank, the preprocessing QC table and ``participants.tsv``:

- the core set (children with all six seated paradigms kept; Table 1),
- the child metadata: covariates, group, outcomes and the three data-quality measures (clean minutes over the six
  paradigms, interpolated-channel fraction and fraction changed by artefact subspace reconstruction),
- the child-level feature matrix of the model ladder (block means of every feature family per paradigm plus the
  evoked and behavioural indices),
- the posterior region-of-interest scalars of the variance-components model and band-averaged log power.

:func:`make_targets` defines the prediction targets (autistic versus TD; SRS-2, MABC-2 and FSIQ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from classification.model_ladder import Target
from config.loader import Config
from features.bank import SHARED_FAMILIES, FeatureBank
from utils.dataset import BidsDataset, Participants
from utils.logging_utils import get_logger

log = get_logger(__name__)
CHILD_FAMILIES = (*SHARED_FAMILIES, "evoked", "behaviour")


def data_quality(qc: pd.DataFrame, paradigms: list[str]) -> pd.DataFrame:
    """Per-child data quality over the kept recordings of ``paradigms`` (the seated paradigms). The standing motor
    recordings (another amplifier) share the QC table but are not part of the covariates."""
    q = qc[(qc.status == "ok") & qc.paradigm.isin(paradigms)].groupby("child")
    return pd.DataFrame(
        {
            "clean_minutes": q["clean_seconds"].sum() / 60,
            "artefact_free_trials": q["n_trials_artefact_free"].sum(),
            "mean_bad_fraction": q["n_bad_channels"].mean() / 64,
            "mean_asr_fraction": q["asr_repaired_fraction"].mean(),
            "mean_ica_removed": q["ica_removed"].mean(),
        }
    )


class Inputs:
    """Core set, child metadata and child-level matrices, built once."""

    def __init__(self, cfg: Config, bank: FeatureBank | None = None) -> None:
        self.cfg = cfg
        self.bank = bank or FeatureBank(cfg)
        ds = BidsDataset.from_config(cfg)
        self.parts = Participants.load(ds, cfg)
        self.paradigms = list(cfg.dataset.seated_paradigms)
        self.qc = pd.read_csv(cfg.path("results") / "preprocessing" / "qc_summary.csv")

    def ok_children(self, paradigm: str) -> set[str]:
        q = self.qc
        return set(q.loc[(q.paradigm == paradigm) & (q.status == "ok"), "child"])

    def core(self) -> list[str]:
        sets = [self.ok_children(p) for p in self.paradigms]
        have = set(self.bank.wide(self.cfg.identification.primary_family).child)
        return sorted(set.intersection(*sets) & have)

    def meta(self) -> pd.DataFrame:
        cov = self.parts.covariates()
        lab = self.parts.labels().rename(columns={"group_code": "group"})
        m = cov.join(lab, how="left")
        m["sex_male"] = (
            (m["sex"].astype(str).str.upper().str.startswith("M")).astype(float).where(m["sex"].notna())
        )
        mc = m["medication_class"]
        m["med_stimulant_or_ssri"] = mc.isin(["stimulant", "ssri"]).astype(float)
        m["med_other"] = (mc == "other").astype(float)
        m["med_unknown"] = (mc == "unknown").astype(float)
        for k in ("stimulant", "ssri", "other", "unknown"):
            m[f"medication_class={k}"] = (mc == k).astype(float)
        m = m.join(data_quality(self.qc, self.paradigms), how="left")
        m["log_clean_minutes"] = np.log(m["clean_minutes"])
        m["retained_data"] = m["clean_minutes"]
        m["qc_bad_channels"] = m["mean_bad_fraction"]
        m["qc_asr_fraction"] = m["mean_asr_fraction"]
        return m

    def child_matrix(self, children: list[str]) -> pd.DataFrame:
        """Child x (view, family, feature): block means of the shared families plus evoked / behaviour."""
        parts = []
        for fam in CHILD_FAMILIES:
            w = self.bank.wide(fam)
            w = w[w.child.isin(children)]
            feats = [c for c in w.columns if c not in ("child", "paradigm", "block")]
            m = w.groupby(["child", "paradigm"])[feats].mean()
            for p in self.paradigms:
                if p not in m.index.get_level_values(1):
                    continue
                b = m.xs(p, level=1).reindex(children)
                if fam in ("evoked", "behaviour"):
                    b = b[[c for c in b.columns if c.startswith(f"{p}__")]]
                b = b.loc[:, b.notna().any()]
                if b.shape[1] == 0:
                    continue
                b.columns = pd.MultiIndex.from_tuples([(p, fam, c) for c in b.columns])
                parts.append(b)
        return pd.concat(parts, axis=1)

    def roi_scalar(self, children: list[str]) -> dict[str, pd.DataFrame]:
        roi = list(self.cfg.variance.h1b_roi)
        out = {}
        spec = {
            "aperiodic_exponent": ("aperiodic", "exponent"),
            "aperiodic_offset": ("aperiodic", "offset"),
            "periodic_alpha": ("periodic", "alpha"),
        }
        for name, (fam, kind) in spec.items():
            if name not in self.cfg.variance.h1b_features:
                continue
            w = self.bank.wide(fam)
            w = w[w.child.isin(children)]
            cols = [f"{c}_{kind}" for c in roi if f"{c}_{kind}" in w.columns]
            d = w[["child", "paradigm", "block"]].assign(value=w[cols].mean(axis=1))
            out[name] = d.dropna().reset_index(drop=True)
        return out

    def band_logpow(self) -> pd.DataFrame:
        w = self.bank.wide("spectral_logpow")
        meta = self.bank.meta()
        meta = meta[meta.family == "spectral_logpow"]
        hz = meta["band"].str.replace("Hz", "").astype(float).to_numpy()
        out = w[["child", "paradigm", "block"]].copy()
        cols = meta["feature"].to_numpy()
        chans = meta["channel"].to_numpy()
        for band, (lo, hi) in self.cfg.features.bands.items():
            for ch in pd.unique(chans):
                sel = cols[(chans == ch) & (hz >= lo) & (hz <= hi)]
                out[f"{ch}_{band}"] = w[sel].mean(axis=1)
        return out


def make_targets(meta: pd.DataFrame, core: list[str], cfg: Config) -> list[Target]:
    m = meta.loc[core]
    at = m[m.group.isin(["ASD", "TD"])]
    sib = m[m.group == "SIB"]
    out = [
        Target(
            "asd_vs_td",
            (at.group == "ASD").astype(int),
            "classification",
            at.group,
            heldout=pd.Series(np.nan, index=sib.index),
        )
    ]
    for col in ("srs2_total_t", "mabc_total_ss", "fsiq"):
        if col not in cfg.prediction.targets:
            continue
        y = at[col].dropna()
        strata = pd.qcut(y.rank(method="first"), 5, labels=False)
        out.append(Target(col, y, "regression", strata, heldout=sib[col].dropna()))
    return out
