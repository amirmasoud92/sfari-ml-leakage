"""Recordings rejected by the quality-control rules, by paradigm and group (Additional file 1, Table S2; Results,
Sample): autistic-versus-TD odds ratio with Fisher's exact test per paradigm and at child level (any seated
recording rejected), and a permutation test of the chi-square statistic across the three groups (10,000
permutations). Group labels are read from ``participants.tsv``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config.loader import Config
from utils.dataset import BidsDataset, Participants


def _reason(status: str) -> str:
    if status == "ok":
        return "ok"
    if "bad channels" in status:
        return "bad_channels"
    if "trial" in status:
        return "too_few_trials"
    if "clean" in status:
        return "too_little_clean_data"
    return "other"


def missingness(cfg: Config, variant: str = "primary") -> dict[str, pd.DataFrame]:
    name = "qc_summary.csv" if variant == "primary" else f"qc_summary_{variant}.csv"
    out = cfg.path("results") / "preprocessing"
    qc = pd.read_csv(out / name)
    labels = Participants.load(BidsDataset.from_config(cfg), cfg).labels()
    qc = qc[qc.child.isin(labels.index)].copy()
    qc["group"] = qc.child.map(labels["group_code"])
    qc["reason"] = qc.status.map(_reason)
    qc["missing"] = qc.reason != "ok"
    tab = (
        qc.groupby(["paradigm", "group"])
        .agg(
            n=("missing", "size"),
            n_missing=("missing", "sum"),
            bad_channels=("reason", lambda r: int((r == "bad_channels").sum())),
            too_few_trials=("reason", lambda r: int((r == "too_few_trials").sum())),
            too_little_clean_data=("reason", lambda r: int((r == "too_little_clean_data").sum())),
        )
        .reset_index()
    )
    tab["fraction_missing"] = tab.n_missing / tab.n
    # child level: any of the child's seated recordings dropped (recordings are clustered within children)
    seated = qc[qc.paradigm != cfg.dataset.motor_paradigm]
    child = seated.groupby("child").agg(group=("group", "first"), missing=("missing", "any")).reset_index()
    child["paradigm"] = "any_seated (child level)"
    tests = []
    for par, g in [*qc.groupby("paradigm"), ("any_seated (child level)", child)]:
        # odds ratio > 1: ASD recordings are dropped more often than TD recordings
        ct = pd.crosstab(g.group, g.missing).reindex(columns=[True, False], fill_value=0)
        row = {"paradigm": par, "n": len(g), "n_missing": int(g.missing.sum())}
        if {"ASD", "TD"} <= set(ct.index):
            sub = ct.loc[["ASD", "TD"]].to_numpy()
            odds, p = stats.fisher_exact(sub)
            row.update({"asd_vs_td_odds_ratio": odds, "asd_vs_td_fisher_p": p})
        if len(ct) > 1 and ct[True].sum() > 0:
            row["all_groups_permutation_chi2_p"] = _permutation_independence(g.group, g.missing, cfg)
        tests.append(row)
    tests = pd.DataFrame(tests)
    tab.to_csv(out / f"missingness_by_group{'' if variant == 'primary' else '_' + variant}.csv", index=False)
    tests.to_csv(out / f"missingness_tests{'' if variant == 'primary' else '_' + variant}.csv", index=False)
    return {"by_group": tab, "tests": tests}


def _permutation_independence(group: pd.Series, missing: pd.Series, cfg: Config, n: int = 10000) -> float:
    """Permutation p for the chi-square statistic of group x missing (exact chi-square is unreliable with the small
    missing counts here)."""
    from stats.rng import generator

    rng, _ = generator(cfg.project.seed, "missingness", str(len(group)))
    g = group.to_numpy()
    m = missing.to_numpy()

    def chi2(labels):
        ct = pd.crosstab(labels, m).to_numpy().astype(float)
        exp = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / ct.sum()
        return float(((ct - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())

    obs = chi2(g)
    null = np.array([chi2(rng.permutation(g)) for _ in range(n)])
    return float((1 + (null >= obs - 1e-12).sum()) / (n + 1))


if __name__ == "__main__":
    from config.loader import load_config

    missingness(load_config())
