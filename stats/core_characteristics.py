"""Characteristics of the core set by group with autistic-versus-TD tests (Table 1).

Continuous variables: median (interquartile range) and a Mann-Whitney U test; binary variables: count (%) and
Fisher's exact test. The data-quality variables are the covariates of the model ladder computed over the six seated
paradigms (clean minutes, interpolated-channel fraction, fraction changed by artefact subspace reconstruction).
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from config.loader import load_config
from utils.inputs import Inputs

CONTINUOUS = {
    "age": "Age (years)",
    "fsiq": "FSIQ",
    "srs2_total_t": "SRS-2 total (T)",
    "mabc_total_ss": "MABC-2 total (standard score)",
    "retained_data": "Clean minutes (six paradigms)",
    "qc_bad_channels": "Interpolated-channel fraction",
    "qc_asr_fraction": "ASR-changed fraction",
}
BINARY = {"sex_male": "Male", "medicated": "Any medication"}


def main() -> None:
    cfg = load_config()
    inp = Inputs(cfg)
    m = inp.meta().loc[inp.core()].copy()
    m["medicated"] = (m["medication_class"] != "none").astype(float)
    groups = ["ASD", "TD", "SIB"]
    rows = []
    for col, label in {**CONTINUOUS, **BINARY}.items():
        row = {"variable": col, "label": label, "kind": "binary" if col in BINARY else "continuous"}
        for g in groups:
            v = m.loc[m.group == g, col].dropna()
            row[f"{g}_n"] = len(v)
            if col in BINARY:
                row[f"{g}_count"] = int(v.sum())
            else:
                row[f"{g}_median"], row[f"{g}_q1"], row[f"{g}_q3"] = (
                    v.median(),
                    v.quantile(0.25),
                    v.quantile(0.75),
                )
        x, y = (m.loc[m.group == g, col].dropna() for g in ("ASD", "TD"))
        if col in BINARY:
            row["p_asd_td"] = fisher_exact([[x.sum(), len(x) - x.sum()], [y.sum(), len(y) - y.sum()]])[1]
            row["test"] = "Fisher exact"
        else:
            row["p_asd_td"] = mannwhitneyu(x, y).pvalue
            row["test"] = "Mann-Whitney U"
        rows.append(row)
    out = cfg.path("results") / "estimation" / "core_characteristics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
