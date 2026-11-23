"""QC summary table: one row per preprocessed recording from the QC records (status, bad channels, removed
independent components, ASR-changed fraction, residual windows, clean seconds, artefact-free trials). It is the
source of Additional file 1, Table S1, of the retention status used by the core set and of the data-quality
covariates."""

from __future__ import annotations

import json

import pandas as pd

from config.loader import Config
from preprocessing.biosemi import derivatives_dir


def qc_summary(cfg: Config, variant: str = "primary") -> pd.DataFrame:
    """One row per recording from the QC JSONs; written to derivatives and to results/preprocessing/."""
    rows = []
    for p in sorted(derivatives_dir(cfg, variant).glob("sub-*/eeg/*_desc-qc.json")):
        q = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "child": q["child"],
                "paradigm": q["paradigm"],
                "status": q["status"],
                "duration_s": q.get("duration_s"),
                "n_joins": q.get("n_joins"),
                "valid_fraction": (q.get("segments") or {}).get("valid_fraction"),
                "n_bad_channels": q.get("n_bad_channels"),
                "bad_channels": ",".join(q.get("interpolated", [])),
                "ica_components": (q.get("ica") or {}).get("n_components"),
                "ica_removed": q.get("n_components_removed"),
                "ica_vetoed": len((q.get("ica_selection") or {}).get("vetoed", [])),
                "removed_labels": ",".join(q.get("removed_labels", [])),
                "asr_repaired_fraction": (q.get("asr") or {}).get("repaired_fraction"),
                "residual_fraction": q.get("residual_fraction_of_valid"),
                "clean_seconds": q.get("clean_seconds"),
                "n_trials": q.get("n_trials"),
                "n_trials_artefact_free": q.get("n_trials_artefact_free"),
                "n_blocks": q.get("n_blocks"),
                "line_noise_db": (q.get("line_noise_60hz") or {}).get("median_db"),
                "exg_connected": ",".join(q.get("exg_connected", [])),
                "seconds": q.get("seconds_total"),
                "preprocessing_hash": q.get("preprocessing_hash"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(derivatives_dir(cfg, variant) / "qc_summary.tsv", sep="\t", index=False)
    out = cfg.path("results") / "preprocessing"
    out.mkdir(parents=True, exist_ok=True)
    name = "qc_summary.csv"
    df.to_csv(out / (name if variant == "primary" else f"qc_summary_{variant}.csv"), index=False)
    return df


if __name__ == "__main__":
    from config.loader import load_config

    qc_summary(load_config())
