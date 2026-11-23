"""Access to the BIDS dataset (OpenNeuro ds006780 v1.0.0): participants, recordings, the restart rule and the
event-defined blocks.

Blocks (Additional file 1, Supplementary Methods S1): ASSR and Beep-Flash blocks start at the block markers, AVSRT
blocks are consecutive sets of 100 stimuli, FAST blocks split at every face/object change and then into 60-trial
chunks, IC blocks are delimited by pause markers, and each resting block runs from a recording-start marker to the
next marker or the end of the run. A first run with fewer than 20% of the expected trials is dropped as a restart.

Participants: ``participants.tsv`` with its ``*_dup`` columns merged, medication coded by
``config/medication_coding.yaml`` and the exclusion rules of the Methods (subject folders without a participants
row and rows without a group label are not usable). Nothing here reads EEG samples.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from config.loader import PROJECT_ROOT, BlockRule, Config
from utils.logging_utils import get_logger

log = get_logger(__name__)

_RUN_RE = re.compile(r"^(?P<sub>sub-[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>\d+)_eeg\.(?P<ext>bdf|vhdr)$")


@dataclass(frozen=True)
class RunFile:
    """One BIDS run: the EEG file and its sidecars."""

    subject: str
    paradigm: str
    run: int
    eeg: Path

    @property
    def stem(self) -> str:
        return self.eeg.name.rsplit("_eeg.", 1)[0]

    @property
    def events(self) -> Path:
        return self.eeg.with_name(f"{self.stem}_events.tsv")

    @property
    def channels(self) -> Path:
        return self.eeg.with_name(f"{self.stem}_channels.tsv")

    @property
    def sidecar(self) -> Path:
        return self.eeg.with_name(f"{self.stem}_eeg.json")

    @property
    def fmt(self) -> str:
        return "bdf" if self.eeg.suffix == ".bdf" else "brainvision"

    @property
    def has_events(self) -> bool:
        return self.events.is_file()

    def read_events(self) -> pd.DataFrame:
        """``events.tsv`` (an empty table with the BIDS columns if the file is missing)."""
        if not self.has_events:
            return pd.DataFrame(
                {
                    "onset": pd.Series(dtype=float),
                    "duration": pd.Series(dtype=float),
                    "trial_type": pd.Series(dtype=str),
                    "value": pd.Series(dtype=str),
                    "sample": pd.Series(dtype=float),
                }
            )
        ev = pd.read_csv(self.events, sep="\t", dtype={"trial_type": str}, keep_default_na=False)
        ev["onset"] = pd.to_numeric(ev["onset"], errors="coerce")
        ev["sample"] = pd.to_numeric(ev.get("sample"), errors="coerce")
        return ev

    def read_sidecar(self) -> dict:
        return json.loads(self.sidecar.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Recording:
    """All kept runs of one child x paradigm (restarts merged, see :func:`apply_restart_rule`)."""

    subject: str
    paradigm: str
    runs: tuple[RunFile, ...]
    dropped_runs: tuple[RunFile, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.subject}_task-{self.paradigm}"


class BidsDataset:
    """Index of the BIDS tree (subjects, runs) built from file names only."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not (self.root / "dataset_description.json").is_file():
            raise FileNotFoundError(f"{self.root} is not a BIDS root")

    @classmethod
    def from_config(cls, cfg: Config) -> BidsDataset:
        return cls(cfg.path("data_root"))

    @cached_property
    def description(self) -> dict:
        return json.loads((self.root / "dataset_description.json").read_text(encoding="utf-8"))

    @cached_property
    def subject_dirs(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith("sub-"))

    @cached_property
    def run_files(self) -> list[RunFile]:
        out = []
        for sub in self.subject_dirs:
            eeg_dir = self.root / sub / "eeg"
            if not eeg_dir.is_dir():
                continue
            for f in sorted(eeg_dir.iterdir()):
                m = _RUN_RE.match(f.name)
                if m:
                    out.append(RunFile(m["sub"], m["task"], int(m["run"]), f))
        return out

    def runs(self, subject: str, paradigm: str) -> list[RunFile]:
        return sorted(
            (r for r in self.run_files if r.subject == subject and r.paradigm == paradigm),
            key=lambda r: r.run,
        )

    def paradigms_of(self, subject: str) -> list[str]:
        return sorted({r.paradigm for r in self.run_files if r.subject == subject})


def count_trials(ev: pd.DataFrame, pattern: str) -> int:
    return int(ev["trial_type"].str.match(pattern).sum())


def apply_restart_rule(
    runs: list[RunFile], paradigm: str, cfg: Config
) -> tuple[list[RunFile], list[RunFile]]:
    """Runs of one task are one recording, except that the first run is dropped if it holds fewer than
    ``restart_min_fraction`` of the expected trials (a restart). Returns ``(kept, dropped)``."""
    if len(runs) < 2 or paradigm not in cfg.dataset.expected_trials:
        return list(runs), []
    expected = cfg.dataset.expected_trials[paradigm]
    n_first = count_trials(runs[0].read_events(), cfg.dataset.trial_patterns[paradigm])
    if n_first < cfg.dataset.restart_min_fraction * expected:
        return list(runs[1:]), [runs[0]]
    return list(runs), []


def build_recordings(
    ds: BidsDataset, cfg: Config, subjects: list[str] | None = None, paradigms: list[str] | None = None
) -> list[Recording]:
    subs = subjects if subjects is not None else ds.subject_dirs
    pars = paradigms if paradigms is not None else list(cfg.dataset.seated_paradigms)
    out = []
    for s in subs:
        for p in pars:
            runs = ds.runs(s, p)
            if not runs:
                continue
            kept, dropped = apply_restart_rule(runs, p, cfg)
            out.append(Recording(s, p, tuple(kept), tuple(dropped)))
    return out


def load_medication_coding(path: Path | None = None) -> dict[str, dict]:
    p = path or PROJECT_ROOT / "config" / "medication_coding.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))["classes"]


def _missing(v: str) -> bool:
    return v.strip() in ("", "n/a", "NA", "NaN", "nan")


@dataclass
class Participants:
    """``participants.tsv`` with ``_dup`` columns merged, medication coded and exclusions applied.

    Attributes
    ----------
    table
        One row per ``participants.tsv`` row, all columns as merged strings plus coded columns.
    merge_log
        Column -> number of values filled from the ``_dup`` twin.
    """

    table: pd.DataFrame
    merge_log: dict[str, int] = field(default_factory=dict)
    cfg: Config | None = None

    @classmethod
    def load(cls, ds: BidsDataset, cfg: Config) -> Participants:
        raw = pd.read_csv(ds.root / "participants.tsv", sep="\t", dtype=str, keep_default_na=False)
        df = raw.copy()
        merge_log = {}
        for col in cfg.dataset.phenotype_columns:
            dup = f"{col}_dup"
            if dup not in df.columns:
                continue
            fill = df[col].map(_missing) & ~df[dup].map(_missing)
            merge_log[col] = int(fill.sum())
            df.loc[fill, col] = df.loc[fill, dup]
        df = df.drop(columns=[c for c in df.columns if c.endswith("_dup")])
        coding = load_medication_coding()
        med = df["medication"].str.strip()
        unmapped = sorted(set(med) - set(coding))
        if unmapped:
            raise ValueError(f"medication strings without a coding entry: {unmapped}")
        df["medication_raw"] = df["medication"]
        df["medication_class"] = med.map(lambda s: coding[s]["class"])
        df["med_stimulant"] = med.map(lambda s: bool(coding[s]["stimulant"]))
        df["med_ssri"] = med.map(lambda s: bool(coding[s]["ssri"]))
        for col in ("age", "fsiq", "srs2_total_t", "mabc_total_ss", "ados_css"):
            df[col] = pd.to_numeric(df[col].where(~df[col].map(_missing)), errors="coerce")
        df["group_code"] = df["group"].map(cfg.dataset.group_map)  # NaN for 'NAN' / unknown labels
        df["in_participants_tsv"] = True
        folders = set(ds.subject_dirs)
        no_row = sorted(folders - set(df["participant_id"]))
        extra = pd.DataFrame({"participant_id": no_row, "in_participants_tsv": False})
        df = pd.concat([df, extra], ignore_index=True)
        df["has_folder"] = df["participant_id"].isin(folders)
        df["usable"] = df["has_folder"] & df["in_participants_tsv"] & df["group_code"].notna()
        df["exclusion_reason"] = np.select(
            [~df["in_participants_tsv"], df["group_code"].isna(), ~df["has_folder"]],
            ["no participants.tsv row", "group label missing", "no subject folder"],
            default="",
        )
        return cls(df.sort_values("participant_id").reset_index(drop=True), merge_log, cfg)

    @property
    def usable_ids(self) -> list[str]:
        return self.table.loc[self.table["usable"], "participant_id"].tolist()

    @property
    def excluded(self) -> pd.DataFrame:
        return self.table.loc[~self.table["usable"], ["participant_id", "exclusion_reason"]]

    def covariates(self) -> pd.DataFrame:
        """Covariate columns (age, sex, handedness, medication), usable children only."""
        assert self.cfg is not None
        outcomes = set(self.cfg.dataset.outcome_columns) | {"group_code"}
        keep = [
            c
            for c in (
                "participant_id",
                "age",
                "sex",
                "handedness",
                "medication_raw",
                "medication_class",
                "med_stimulant",
                "med_ssri",
            )
            if c not in outcomes
        ]
        return self.table.loc[self.table["usable"], keep].set_index("participant_id")

    def labels(self) -> pd.DataFrame:
        """Group (coded ASD / TD / SIB) and outcome columns of ``participants.tsv``, usable children only."""
        assert self.cfg is not None
        cols = [
            "participant_id",
            "group_code",
            *[c for c in self.cfg.dataset.outcome_columns if c != "group"],
        ]
        return self.table.loc[self.table["usable"], cols].set_index("participant_id")


def _trial_rows(ev: pd.DataFrame, pattern: str) -> pd.DataFrame:
    return ev.loc[ev["trial_type"].str.match(pattern)].sort_values("onset")


def _merge_small(labels: np.ndarray, min_trials: int) -> np.ndarray:
    """Merge blocks with fewer than ``min_trials`` trials into the previous block (the first into the next)."""
    labels = labels.copy()
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        small = ids[counts < min_trials]
        if len(small) == 0 or len(ids) == 1:
            break
        b = small[0]
        pos = int(np.flatnonzero(ids == b)[0])
        target = ids[pos - 1] if pos > 0 else ids[pos + 1]
        labels[labels == b] = target
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def assign_blocks(ev: pd.DataFrame, rule: BlockRule, trial_pattern: str, min_trials: int) -> pd.DataFrame:
    """Trial table of one run with a ``block`` column (0-based within the run) following ``rule``."""
    trials = _trial_rows(ev, trial_pattern).copy()
    if trials.empty:
        trials["block"] = pd.Series(dtype=int)
        return trials
    if rule.rule == "trial_count":
        lab = np.arange(len(trials)) // int(rule.size or 1)
    elif rule.rule == "category_chunks":
        cat = trials["trial_type"].str.extract(f"({rule.category_regex})")[0].fillna("").to_numpy()
        run_id = np.concatenate([[0], np.cumsum(cat[1:] != cat[:-1])])
        lab = np.zeros(len(trials), int)
        nxt = 0
        size = int(rule.size or 1)
        for r in np.unique(run_id):
            idx = np.flatnonzero(run_id == r)
            chunk = np.arange(len(idx)) // size
            n_chunks = int(chunk.max()) + 1
            if n_chunks > 1 and (chunk == n_chunks - 1).sum() < min_trials:
                chunk[chunk == n_chunks - 1] = (
                    n_chunks - 2
                )  # short tail joins the previous chunk of the same run
            lab[idx] = nxt + chunk
            nxt += int(chunk.max()) + 1
        trials["block"] = np.unique(lab, return_inverse=True)[1]
        return trials
    elif rule.rule in ("marker", "separator"):
        marks = np.sort(ev.loc[ev["trial_type"].str.match(rule.marker or ""), "onset"].to_numpy(float))
        lab = np.searchsorted(marks, trials["onset"].to_numpy(float), side="right")
    else:
        raise ValueError(f"rule {rule.rule!r} does not apply to trial-based paradigms")
    trials["block"] = _merge_small(np.asarray(lab), min_trials)
    return trials


def block_table(
    rec: Recording, cfg: Config, rule: BlockRule | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blocks of one recording and its trial table.

    Returns
    -------
    blocks
        One row per block: ``child, paradigm, run, block, start_s, stop_s, duration_s, n_trials, n_responses``
        (+ ``category`` for FAST, ``frequency`` for ASSR). Times are in seconds of that run's file. ``block`` is
        numbered across the kept runs.
    trials
        One row per trial with ``child, paradigm, run, block, onset, sample, trial_type``.
    """
    p = rec.paradigm
    rule = rule or cfg.blocks.rule_for(p)
    rows, trial_tabs, offset = [], [], 0
    for rf in rec.runs:
        ev = rf.read_events()
        if rule.rule == "rest_marker":
            dur = float(rf.read_sidecar().get("RecordingDuration", np.nan))
            starts = np.sort(ev.loc[ev["trial_type"].str.match(rule.marker or ""), "onset"].to_numpy(float))
            if starts.size == 0:
                continue
            stops = np.append(starts[1:], dur)
            for a, b in zip(starts, stops, strict=True):
                if b - a >= cfg.blocks.min_rest_block_s:
                    rows.append(
                        {
                            "run": rf.run,
                            "block": offset,
                            "start_s": float(a),
                            "stop_s": float(b),
                            "n_trials": 0,
                            "n_responses": 0,
                        }
                    )
                    offset += 1
            continue
        trials = assign_blocks(ev, rule, cfg.dataset.trial_patterns[p], cfg.blocks.min_trials_per_block)
        if trials.empty:
            continue
        resp = ev.loc[ev["trial_type"].str.match(cfg.dataset.response_pattern), "onset"].to_numpy(float)
        for b, g in trials.groupby("block", sort=True):
            t0 = float(g["onset"].min()) - cfg.blocks.pad_s
            t1 = float(g["onset"].max()) + cfg.preprocessing.epochs.tmax + cfg.blocks.pad_s
            row = {
                "run": rf.run,
                "block": offset + int(b),
                "start_s": max(t0, 0.0),
                "stop_s": t1,
                "n_trials": len(g),
                "n_responses": int(((resp >= t0) & (resp <= t1)).sum()),
            }
            if p == "FAST":
                cats = g["trial_type"].str.split("_").str[0].value_counts()
                row["category"] = cats.index[0] if len(cats) == 1 else "mixed:" + "+".join(sorted(cats.index))
            if p == "ASSR":
                fr = g["trial_type"].str.extract(r"^(\d+)_Hz")[0].value_counts()
                row["frequency"] = fr.index[0] if len(fr) == 1 else "mixed"
            rows.append(row)
        n_local = int(trials["block"].max()) + 1
        trials = trials.assign(run=rf.run, block=trials["block"] + offset)
        trial_tabs.append(trials[["run", "block", "onset", "sample", "trial_type"]])
        offset += n_local
    blocks = pd.DataFrame(rows)
    if not blocks.empty:
        blocks.insert(0, "paradigm", p)
        blocks.insert(0, "child", rec.subject)
        blocks["duration_s"] = blocks["stop_s"] - blocks["start_s"]
    trials = (
        pd.concat(trial_tabs, ignore_index=True)
        if trial_tabs
        else pd.DataFrame(columns=["run", "block", "onset", "sample", "trial_type"])
    )
    trials.insert(0, "paradigm", p)
    trials.insert(0, "child", rec.subject)
    return blocks, trials
