"""Candidate markers computed on the full recording and on two independent halves (Methods, Reliability and
validity of candidate markers; Additional file 1, Supplementary Methods S4).

Evoked and behavioural markers: every function of :mod:`features.evoked` is run unchanged on a recording whose
stimulus events are thinned to alternate trials within each trial type (half 0: 1st, 3rd, ...; half 1: 2nd, 4th,
...). AVSRT, whose race-model and switch-cost measures depend on trial sequences, is split in time instead (first
and second half of its trials). Responses, cues that are not trials, block markers and the cleaned data are
untouched, so each half is the same recording with half of its trials.

Spectral markers (posterior region of interest): periodic alpha power, alpha centre frequency, aperiodic exponent
and offset at rest, and the rest-minus-task change in periodic alpha, from the block-level feature bank; half 0 /
half 1 = odd / even blocks of each recording.

Output: a long table child x marker x half (half -1 = full recording) and the spectral markers per paradigm.
"""

from __future__ import annotations

import time
from functools import cached_property

import pandas as pd
from joblib import Parallel, delayed

from features.evoked import EVOKED
from features.signals import CleanRecording
from utils import common as C

ROI = list(C.CFG.variance.h1b_roi)
TASKS = [p for p in C.PARADIGMS if p != "Restingstate"]


class HalfRecording(CleanRecording):
    """A cleaned recording whose trial events are thinned to alternate trials within each trial type."""

    def __init__(self, cfg, child, paradigm, half: int):
        super().__init__(cfg, child, paradigm, "primary")
        self.half = half

    @cached_property
    def events(self) -> pd.DataFrame:
        ev = CleanRecording.events.func(self).copy()
        pat = self.cfg.dataset.trial_patterns[self.paradigm]
        is_trial = ev["trial_type"].str.match(pat)
        keep = pd.Series(True, index=ev.index)
        if self.paradigm == "AVSRT":
            # race-model gain and switch cost use trial sequences (repeat vs switch trials), which alternate
            # thinning would break: AVSRT is split in time (trials before / after the median trial onset)
            s = ev.loc[is_trial, "sample"]
            keep[s.index] = (s > s.median()) == bool(self.half)
        else:
            rank = ev[is_trial].groupby("trial_type")["sample"].rank(method="first").astype(int) - 1
            keep[rank.index] = (rank % 2) == self.half
        return ev[keep].reset_index(drop=True)


def evoked_halves(child: str, paradigm: str) -> list[dict]:
    from threadpoolctl import threadpool_limits

    fn = EVOKED[paradigm]
    rows = []
    with threadpool_limits(1):
        full = CleanRecording(C.CFG, child, paradigm, "primary")
        if not full.paths["qc"].is_file() or not full.usable:
            return rows
        for half in (0, 1):
            r = HalfRecording(C.CFG, child, paradigm, half)
            try:
                vals = fn(r, C.CFG)
            except Exception as e:  # noqa: BLE001 - a half with too few trials of a condition
                vals = {"_error": str(e)[:80]}
            for k, v in vals.items():
                if k != "_error":
                    rows.append(
                        {"child": child, "marker": f"{paradigm}__{k}", "half": half, "value": float(v)}
                    )
    return rows


def roi_blocks(family: str, suffix: str, agg: str = "mean") -> pd.DataFrame:
    w = pd.read_parquet(C.BANK / f"{family}.parquet")
    cols = [f"{ch}_{suffix}" for ch in ROI]
    v = w[cols].median(axis=1) if agg == "median" else w[cols].mean(axis=1)
    d = w[["child", "paradigm", "block"]].assign(v=v)
    d["rank"] = d.groupby(["child", "paradigm"]).block.rank(method="first").astype(int)
    d["half"] = (d["rank"] + 1) % 2  # odd blocks -> 0, even -> 1
    return d


def spectral_halves() -> pd.DataFrame:
    rows = []
    specs = {
        "rest_periodic_alpha": ("periodic", "alpha", "mean"),
        "rest_alpha_cf": ("alpha_cf", "alphacf", "median"),
        "rest_aperiodic_exponent": ("aperiodic", "exponent", "mean"),
        "rest_aperiodic_offset": ("aperiodic", "offset", "mean"),
    }
    for name, (fam, suf, agg) in specs.items():
        d = roi_blocks(fam, suf, agg)
        r = d[d.paradigm == "Restingstate"]
        for half, g in ((0, r[r.half == 0]), (1, r[r.half == 1]), (-1, r)):
            m = g.groupby("child").v.mean()
            rows += [{"child": c, "marker": name, "half": half, "value": float(x)} for c, x in m.items()]
        if name == "rest_periodic_alpha":
            for half in (0, 1, -1):
                g = d if half == -1 else d[d.half == half]
                rest = g[g.paradigm == "Restingstate"].groupby("child").v.mean()
                task = g[g.paradigm.isin(TASKS)].groupby("child").v.mean()
                mod = (rest - task).dropna()
                rows += [
                    {"child": c, "marker": "alpha_rest_minus_task", "half": half, "value": float(x)}
                    for c, x in mod.items()
                ]
    return pd.DataFrame(rows)


def cross_paradigm() -> pd.DataFrame:
    """Each spectral marker computed in every seated paradigm (all blocks): child x paradigm table."""
    rows = []
    for name, (fam, suf, agg) in {
        "periodic_alpha": ("periodic", "alpha", "mean"),
        "alpha_cf": ("alpha_cf", "alphacf", "median"),
        "aperiodic_exponent": ("aperiodic", "exponent", "mean"),
        "aperiodic_offset": ("aperiodic", "offset", "mean"),
    }.items():
        d = roi_blocks(fam, suf, agg).groupby(["child", "paradigm"]).v.mean().reset_index()
        rows.append(d.assign(marker=name))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    t0 = time.time()
    out = C.out_dir("marker_halves")
    recs = [(c, p) for c, p, _ in C.recordings() if p in EVOKED]
    # workers get the function by reference (a function of __main__ would be pickled by value)
    from markers import marker_halves as mod

    res = Parallel(n_jobs=30)(delayed(mod.evoked_halves)(c, p) for c, p in recs)
    ev = pd.DataFrame([r for rr in res for r in rr])
    full = pd.read_parquet(C.BANK / "evoked.parquet").melt(
        id_vars=["child", "paradigm", "block"], var_name="marker"
    )
    beh = pd.read_parquet(C.BANK / "behaviour.parquet").melt(
        id_vars=["child", "paradigm", "block"], var_name="marker"
    )
    full = pd.concat([full, beh]).dropna(subset=["value"])
    full["marker"] = full.marker.str.replace("__beh_", "__", regex=False)
    ev["marker"] = ev.marker.str.replace("__beh_", "__", regex=False)
    allv = pd.concat(
        [ev, full[["child", "marker", "value"]].assign(half=-1), spectral_halves()], ignore_index=True
    )
    allv.to_parquet(out / "marker_values.parquet")
    cross_paradigm().to_parquet(out / "spectral_by_paradigm.parquet")
    n = allv[allv.half >= 0].groupby("marker").child.nunique()
    print(n.to_string(), flush=True)
    C.write_json(
        out / "b1_summary.json",
        {"seconds": time.time() - t0, "n_recordings": len(recs), "n_markers": int(allv.marker.nunique())},
    )


if __name__ == "__main__":
    main()
