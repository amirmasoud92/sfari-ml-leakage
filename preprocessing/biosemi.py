"""``BiosemiPreprocessor``: one seated recording (all kept runs of one child and paradigm) to a cleaned
256-Hz recording (Methods, Preprocessing; Additional file 1, Supplementary Methods S1 and Table S1).

Steps: the kept runs are loaded, cropped to the task events and concatenated with their joins; rail, flat and
pause-join spans split the data into filter segments; a 1-100 Hz copy (notched at 60 and 120 Hz) is used for
PyPREP bad-channel detection in a robust-reference loop (additionally low-passed at 30 Hz for the detection, blink
periods left out) and for Picard ICA with ICLabel and a posterior-alpha veto; the analysis copy is band-passed at
0.5-45 Hz (zero-phase FIR, per valid span) and decimated from 512 to 256 Hz; the ICA is applied, bad channels are
interpolated and the data re-referenced to the robust average; artefact subspace reconstruction (cutoff 20,
calibrated on the cleanest 60 s) repairs transients; 1-s windows above 200 uV after repair are marked bad; and the
retention rule (60 s of clean rest or 150 artefact-free trials) and the 15% bad-channel limit are applied. The
eight external channels are typed as miscellaneous and only filtered and decimated.

Outputs per recording in ``<derivatives>/preprocessing/<variant>/sub-*/eeg/``: the cleaned recording, events,
blocks, the ICA and a QC record (``*_desc-qc.json``).
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.loader import Config
from preprocessing import adapter
from preprocessing import steps as st
from stats.rng import derive_seed
from utils.base import Preprocessor
from utils.dataset import Recording, block_table
from utils.logging_utils import get_logger

log = get_logger(__name__)

BIOSEMI_EEG = None  # filled lazily from the MNE biosemi64 montage (channel order of the BDF files)


def preprocessing_hash(cfg: Config, variant: str) -> str:
    """Hash of everything that changes a preprocessing output (dataset, blocks and preprocessing sections)."""
    payload = {
        "dataset": cfg.dataset.model_dump(mode="json"),
        "blocks": cfg.blocks.model_dump(mode="json"),
        "preprocessing": cfg.preprocessing.model_dump(mode="json"),
        "seed": cfg.project.seed,
        "variant": variant,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def upstream_hash(cfg: Config) -> str:
    """Hash of what changes the bad channels and the ICA (not ASR, residual, epochs, retention or the variant):
    the cached ICA of the primary variant is reused by every variant with the same upstream hash."""
    p = cfg.preprocessing.model_dump(mode="json")
    for k in ("asr", "residual", "epochs", "minimum_retained", "diagnostics_subjects"):
        p.pop(k, None)
    for k in (
        "iclabel_rule",
        "iclabel_exclude",
        "alpha_veto",
        "eye_validation",
    ):  # selection is recomputed from the stored ICLabel probabilities, so it does not need a refit
        p["ica"].pop(k, None)
    payload = {
        "dataset": cfg.dataset.model_dump(mode="json"),
        "blocks": cfg.blocks.model_dump(mode="json"),
        "preprocessing": p,
        "seed": cfg.project.seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def derivatives_dir(cfg: Config, variant: str = "primary") -> Path:
    return cfg.path("derivatives") / "preprocessing" / variant


def output_paths(cfg: Config, child: str, paradigm: str, variant: str = "primary") -> dict[str, Path]:
    d = derivatives_dir(cfg, variant) / child / "eeg"
    stem = f"{child}_task-{paradigm}"
    return {
        "dir": d,
        "raw": d / f"{stem}_desc-clean_eeg.fif",
        "events": d / f"{stem}_desc-clean_events.tsv",
        "blocks": d / f"{stem}_desc-clean_blocks.tsv",
        "ica": d / f"{stem}_desc-ica_ica.fif",
        "ica_meta": d / f"{stem}_desc-icameta.json",
        "qc": d / f"{stem}_desc-qc.json",
    }


@dataclass
class Assembled:
    """Kept runs of one recording, cropped and concatenated (512 Hz, volts)."""

    data: np.ndarray
    sfreq: float
    eeg: list[str]
    exg: list[str]
    joins: np.ndarray
    events: pd.DataFrame
    blocks: pd.DataFrame
    runs: list[dict] = field(default_factory=list)


@dataclass
class PreprocResult:
    child: str
    paradigm: str
    status: str
    qc: dict
    paths: dict[str, Path]


class BiosemiPreprocessor(Preprocessor):
    """Preprocess one seated BioSemi recording (see the module docstring)."""

    def __init__(
        self, cfg: Config, variant: str = "primary", asr: bool | None = None, diagnostics: bool = False
    ):
        super().__init__(cfg)
        self.variant = variant
        self.use_asr = cfg.preprocessing.asr.enabled if asr is None else asr
        self.diagnostics = diagnostics
        self.hash = preprocessing_hash(cfg, variant)
        self.lc = adapter.step_configs(cfg)

    def is_done(self, rec: Recording) -> bool:
        p = output_paths(self.cfg, rec.subject, rec.paradigm, self.variant)["qc"]
        if not p.is_file():
            return False
        try:
            q = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return False
        return q.get("preprocessing_hash") == self.hash and q.get("status", "").split(":")[0] in (
            "ok",
            "missing",
        )

    def run(self, recording: Recording) -> PreprocResult:
        paths = output_paths(self.cfg, recording.subject, recording.paradigm, self.variant)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        qc: dict[str, Any] = {
            "child": recording.subject,
            "paradigm": recording.paradigm,
            "variant": self.variant,
            "preprocessing_hash": self.hash,
            "runs_kept": [r.run for r in recording.runs],
            "runs_dropped_restart": [r.run for r in recording.dropped_runs],
            "asr_enabled": self.use_asr,
        }
        try:
            status = self._process(recording, paths, qc)
        except Exception as e:  # noqa: BLE001 - recorded as a missing recording with its reason, never silent
            log.exception("%s: preprocessing failed", recording.key)
            status = f"error:{type(e).__name__}: {e}"
            qc["traceback"] = traceback.format_exc()
        qc["status"] = status
        qc["seconds_total"] = round(time.perf_counter() - t0, 1)
        paths["qc"].write_text(json.dumps(qc, indent=1, default=_json), encoding="utf-8")
        log.info("%s: %s (%.0f s)", recording.key, status, qc["seconds_total"])
        return PreprocResult(recording.subject, recording.paradigm, status, qc, paths)

    def _eye_validator(self, hp, raw, A: Assembled, seg, good, ica, blinks, probs, qc):
        """Decides argmax-eye components with p_eye < strict_p by their eye-locking (blink ratio, HEOG)."""
        import mne

        ev = self.cfg.preprocessing.ica.eye_validation
        step = 4
        sf = A.sfreq / step
        X = hp._data[[A.eeg.index(c) for c in ica.ch_names]][:, ::step]
        X = X - X.mean(axis=0, keepdims=True)  # average reference over the ICA channels, as in the fit
        S = adapter.component_sources(ica, X)
        valid = seg.valid_mask()[::step]
        bm = st.spans_to_mask(blinks, seg.n_times)[::step] & valid
        heog = None
        if all(c in qc.get("exg_connected", []) for c in ("EXG1", "EXG2")):
            i1, i2 = len(A.eeg) + A.exg.index("EXG1"), len(A.eeg) + A.exg.index("EXG2")
            h = raw._data[i1] - raw._data[i2]
            heog = mne.filter.filter_data(h - np.median(h), A.sfreq, 1.0, 15.0, verbose="error")[::step]
        eye = adapter.iclabel_index("eye blink")
        top = np.argmax(probs, axis=1)
        info = {}
        for i in np.flatnonzero(top == eye):
            s = S[i]
            ratio = float(np.abs(s[bm]).mean() / np.abs(s[valid & ~bm]).mean()) if bm.sum() > sf else np.nan
            r = float(np.corrcoef(s[valid], heog[valid])[0, 1]) if heog is not None else np.nan
            keep = bool(
                probs[i, eye] >= ev.strict_p
                or (np.isfinite(ratio) and ratio >= ev.blink_ratio)
                or (np.isfinite(r) and abs(r) >= ev.heog_r)
            )
            info[int(i)] = {
                "p_eye": round(float(probs[i, eye]), 3),
                "blink_ratio": round(ratio, 2),
                "heog_r": round(r, 3),
                "removed_as_eye": keep,
            }
        return (lambda i: info.get(int(i), {}).get("removed_as_eye", True)), info

    def detection_segments(self, hp, A: Assembled, seg, blinks: np.ndarray, qc: dict):
        """Segment info for PyPREP only: blink periods and whole-head artefact windows left out,
        together capped at ``exclude_blinks.max_fraction`` of the valid data (blinks first, then global windows
        by score); plus paradigm-specific exclusions (motor walking runs)."""
        bcfg = self.cfg.preprocessing.bad_channels
        eb, eg = bcfg.exclude_blinks, bcfg.exclude_global_artefacts
        valid = seg.valid_mask()
        nv = max(int(valid.sum()), 1)
        excl = np.zeros(seg.n_times, bool)
        info: dict = {"n_blink_spans": int(len(blinks))}
        if eb.enabled and len(blinks):
            m = st.spans_to_mask(blinks, seg.n_times) & valid
            info["blink_fraction"] = float(m.sum() / nv)
            if m.sum() / nv <= eb.max_fraction:
                excl |= m
        n_ex = int(excl.sum())
        if eg.enabled:
            X = hp._data
            med = np.empty(X.shape[1])
            for a in range(0, X.shape[1], 1 << 16):
                med[a : a + (1 << 16)] = np.median(X[:, a : a + (1 << 16)], axis=0)
            gw, _ = st.global_artefact_windows(X - med, valid, A.sfreq, eg.window_s, eg.z)
            pad = int(round(eg.pad_s * A.sfreq))
            added = 0
            for a, b in gw:  # highest score first, until the cap
                lo, hi = max(a - pad, 0), min(b + pad, seg.n_times)
                new = int((valid[lo:hi] & ~excl[lo:hi]).sum())
                if (n_ex + new) / nv > eb.max_fraction:
                    break
                excl[lo:hi] |= valid[lo:hi]
                n_ex += new
                added += 1
            info.update({"global_windows": int(len(gw)), "global_windows_excluded": added})
        info["excluded_fraction"] = float(n_ex / nv)
        qc["detection_exclusion"] = info
        seg_det = seg
        if excl.any():
            seg_det = adapter.with_extra_bad(
                seg, st.mask_to_spans(excl), "excluded from bad-channel detection"
            )
        extra = self.detection_exclusions(A)
        if len(extra):
            seg_det = adapter.with_extra_bad(seg_det, extra, "excluded from bad-channel detection (paradigm)")
            info["paradigm_exclusion_s"] = float((extra[:, 1] - extra[:, 0]).sum() / A.sfreq)
        return seg_det

    def detection_exclusions(self, A: Assembled) -> np.ndarray:
        """Extra sample spans left out of bad-channel detection only (none for seated paradigms)."""
        del A
        return np.zeros((0, 2), np.int64)

    def _cached_ica(self, rec: Recording):
        """(ICA or None, meta) from the primary variant if its upstream hash matches; else None."""
        import mne

        prim = output_paths(self.cfg, rec.subject, rec.paradigm, "primary")
        if not prim["ica_meta"].is_file():
            return None
        meta = json.loads(prim["ica_meta"].read_text(encoding="utf-8"))
        if meta.get("upstream_hash") != upstream_hash(self.cfg):
            return None
        if meta["labels"] and "iclabel_probabilities" not in (meta.get("ica") or {}):
            return None
        ica = None
        if meta["labels"]:
            if not prim["ica"].is_file():
                return None
            ica = mne.preprocessing.read_ica(prim["ica"], verbose="error")
        meta["path"] = str(prim["ica_meta"])
        return ica, meta

    def _store_ica_meta(self, paths, bads, labels, sel, qc) -> None:
        if self.variant != "primary":
            return
        meta = {
            "upstream_hash": upstream_hash(self.cfg),
            "bads": bads,
            "labels": labels,
            "selection": sel,
            "bad_channels": qc.get("bad_channels"),
            "ica": qc.get("ica"),
            "detection_exclusion": qc.get("detection_exclusion"),
        }
        paths["ica_meta"].write_text(json.dumps(meta, default=_json), encoding="utf-8")

    def assemble(self, rec: Recording) -> Assembled:
        """Load, type, crop and concatenate the kept runs; events and blocks in recording samples (512 Hz)."""
        import mne

        global BIOSEMI_EEG
        if BIOSEMI_EEG is None:
            BIOSEMI_EEG = list(mne.channels.make_standard_montage("biosemi64").ch_names)
        p = self.cfg.preprocessing
        blocks, _ = block_table(rec, self.cfg)
        datas, joins, evs, blks, runs = [], [], [], [], []
        offset = 0
        exg_names: list[str] | None = None
        sf = None
        for rf in rec.runs:
            raw = mne.io.read_raw_bdf(rf.eeg, preload=False, verbose="error")
            sf = float(raw.info["sfreq"])
            missing = [c for c in BIOSEMI_EEG if c not in raw.ch_names]
            if missing:
                raise ValueError(f"{rf.eeg.name}: EEG channels missing {missing[:5]}")
            exg = [c for c in p.exg_channels if c in raw.ch_names]
            exg_names = exg if exg_names is None else [c for c in exg_names if c in exg]
            ev = rf.read_events()
            is_rest = rec.paradigm == "Restingstate"
            a, b = (
                (0, raw.n_times) if is_rest else st.crop_bounds(ev["onset"], raw.n_times, sf, p.crop_margin_s)
            )
            x = raw.get_data(picks=BIOSEMI_EEG + exg, start=a, stop=b)
            j = adapter.status_joins(rf.eeg, p.segments.join_bit)
            j = j[(j > a) & (j < b)] - a + offset
            if offset > 0:
                joins.append(np.array([offset]))
            joins.append(j)
            e = ev.copy()
            e["sample_rec"] = np.rint(e["onset"].to_numpy(float) * sf).astype(np.int64) - a + offset
            e["run"] = rf.run
            evs.append(e[(e["sample_rec"] >= offset) & (e["sample_rec"] < offset + (b - a))])
            bb = blocks[blocks["run"] == rf.run].copy()
            if not bb.empty:
                bb["start_sample"] = (
                    np.clip(np.rint(bb["start_s"] * sf).astype(np.int64) - a, 0, b - a) + offset
                )
                bb["stop_sample"] = (
                    np.clip(np.rint(bb["stop_s"] * sf).astype(np.int64) - a, 0, b - a) + offset
                )
                blks.append(bb)
            runs.append(
                {"run": rf.run, "file_start_sample": int(a), "file_stop_sample": int(b), "offset": offset}
            )
            datas.append((x, exg))
            offset += b - a
        exg_names = exg_names or []
        rows = []
        for x, exg in datas:
            keep = list(range(len(BIOSEMI_EEG))) + [len(BIOSEMI_EEG) + exg.index(c) for c in exg_names]
            rows.append(x[keep])
        data = np.concatenate(rows, axis=1)
        jj = np.unique(np.concatenate(joins)) if joins else np.zeros(0, np.int64)
        jj = jj[(jj > 0) & (jj < data.shape[1])]
        events = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame()
        blk = pd.concat(blks, ignore_index=True) if blks else pd.DataFrame()
        return Assembled(data, float(sf), list(BIOSEMI_EEG), list(exg_names), jj, events, blk, runs)

    def _process(self, rec: Recording, paths: dict[str, Path], qc: dict) -> str:
        import mne

        cfg, p, lc = self.cfg, self.cfg.preprocessing, self.lc
        seed = derive_seed(cfg.project.seed, rec.subject, rec.paradigm, "preprocessing")
        qc["seed"] = seed
        A = self.assemble(rec)
        qc["n_samples_512"] = int(A.data.shape[1])
        qc["duration_s"] = A.data.shape[1] / A.sfreq
        qc["n_joins"] = int(A.joins.size)
        qc["runs"] = A.runs
        if A.blocks.empty:
            return "missing:no blocks (no events or no trials)"
        n_eeg = len(A.eeg)
        types = ["eeg"] * n_eeg + ["misc"] * len(A.exg)
        info = mne.create_info(A.eeg + A.exg, A.sfreq, types)
        raw = mne.io.RawArray(A.data, info, verbose="error")
        raw.set_montage(p.montage, on_missing="ignore", verbose="error")
        del A.data

        # segments (rail / flat / short) on the raw EEG
        kernels = [adapter.fir_taps(lc.hp_filter, A.sfreq), adapter.fir_taps(lc.analysis_filter, A.sfreq)]
        seg = adapter.segment_info(raw._data[:n_eeg], A.sfreq, A.joins, lc, kernels)
        qc["segments"] = seg.summary()
        if seg.fraction("valid") * qc["duration_s"] < 30:
            return "missing:less than 30 s of valid (non-rail, non-flat) data"
        adapter.annotate(raw, seg)

        # EXG connection check on the raw signal
        if A.exg:
            ex = raw._data[n_eeg:]
            vm = seg.valid_mask()
            exf = mne.filter.filter_data(
                ex[:, vm][:, : int(120 * A.sfreq)], A.sfreq, 1.0, 40.0, verbose="error"
            )
            conn = st.exg_connected(ex[:, vm], exf, p.exg_unconnected.rail_v, p.exg_unconnected.max_sd_uv)
            qc["exg_connected"] = [c for c, ok in zip(A.exg, conn, strict=True) if ok]
        else:
            qc["exg_connected"] = []
        qc["exg_present"] = A.exg

        # 1-100 Hz copy: line-noise QC, notch, bad channels, ICA
        t = time.perf_counter()
        hp = raw.copy().pick(A.eeg)
        adapter.filter_per_span(hp, seg, lc.hp_filter, A.eeg)
        hv = adapter.valid_concat(hp._data, seg)
        qc["line_noise_60hz"] = adapter.line_noise_db(hv[:, : int(240 * A.sfreq)], A.sfreq, p.line_freq)
        del hv
        adapter.notch_per_span(hp, seg, lc.notch_freqs, A.eeg)
        adapter.zero_bad(hp, seg)
        eb = p.bad_channels.exclude_blinks
        bl = np.zeros((0, 2), np.int64)
        if all(c in A.eeg for c in eb.channels):
            prox = hp._data[[A.eeg.index(c) for c in eb.channels]].mean(axis=0)
            bl = st.blink_spans(prox, seg.valid_mask(), A.sfreq, eb.z, eb.pad_s)
        cached = self._cached_ica(rec)
        if cached is not None:
            ica, meta = cached
            bads, labels = meta["bads"], meta["labels"]
            qc.update({k: meta[k] for k in ("bad_channels", "ica", "detection_exclusion") if k in meta})
            qc["ica_reused_from"] = meta["path"]
            sel = {"excluded": [], "vetoed": [], "iclabel_excluded": []}
        else:
            seg_det = self.detection_segments(hp, A, seg, bl, qc)
            bc = adapter.bad_channels(
                hp, raw._data[:n_eeg], seg_det, A.eeg, lc, seed, p.bad_channels.detection_lowpass_hz
            )
            bads = list(bc.bads)
            qc["bad_channels"] = bc.qc()
            qc["seconds_bad_channels"] = round(time.perf_counter() - t, 1)
            ica, labels, sel = None, [], {"excluded": [], "vetoed": [], "iclabel_excluded": []}
            if len(bads) <= p.bad_channels.max_bad_fraction * n_eeg:
                t = time.perf_counter()
                ica_res, sel_full = adapter.run_ica(
                    hp, seg, [c for c in A.eeg if c not in bads], lc, seed, p.montage
                )
                ica, labels = ica_res.ica, ica_res.labels
                sel = {k: sel_full[k] for k in ("iclabel_excluded", "vetoed", "excluded")}
                qc["ica"] = {k: v for k, v in ica_res.qc.items() if k != "component_explained_variance"}
                qc["ica"]["labels"] = labels
                qc["ica"]["iclabel_probabilities"] = np.round(ica_res.probabilities, 4).tolist()
                qc["ica"]["explained_variance"] = np.round(
                    ica_res.qc["component_explained_variance"], 5
                ).tolist()
                qc["seconds_ica"] = round(time.perf_counter() - t, 1)
                ica.save(paths["ica"], overwrite=True, verbose="error")
            self._store_ica_meta(paths, bads, labels, sel, qc)
        qc["n_bad_channels"] = len(bads)
        if len(bads) > p.bad_channels.max_bad_fraction * n_eeg:
            return f"missing:{len(bads)} bad channels (> {p.bad_channels.max_bad_fraction:.0%})"
        if ica is not None:  # one selection path for fresh and cached ICA: ICLabel rule, eye validation, veto
            good = [c for c in A.eeg if c not in bads]
            probs = np.asarray(qc["ica"]["iclabel_probabilities"])
            keep_eye, qc["eye_validation"] = self._eye_validator(hp, raw, A, seg, good, ica, bl, probs, qc)
            sel = adapter.reselect(hp, seg, good, ica, probs, lc, p.montage, keep_eye)
        qc["ica_selection"] = sel
        qc["n_components_removed"] = len(sel["excluded"])
        qc["removed_labels"] = [labels[i] for i in sel["excluded"]]
        del hp

        # 0.5-45 Hz analysis copy, decimated to 256 Hz
        adapter.filter_per_span(raw, seg, lc.analysis_filter, A.eeg + A.exg)
        adapter.zero_bad(raw, seg)
        dec = int(round(A.sfreq / p.resample_sfreq))
        if abs(A.sfreq / dec - p.resample_sfreq) > 1e-6:
            raise ValueError(f"cannot decimate {A.sfreq} Hz to {p.resample_sfreq} Hz by an integer factor")
        sf = A.sfreq / dec
        x = raw._data[:, ::dec].copy()
        del raw
        info = mne.create_info(A.eeg + A.exg, sf, types)
        an = mne.io.RawArray(x, info, verbose="error")
        an.set_montage(p.montage, on_missing="ignore", verbose="error")
        with an.info._unlock():
            an.info["highpass"], an.info["lowpass"] = p.bandpass.l_freq, p.bandpass.h_freq
        an.info["bads"] = list(bads)
        bad256 = (seg.bad_spans // dec).astype(np.int64)
        valid256 = (seg.valid_spans // dec).astype(np.int64)
        n256 = an.n_times
        adapter.clean_from_parts(an, ica, sel["excluded"])
        qc["interpolated"] = list(bads)

        # ASR on the EEG
        eeg_idx = np.arange(n_eeg)
        valid_mask = st.spans_to_mask(valid256, n256)
        qc["asr"] = {"enabled": self.use_asr}
        if self.use_asr:
            t = time.perf_counter()
            before = an._data[eeg_idx].copy()
            win = int(round(p.asr.window_s * sf))
            starts = st.cleanest_windows(before, valid256, win, int(p.asr.calibration_s / p.asr.window_s))
            cal = np.concatenate([before[:, s : s + win] for s in starts], axis=1)
            after = _asr(cal, before, sf, A.eeg, p.asr.cutoff)
            an._data[eeg_idx] = np.where(valid_mask, after, 0.0)
            qc["asr"].update(
                {
                    "cutoff": p.asr.cutoff,
                    "calibration_windows": int(len(starts)),
                    "repaired_fraction": st.repaired_fraction(
                        before, an._data[eeg_idx], valid_mask, p.asr.repaired_threshold
                    ),
                    "seconds": round(time.perf_counter() - t, 1),
                }
            )
            del before, after

        # residual artefact windows
        res = st.residual_spans(
            an._data[eeg_idx], valid256, int(p.residual.window_s * sf), p.residual.ptp_uv * 1e-6
        )
        good_mask = valid_mask & ~st.spans_to_mask(res, n256)
        qc["residual_fraction_of_valid"] = float(1 - good_mask.sum() / max(valid_mask.sum(), 1))
        qc["clean_seconds"] = float(good_mask.sum() / sf)
        _annotate(an, bad256, res, A.joins // dec, sf)

        # events, blocks, retention
        ev = A.events.copy()
        ev["sample"] = (ev["sample_rec"] // dec).astype(np.int64)
        ev["onset_s"] = ev["sample"] / sf
        pat = cfg.dataset.trial_patterns.get(rec.paradigm)
        e = p.epochs
        if pat is not None:
            is_trial = ev["trial_type"].str.match(pat).to_numpy()
            ok = np.zeros(len(ev), bool)
            ok[is_trial] = st.trial_ok(
                ev.loc[is_trial, "sample"].to_numpy(),
                an._data[eeg_idx],
                good_mask,
                sf,
                e.tmin,
                e.tmax,
                e.reject_ptp_uv * 1e-6,
            )
            ev["artefact_free"] = np.where(
                is_trial, ok.astype(float), np.nan
            )  # 1/0 for trials, NaN otherwise
            n_trials, n_ok = int(is_trial.sum()), int(ok.sum())
            qc["n_trials"], qc["n_trials_artefact_free"] = n_trials, n_ok
            qc["trials_artefact_free_by_type"] = (
                ev.loc[is_trial].groupby("trial_type")["artefact_free"].sum().astype(int).to_dict()
            )
        blocks = A.blocks.copy()
        blocks["start_sample"] = blocks["start_sample"] // dec
        blocks["stop_sample"] = blocks["stop_sample"] // dec
        blocks["clean_s"] = [
            good_mask[a:b].sum() / sf for a, b in zip(blocks.start_sample, blocks.stop_sample, strict=True)
        ]
        qc["n_blocks"] = int(len(blocks))

        # save
        an.save(paths["raw"], fmt="single", overwrite=True, verbose="error")
        ev.drop(columns=["sample_rec"]).to_csv(paths["events"], sep="\t", index=False)
        blocks.to_csv(paths["blocks"], sep="\t", index=False)

        # minimum retention
        mr = p.minimum_retained
        if rec.paradigm in ("Restingstate", cfg.dataset.motor_paradigm):
            if qc["clean_seconds"] < mr.rest_s:
                return f"missing:{qc['clean_seconds']:.0f} s clean rest (< {mr.rest_s:.0f})"
        elif qc.get("n_trials_artefact_free", 0) < mr.evoked_trials:
            return (
                f"missing:{qc.get('n_trials_artefact_free', 0)} artefact-free trials (< {mr.evoked_trials})"
            )
        return "ok"


def _asr(
    cal: np.ndarray, X: np.ndarray, sf: float, names: list[str] | None, cutoff: float, chunk_s: float = 0.5
) -> np.ndarray:
    """ASR (meegkit) calibrated on ``cal``, applied to ``X`` (ch x samples).

    meegkit's ``transform`` is a streaming step: it estimates one covariance from the samples it is given (with a
    2-s memory of earlier calls) and reconstructs them together. It is therefore called on consecutive
    ``chunk_s`` chunks, as in EEGLAB's ``asr_process``; one call on a whole recording would apply a single global
    projection. Data are scaled to microvolts
    and back so that its tolerances see EEG-sized numbers.
    """
    from meegkit.asr import ASR

    del names
    asr = ASR(sfreq=sf, cutoff=cutoff, method="euclid")
    asr.fit(cal * 1e6)
    step = max(int(round(chunk_s * sf)), 1)
    out = np.empty_like(X)
    for a in range(0, X.shape[1], step):
        out[:, a : a + step] = asr.transform(X[:, a : a + step] * 1e6) * 1e-6
    return out


def _annotate(raw, bad: np.ndarray, residual: np.ndarray, joins: np.ndarray, sf: float) -> None:
    import mne

    on, du, de = [], [], []
    for spans, desc in ((bad, "BAD_segment"), (residual, "BAD_residual")):
        for a, b in np.asarray(spans, np.int64).reshape(-1, 2):
            on.append(a / sf)
            du.append((b - a) / sf)
            de.append(desc)
    for j in np.asarray(joins, np.int64):
        on.append(j / sf)
        du.append(0.0)
        de.append("EDGE boundary")
    raw.set_annotations(mne.Annotations(on, du, de))


def _json(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def preprocess_all(cfg: Config, variant: str = "primary", asr: bool | None = None) -> None:
    """Preprocess every recording of every usable child (seated paradigms with :class:`BiosemiPreprocessor`, the
    motor paradigm with :class:`~preprocessing.brainvision.BrainVisionPreprocessor`) and write the QC summary.
    ``variant="no_asr", asr=False`` gives the preprocessing without ASR of Table S8; ``config/muscle_variant.yaml``
    with ``variant="muscle_argmax"`` the muscle variant of Table S6."""
    from preprocessing.brainvision import BrainVisionPreprocessor
    from preprocessing.qc_summary import qc_summary
    from utils.dataset import BidsDataset, Participants, build_recordings

    ds = BidsDataset.from_config(cfg)
    usable = Participants.load(ds, cfg).usable_ids
    for rec in build_recordings(ds, cfg, subjects=usable):
        pre = BiosemiPreprocessor(cfg, variant=variant, asr=asr)
        if not pre.is_done(rec):
            pre.run(rec)
    for rec in build_recordings(ds, cfg, subjects=usable, paradigms=[cfg.dataset.motor_paradigm]):
        pre = BrainVisionPreprocessor(cfg, variant=variant, asr=asr)
        if not pre.is_done(rec):
            pre.run(rec)
    qc_summary(cfg, variant)


if __name__ == "__main__":
    from config.loader import load_config

    preprocess_all(load_config())
