"""Access to the cleaned recordings and the signal operations shared by the feature, marker and deep-learning
code: clean-segment selection (2-s segments that touch no bad span and no long inter-trial pause), epoching,
Welch spectra, 1-Hz log-power bins and specparam fits.

Only one recording is held in memory at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from config.loader import Config
from preprocessing.biosemi import output_paths
from preprocessing.steps import spans_to_mask


@dataclass
class CleanRecording:
    """One cleaned recording: data (volts, 256 Hz), good-sample mask, events, blocks and QC."""

    cfg: Config
    child: str
    paradigm: str
    variant: str = "primary"
    # "alternative": equal-duration clean-data blocks (robustness rerun); "standing": motor standing runs only
    block_mode: str = "primary"

    @cached_property
    def paths(self) -> dict[str, Path]:
        return output_paths(self.cfg, self.child, self.paradigm, self.variant)

    @cached_property
    def qc(self) -> dict:
        return json.loads(self.paths["qc"].read_text(encoding="utf-8"))

    @property
    def usable(self) -> bool:
        if not self.paths["qc"].is_file():
            return False
        return self.qc.get("status") == "ok" and self.paths["raw"].is_file()

    @cached_property
    def raw(self):
        import mne

        return mne.io.read_raw_fif(self.paths["raw"], preload=True, verbose="error")

    @property
    def sfreq(self) -> float:
        return float(self.raw.info["sfreq"])

    @cached_property
    def eeg_names(self) -> list[str]:
        import mne

        return [self.raw.ch_names[i] for i in mne.pick_types(self.raw.info, eeg=True)]

    @cached_property
    def exg_names(self) -> list[str]:
        return [c for c in self.raw.ch_names if c.startswith("EXG")]

    def data(self, picks: list[str] | None = None) -> np.ndarray:
        idx = [self.raw.ch_names.index(c) for c in (picks or self.eeg_names)]
        return self.raw._data[idx]

    @cached_property
    def good_mask(self) -> np.ndarray:
        """True for samples outside every ``BAD*`` annotation. The first sample after a run or pause join (an
        ``EDGE boundary`` annotation) is False as well, so that no segment straddles two separately filtered
        pieces."""
        n = self.raw.n_times
        sf = self.sfreq
        bad, edges = [], []
        for o, d, de in zip(
            self.raw.annotations.onset - self.raw.first_time,
            self.raw.annotations.duration,
            self.raw.annotations.description,
            strict=True,
        ):
            if de.upper().startswith("BAD"):
                bad.append((int(round(o * sf)), int(round((o + d) * sf))))
            elif de.upper().startswith("EDGE"):
                edges.append(int(round(o * sf)))
        m = ~spans_to_mask(np.array(bad, np.int64).reshape(-1, 2), n)
        e = np.array([k for k in edges if 0 < k < n], np.int64)
        m[e] = False
        return m

    @cached_property
    def events(self) -> pd.DataFrame:
        return pd.read_csv(self.paths["events"], sep="\t", keep_default_na=False, dtype={"trial_type": str})

    @cached_property
    def primary_blocks(self) -> pd.DataFrame:
        return pd.read_csv(self.paths["blocks"], sep="\t")

    @cached_property
    def _alternative(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Alternative blocks: consecutive ``window_s`` of allowed clean data, at most ``max_blocks``,
        cut from the union of the primary blocks' allowed samples."""
        alt = self.cfg.blocks.alternative
        M = np.zeros(self.raw.n_times, bool)
        for _, b in self.primary_blocks.iterrows():
            M |= self._primary_allowed(b)
        cs = np.cumsum(M)
        win = int(round((alt.window_s or 60.0) * self.sfreq))
        rows = []
        for k in range(int(alt.max_blocks or 6)):
            lo, hi = k * win, (k + 1) * win
            if cs[-1] < hi:
                break
            a = int(np.searchsorted(cs, lo + 1))
            b = int(np.searchsorted(cs, hi)) + 1
            rows.append({"block": k, "start_sample": a, "stop_sample": b, "clean_s": win / self.sfreq})
        return pd.DataFrame(rows, columns=["block", "start_sample", "stop_sample", "clean_s"]), M

    @property
    def blocks(self) -> pd.DataFrame:
        if self.block_mode == "alternative":
            return self._alternative[0]
        b = self.primary_blocks
        if self.block_mode == "standing" and "walking" in b:
            return b[~b["walking"].astype(bool)].reset_index(drop=True)
        return b

    def trials(self, pattern: str | None = None, artefact_free: bool = True) -> pd.DataFrame:
        pat = pattern or self.cfg.dataset.trial_patterns[self.paradigm]
        ev = self.events
        t = ev[ev["trial_type"].str.match(pat)]
        if artefact_free and "artefact_free" in t:
            t = t[pd.to_numeric(t["artefact_free"], errors="coerce") == 1]
        return t

    def allowed_mask(self, block: pd.Series) -> np.ndarray:
        """Samples of one block usable for continuous features: inside the block, good, and (task paradigms) not
        inside an inter-trial gap longer than ``features.max_gap_s``."""
        if self.block_mode == "alternative":
            M = self._alternative[1]
            m = np.zeros(M.size, bool)
            a, b = int(block["start_sample"]), int(block["stop_sample"])
            m[a:b] = M[a:b]
            return m
        return self._primary_allowed(block)

    def _primary_allowed(self, block: pd.Series) -> np.ndarray:
        n = self.raw.n_times
        sf = self.sfreq
        a, b = int(block["start_sample"]), int(block["stop_sample"])
        m = np.zeros(n, bool)
        if self.paradigm in ("Restingstate", self.cfg.dataset.motor_paradigm):
            m[a:b] = True
        else:
            on = self.trials(artefact_free=False)
            s = np.sort(on["sample"].to_numpy(np.int64))
            s = s[(s >= a) & (s < b)]
            gap = np.diff(np.append(s, s[-1] + int(2 * sf))) if s.size else np.zeros(0)
            maxg = self.cfg.features.max_gap_s * sf
            for si, g in zip(s, gap, strict=True):
                end = si + (g if g <= maxg else int(2 * sf))
                m[max(si - int(sf), a) : min(end, b)] = True
        return m & self.good_mask

    def segment_starts(self, block: pd.Series, seg_s: float, step_s: float) -> np.ndarray:
        """Starts of ``seg_s`` windows stepped by ``step_s`` that lie wholly in :meth:`allowed_mask`."""
        m = self.allowed_mask(block)
        w = int(round(seg_s * self.sfreq))
        step = int(round(step_s * self.sfreq))
        c = np.concatenate([[0], np.cumsum(m)])
        starts = np.arange(int(block["start_sample"]), int(block["stop_sample"]) - w + 1, step)
        ok = (c[starts + w] - c[starts]) == w
        return starts[ok]

    def segments(
        self,
        block: pd.Series,
        picks: list[str] | None = None,
        seg_s: float | None = None,
        step_s: float | None = None,
    ) -> np.ndarray:
        seg_s = seg_s or self.cfg.features.segment_s
        st = self.segment_starts(block, seg_s, step_s or seg_s)
        X = self.data(picks)
        w = int(round(seg_s * self.sfreq))
        return np.stack([X[:, s : s + w] for s in st]) if st.size else np.zeros((0, X.shape[0], w))

    def epochs(
        self,
        samples: np.ndarray,
        tmin: float,
        tmax: float,
        picks: list[str] | None = None,
        baseline: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(n, ch, t)`` epochs and their time axis; epochs leaving the recording are dropped."""
        sf = self.sfreq
        a0, b0 = int(round(tmin * sf)), int(round(tmax * sf)) + 1
        X = self.data(picks)
        s = np.asarray(samples, np.int64)
        s = s[(s + a0 >= 0) & (s + b0 <= X.shape[1])]
        E = np.stack([X[:, k + a0 : k + b0] for k in s]) if s.size else np.zeros((0, X.shape[0], b0 - a0))
        t = np.arange(a0, b0) / sf
        if baseline is not None and E.size:
            bm = (t >= baseline[0]) & (t <= baseline[1])
            E = E - E[:, :, bm].mean(axis=2, keepdims=True)
        return E, t


def welch_from_segments(S: np.ndarray, sf: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean Hann-windowed periodogram over segments ``(n, ch, t)`` -> (freqs, psd ``(ch, f)``) in V^2/Hz."""
    n, _, w = S.shape
    win = np.hanning(w)
    scale = 1.0 / (sf * (win**2).sum())
    F = np.fft.rfft((S - S.mean(axis=2, keepdims=True)) * win, axis=2)
    P = (np.abs(F) ** 2) * scale
    P[:, :, 1:-1] *= 2
    return np.fft.rfftfreq(w, 1 / sf), P.mean(axis=0)


def bin_log_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float, width: float = 1.0) -> tuple:
    """log10 power averaged into ``width``-Hz bins centred on lo..hi (inclusive)."""
    centres = np.arange(lo, hi + 1e-9, width)
    out = np.stack(
        [psd[:, (freqs >= c - width / 2) & (freqs < c + width / 2)].mean(axis=1) for c in centres], 1
    )
    return centres, np.log10(out)


@dataclass(frozen=True)
class SpecFit:
    offset: float
    knee: float
    exponent: float
    r2: float
    error: float
    peaks: np.ndarray  # (k, 3) CF, PW, BW


def fit_specparam(freqs: np.ndarray, psd: np.ndarray, cfg: Config, mode: str) -> list[SpecFit | None]:
    """specparam per row of ``psd`` (channels x freqs); ``None`` where the fit fails."""
    from specparam import SpectralModel

    sp = cfg.features.spectral.specparam
    out: list[SpecFit | None] = []
    for row in psd:
        if not np.all(np.isfinite(row)) or np.any(row <= 0):
            out.append(None)
            continue
        fm = SpectralModel(
            peak_width_limits=list(sp.peak_width_limits),
            max_n_peaks=sp.max_n_peaks,
            min_peak_height=sp.min_peak_height,
            peak_threshold=sp.peak_threshold,
            aperiodic_mode=mode,
            verbose=False,
        )
        try:
            fm.fit(freqs, row, list(sp.freq_range))
            ap = np.atleast_1d(fm.get_params("aperiodic"))
            pk = np.atleast_2d(fm.get_params("peak"))
            pk = pk[np.all(np.isfinite(pk), axis=1)] if pk.size else np.zeros((0, 3))
            r2 = float(fm.get_metrics("gof"))
            err = float(fm.get_metrics("error"))
        except Exception:  # noqa: BLE001 - a failed fit is a missing value, counted in the logs
            out.append(None)
            continue
        if mode == "knee":
            out.append(SpecFit(float(ap[0]), float(ap[1]), float(ap[2]), r2, err, pk))
        else:
            out.append(SpecFit(float(ap[0]), 0.0, float(ap[1]), r2, err, pk))
    return out


def aperiodic_log10(freqs: np.ndarray, fit: SpecFit) -> np.ndarray:
    return fit.offset - np.log10(fit.knee + freqs**fit.exponent)


def periodic_band(freqs: np.ndarray, psd_row: np.ndarray, fit: SpecFit, band: tuple[float, float]) -> float:
    """Mean of the aperiodic-removed log10 spectrum in ``band`` (periodic power)."""
    m = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.mean(np.log10(psd_row[m]) - aperiodic_log10(freqs[m], fit)))


def largest_peak(fit: SpecFit, band: tuple[float, float]) -> tuple[float, float]:
    """(CF, PW) of the highest peak with CF in ``band``; (nan, 0) if none."""
    pk = fit.peaks
    if pk.size == 0:
        return float("nan"), 0.0
    sel = pk[(pk[:, 0] >= band[0]) & (pk[:, 0] <= band[1])]
    if sel.size == 0:
        return float("nan"), 0.0
    i = int(np.argmax(sel[:, 1]))
    return float(sel[i, 0]), float(sel[i, 1])
