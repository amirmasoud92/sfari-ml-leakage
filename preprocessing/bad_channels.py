"""Bad EEG channel detection: PyPREP ``NoisyChannels`` inside a robust-reference loop (Methods, Preprocessing;
Additional file 1, Supplementary Methods S1).

1. Input (:func:`detection_data`): the 1-100 Hz copy (per-segment FIR; MNE ``notch`` at the mains frequencies,
   also per segment), restricted to the valid spans and concatenated, optionally decimated (512 -> 256 Hz after the
   100-Hz low-pass). Joins, rail, flat and unusable spans never reach PyPREP, and each piece was high-passed on its
   own, so the splice points carry no DC step. PyPREP's own detrending is therefore switched off
   (``do_detrend=False``).
2. Robust reference loop (:func:`find_bad_channels`), PREP-style: the first pass runs on data referenced to the
   per-sample channel *median*; every further pass on data referenced to the *mean of the channels not yet bad*.
   Bad sets only grow; the loop stops when a pass adds nothing (or after ``max_iterations``). Detection on a
   reference-free (CMS) signal is avoided on purpose: the channels next to CMS/DRL (P1/PO3/POz) have near-zero
   CMS-referenced variance and would be flagged.
3. Criteria (PyPREP, PREP defaults unless configured; here NaN/flat, deviation, correlation (+ dropout) with
   ``correlation_min_bad_windows`` and RANSAC once on the final reference): NaN/flat, deviation (robust z of the
   IQR-based amplitude, so sparse blinks do not flag Fp channels as a variance rule would), HF noise, correlation
   (+ dropout), SNR, optional PSD, and RANSAC (deterministic ``random_state``). Plus two non-PyPREP sources:
   manually marked channels (not used here), and channels whose raw value exceeds +-250 mV in more than
   ``saturation.max_frac`` of the valid samples (``saturation``; intermittent rail-to-rail faults).

The result lists every bad channel with the criteria that flagged it. Detection that was skipped or cut short (too
little data, RANSAC failure, < 8 good channels for the final RANSAC) is listed in ``errors``.

RANSAC seed: the bad set is deterministic for the fixed seed but can change by one borderline channel with another
seed.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from preprocessing.reference import median_reference
from preprocessing.segments import SegmentInfo
from preprocessing.step_config import BadChannelsConfig
from utils.logging_utils import get_logger

log = get_logger(__name__)

#: PyPREP ``get_bads(as_dict=True)`` key -> our criterion label, and the config criterion that enables it.
PYPREP_KEYS: dict[str, tuple[str, str]] = {
    "bad_by_nan": ("nan", "nan_flat"),
    "bad_by_flat": ("flat", "nan_flat"),
    "bad_by_deviation": ("deviation", "deviation"),
    "bad_by_hf_noise": ("hf_noise", "hf_noise"),
    "bad_by_correlation": ("correlation", "correlation"),
    "bad_by_dropout": ("dropout", "correlation"),
    "bad_by_SNR": ("snr", "snr"),
    "bad_by_psd": ("psd", "psd"),
    "bad_by_ransac": ("ransac", "ransac"),
}


@dataclass
class BadChannelResult:
    """Outcome of :func:`find_bad_channels`.

    Attributes
    ----------
    bads
        Sorted bad EEG channel names.
    reasons
        Channel -> sorted list of criteria that flagged it (any iteration).
    iterations
        Per pass: reference used and the channels flagged per criterion.
    converged
        Whether the last pass added no channel.
    """

    bads: list[str]
    reasons: dict[str, list[str]]
    iterations: list[dict]
    converged: bool
    n_samples: int
    sfreq: float
    seconds: float
    saturation_frac: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def by_criterion(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for ch, rs in self.reasons.items():
            for r in rs:
                out.setdefault(r, []).append(ch)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def qc(self) -> dict:
        return {"bads": self.bads, "n_bads": len(self.bads), "reasons": self.reasons,
                "by_criterion": self.by_criterion(), "n_iterations": len(self.iterations),
                "converged": self.converged, "iterations": self.iterations, "n_samples": self.n_samples,
                "sfreq": self.sfreq, "seconds": round(self.seconds, 2),
                "saturation_frac_nonzero": {k: v for k, v in self.saturation_frac.items() if v > 0},
                "channel_order": self.names, "errors": self.errors}


def valid_concat(data: np.ndarray, seg: SegmentInfo, decimate: int = 1) -> np.ndarray:
    """Concatenate the valid spans of ``data`` (channels x samples), keeping every ``decimate``-th sample of each."""
    parts = [data[:, a:b:decimate] for a, b in seg.valid_spans]
    if not parts:
        return np.zeros((data.shape[0], 0), data.dtype)
    return np.concatenate(parts, axis=1)


def detection_data(hp_raw, seg: SegmentInfo, cfg: BadChannelsConfig, eeg_names: Sequence[str]) -> np.ndarray:
    """EEG of the (already per-segment filtered and notched) copy on valid spans only, decimated."""
    idx = [hp_raw.ch_names.index(c) for c in eeg_names]
    return valid_concat(hp_raw._data[idx], seg, cfg.decimate)


def saturation_fractions(raw_eeg: np.ndarray, seg: SegmentInfo, names: Sequence[str], threshold_v: float) -> dict[str, float]:
    """Per channel: fraction of valid samples of the *raw* (unfiltered) signal with ``|x| > threshold_v``."""
    n = 0
    hits = np.zeros(raw_eeg.shape[0], np.int64)
    for a, b in seg.valid_spans:
        hits += (np.abs(raw_eeg[:, a:b]) > threshold_v).sum(axis=1)
        n += b - a
    return {c: (float(h) / n if n else 0.0) for c, h in zip(names, hits)}


def _make_raw(data: np.ndarray, sfreq: float, names: Sequence[str], montage: str):
    import mne

    info = mne.create_info(list(names), sfreq, "eeg")
    r = mne.io.RawArray(data, info, verbose="error")
    r.set_montage(montage, on_missing="raise", verbose="error")
    return r


def _scores(nc, names: Sequence[str]) -> dict[str, list[float | None]]:
    """Per-channel PyPREP statistics of one pass (for offline threshold evaluation; see :data:`SCORE_KEYS`)."""
    ex = getattr(nc, "_extra_info", {}) or {}

    def get(crit: str, key: str):
        v = (ex.get(crit) or {}).get(key)
        if v is None:
            return None
        v = np.asarray(v, float).ravel()
        return [None if not np.isfinite(x) else round(float(x), 4) for x in v] if v.size == len(names) else None

    out = {
        "deviation_z": get("bad_by_deviation", "robust_channel_deviations"),
        "hf_noise_z": get("bad_by_hf_noise", "hf_noise_zscores"),
        "correlation_bad_frac": get("bad_by_correlation", "bad_window_fractions"),
        "correlation_median_max": get("bad_by_correlation", "median_max_correlations"),
        "dropout_frac": get("bad_by_dropout", "bad_window_fractions"),
        "ransac_bad_frac": get("bad_by_ransac", "bad_window_fractions"),
    }
    return {k: v for k, v in out.items() if v is not None}


#: Score lists stored per pass (channel order = ``names``): robust deviation z, HF-noise z, fraction of
#: correlation windows below threshold, median max correlation, dropout fraction, RANSAC bad-window fraction.
SCORE_KEYS = ("deviation_z", "hf_noise_z", "correlation_bad_frac", "correlation_median_max", "dropout_frac",
              "ransac_bad_frac")


def _apply_min_bad_windows(nc, names: Sequence[str], k: int) -> None:
    """Keep a correlation / dropout flag only if it rests on at least ``k`` bad windows.

    PyPREP flags a channel when the *fraction* of bad correlation windows exceeds ``frac_bad`` (PREP default 0.01).
    In a 60-s resting run that fraction is exceeded by a single 1-s window, so the criterion depends on the
    recording length; the count condition removes that dependence for short runs and is inactive for long ones
    (k / n_windows < frac_bad).
    """
    if k <= 1:
        return
    ex = getattr(nc, "_extra_info", None)
    if not isinstance(ex, dict):
        raise RuntimeError("PyPREP NoisyChannels has no _extra_info: the correlation_min_bad_windows rule "
                           " cannot be applied with this PyPREP version")
    for crit, attr, win_key in (("bad_by_correlation", "bad_by_correlation", "max_correlations"),
                                ("bad_by_dropout", "bad_by_dropout", "dropouts")):
        info = ex.get(crit)
        if not isinstance(info, dict) or info.get("bad_window_fractions") is None or info.get(win_key) is None:
            raise RuntimeError(f"PyPREP _extra_info[{crit!r}] lacks 'bad_window_fractions' / {win_key!r}: the "
                               f"correlation_min_bad_windows rule cannot be applied with this PyPREP "
                               f"version")
        frac = info["bad_window_fractions"]
        n_win = np.asarray(info[win_key]).shape[1]
        frac = np.asarray(frac, float)
        keep = [c for c in getattr(nc, attr) if np.rint(frac[list(names).index(c)] * n_win) >= k]
        setattr(nc, attr, keep)


def run_noisy_channels(data: np.ndarray, sfreq: float, names: Sequence[str], cfg: BadChannelsConfig, seed: int,
                       montage: str = "biosemi64", scores: dict | None = None,
                       criteria: Iterable[str] | None = None) -> tuple[dict[str, list[str]], list[str]]:
    """One PyPREP pass on already-referenced data; returns (criterion -> channels, errors).

    If ``scores`` is a dict, it receives the per-channel statistics of the pass (:func:`_scores`).
    ``criteria`` overrides ``cfg.criteria`` for this pass.
    """
    import logging

    from pyprep import NoisyChannels

    logging.getLogger("pyprep").setLevel(logging.WARNING)
    crit = set(cfg.criteria if criteria is None else criteria)
    raw = _make_raw(data, sfreq, names, montage)
    nc = NoisyChannels(raw, do_detrend=False, random_state=seed, matlab_strict=False,
                       ransac="ransac" in crit, correlation="correlation" in crit)
    errors: list[str] = []
    if "deviation" in crit:
        nc.find_bad_by_deviation(deviation_threshold=cfg.deviation_threshold)
    if "hf_noise" in crit:
        nc.find_bad_by_hfnoise(HF_zscore_threshold=cfg.hf_noise_threshold)
    if "correlation" in crit:
        nc.find_bad_by_correlation(correlation_secs=cfg.correlation_secs,
                                   correlation_threshold=cfg.correlation_threshold,
                                   frac_bad=cfg.correlation_frac_bad)
        _apply_min_bad_windows(nc, names, cfg.correlation_min_bad_windows)
    if "snr" in crit:
        nc.find_bad_by_SNR()
    if "psd" in crit:
        nc.find_bad_by_PSD()
    if "ransac" in crit:
        r = cfg.ransac
        try:
            nc.find_bad_by_ransac(n_samples=r.n_samples, sample_prop=r.sample_prop, corr_thresh=r.corr_thresh,
                                  frac_bad=r.frac_bad, corr_window_secs=r.corr_window_secs,
                                  channel_wise=r.channel_wise)
        except (OSError, ValueError, MemoryError) as e:  # e.g. too few good channels left for RANSAC
            errors.append(f"ransac: {type(e).__name__}: {e}")
    got = nc.get_bads(as_dict=True)
    if scores is not None:
        scores.update(_scores(nc, names))
    out: dict[str, list[str]] = {}
    for key, (label, enabled_by) in PYPREP_KEYS.items():
        if enabled_by in crit and got.get(key):
            out[label] = sorted(map(str, got[key]))
    return out, errors


def find_bad_channels(data: np.ndarray, sfreq: float, names: Sequence[str], cfg: BadChannelsConfig, *, seed: int,
                      manual: Iterable[str] = (), saturation_frac: dict[str, float] | None = None,
                      montage: str = "biosemi64") -> BadChannelResult:
    """Bad EEG channels of the detection data (see the module docstring).

    Parameters
    ----------
    data
        ``(n_eeg, n)`` detection data (:func:`detection_data`), volts, any reference.
    sfreq
        Its sampling rate.
    names
        EEG channel names (rows of ``data``).
    cfg
        ``bad_channels`` config.
    seed
        RANSAC random state.
    manual
        Channels already known to be bad (manual list).
    saturation_frac
        Output of :func:`saturation_fractions` (raw data), or ``None``.
    """
    t0 = time.perf_counter()
    names = list(names)
    reasons: dict[str, set[str]] = {}

    def add(chs: Iterable[str], why: str) -> None:
        for c in chs:
            if c in names:
                reasons.setdefault(c, set()).add(why)

    add(manual, "manual")
    sat = saturation_frac or {}
    if cfg.saturation_max_frac is not None:
        add([c for c, f in sat.items() if f > cfg.saturation_max_frac], "saturation")
    iterations: list[dict] = []
    errors: list[str] = []
    converged = False
    final_ransac = cfg.ransac_pass == "final" and "ransac" in cfg.criteria
    loop_crit = [c for c in cfg.criteria if not (final_ransac and c == "ransac")]
    if data.shape[1] < int(10 * sfreq):
        errors.append(f"only {data.shape[1] / sfreq:.1f} s of valid data; PyPREP not run")
    else:
        for it in range(cfg.max_iterations):
            before = set(reasons)
            if it == 0 and cfg.initial_reference == "median":
                x = median_reference(data)
                ref_name = "median"
            else:
                good = [i for i, c in enumerate(names) if c not in reasons]
                if len(good) < 3:
                    errors.append("fewer than 3 good channels; loop stopped")
                    break
                x = data - data[good].mean(axis=0, keepdims=True)
                ref_name = f"mean_of_{len(good)}_good"
            sc: dict = {}
            flagged, errs = run_noisy_channels(x, sfreq, names, cfg, seed, montage, scores=sc, criteria=loop_crit)
            errors.extend(f"iteration {it}: {e}" for e in errs)
            for crit_name, chs in flagged.items():
                add(chs, crit_name)
            added = sorted(set(reasons) - before)
            iterations.append({"iteration": it, "reference": ref_name, "flagged": flagged, "added": added,
                               "scores": sc})
            if it > 0 and not added:
                converged = True
                break
            if it == 0 and not added and cfg.initial_reference != "median":
                converged = True
                break
        if final_ransac:
            # one RANSAC pass on the final robust reference, predicting from the channels that are still good
            # (channels already bad never serve as RANSAC predictors)
            good = [i for i, c in enumerate(names) if c not in reasons]
            if len(good) >= 8:
                x = data[good] - data[good].mean(axis=0, keepdims=True)
                gnames = [names[i] for i in good]
                sc = {}
                flagged, errs = run_noisy_channels(x, sfreq, gnames, cfg, seed, montage, scores=sc,
                                                   criteria=["ransac"])
                errors.extend(f"final ransac: {e}" for e in errs)
                before = set(reasons)
                for crit_name, chs in flagged.items():
                    add(chs, crit_name)
                full = {k: [None] * len(names) for k in sc}
                for k, v in sc.items():
                    for gi, val in zip(good, v):
                        full[k][gi] = val
                iterations.append({"iteration": len(iterations), "reference": f"mean_of_{len(good)}_good",
                                   "pass": "final_ransac", "flagged": flagged,
                                   "added": sorted(set(reasons) - before), "scores": full})
            else:
                errors.append("final ransac skipped: fewer than 8 good channels")
    bads = sorted(reasons, key=names.index)
    if len(bads) > cfg.max_bad_fraction_warn * len(names):
        log.warning("%d of %d EEG channels flagged bad", len(bads), len(names))
    return BadChannelResult(bads=bads, reasons={c: sorted(reasons[c]) for c in bads}, iterations=iterations,
                            converged=converged, n_samples=int(data.shape[1]), sfreq=float(sfreq),
                            seconds=time.perf_counter() - t0, saturation_frac=dict(sat), errors=errors, names=names)
