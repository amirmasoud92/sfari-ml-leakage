"""Translation of ``config/analysis.yaml`` into the preprocessing step configurations, and thin wrappers around
the preprocessing steps used by :mod:`preprocessing.biosemi`: status-channel joins, rail / flat span detection,
per-span FIR filtering and notching, PyPREP robust-reference bad-channel detection, Picard ICA with ICLabel and the
posterior-alpha veto, and :func:`~preprocessing.cleaning.clean_from_parts` (apply ICA, spherical-spline
interpolation, robust average reference).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from config.loader import Config


@lru_cache(maxsize=1)
def _ap() -> SimpleNamespace:
    """The preprocessing step modules, imported lazily (they pull in MNE / PyPREP)."""
    from preprocessing import bad_channels, cleaning, filtering, ica, ica_selection, qc_metrics, segments
    from preprocessing import status_channel as status
    from preprocessing import step_config as apc

    return SimpleNamespace(
        status=status,
        bad=bad_channels,
        filt=filtering,
        ica=ica,
        sel=ica_selection,
        io=cleaning,
        qc=qc_metrics,
        seg=segments,
        cfg=apc,
    )


@dataclass(frozen=True)
class StepConfigs:
    """The configuration objects of the preprocessing steps, built from ``config/analysis.yaml``."""

    segments: Any
    bad_channels: Any
    ica: Any
    selection: Any
    hp_filter: Any
    analysis_filter: Any
    notch_freqs: tuple[float, ...]


def step_configs(cfg: Config) -> StepConfigs:
    """Translate the ``preprocessing`` section into the step configuration dataclasses."""
    apc = _ap().cfg
    AlphaVetoConfig, SelectionConfig = _ap().sel.AlphaVetoConfig, _ap().sel.SelectionConfig

    p = cfg.preprocessing
    s = p.segments
    seg = apc.SegmentsConfig(
        join_annotation="EDGE boundary",
        rail_threshold_v=s.rail_threshold_v,
        rail=apc.SpanRule(s.rail.pad_s, s.rail.merge_gap_s, s.rail.min_duration_s),
        flat_max_abs_diff_uv=s.flat_max_abs_diff_uv,
        flat=apc.SpanRule(s.flat.pad_s, s.flat.merge_gap_s, s.flat.min_duration_s),
        zero_skip_spans=True,
        min_valid_span_s=s.min_valid_span_s,
    )
    b = p.bad_channels
    bad = apc.BadChannelsConfig(
        method="pyprep_noisychannels",
        criteria=tuple(b.criteria),
        deviation_threshold=b.deviation_threshold,
        hf_noise_threshold=5.0,
        correlation_secs=b.correlation_secs,
        correlation_threshold=b.correlation_threshold,
        correlation_frac_bad=b.correlation_frac_bad,
        ransac=apc.RansacConfig(
            b.ransac.n_samples,
            b.ransac.sample_prop,
            b.ransac.corr_thresh,
            b.ransac.frac_bad,
            b.ransac.corr_window_secs,
            False,
        ),
        saturation_threshold_v=b.saturation.threshold_v,
        saturation_max_frac=b.saturation.max_frac,
        notch=True,
        decimate=2,
        initial_reference="median",
        max_iterations=b.max_iterations,
        max_bad_fraction_warn=b.max_bad_fraction,
        manual_bads_from_known_issues=False,
        ransac_pass="final",
        correlation_min_bad_windows=b.correlation_min_bad_windows,
    )
    i = p.ica
    ica = apc.ICAConfig(
        enabled=True,
        method=i.method,
        fit_params=dict(i.fit_params),
        max_iter=i.max_iter,
        n_components="auto",
        max_components=None,
        min_samples_factor=float(i.min_samples_factor),
        decimate=2,
        fit_reject_ptp_uv=i.fit_reject_ptp_uv,
        fit_reject_chunk_s=1.0,
        iclabel_rule=i.iclabel_rule,
        iclabel_exclude=dict(i.iclabel_exclude),
        apply_in_primary=True,
    )
    v = i.alpha_veto
    sel = SelectionConfig(
        alpha_veto=AlphaVetoConfig(
            enabled=True,
            band_hz=tuple(v.band_hz),
            min_peak_db=v.min_peak_db,
            aperiodic_fit_hz=tuple(v.aperiodic_fit_hz),
            welch_window_s=2.0,
            welch_overlap=0.5,
            min_posterior_share=v.min_posterior_share,
            posterior_channels=tuple(v.posterior_channels),
        ),
        mark_fit_rejected=True,
        fit_rejected_description="BAD_icafit",
    )
    bp = p.bandpass
    hp = apc.FilterSpec(l_freq=b.detection_band[0], h_freq=b.detection_band[1])
    an = apc.FilterSpec(
        l_freq=bp.l_freq, h_freq=bp.h_freq, fir_window=bp.fir_window, fir_design=bp.fir_design, phase=bp.phase
    )
    return StepConfigs(seg, bad, ica, sel, hp, an, (p.line_freq, 2 * p.line_freq))


def status_joins(path: Path, join_bit: int) -> np.ndarray:
    """Samples of BioSemi pause/resume joins (rising edges of the Status join bit)."""
    st = _ap().status.read_status_channel(path)
    dec = _ap().status.decode_status(st.values, st.sfreq, join_bit=join_bit, record_samples=st.record_samples)
    return dec.joins


def segment_info(
    eeg_v: np.ndarray, sfreq: float, joins: Sequence[int], lc: StepConfigs, kernels: Sequence[int]
):
    """Rail / flat spans and valid spans (pieces shorter than the longest FIR kernel become BAD_short)."""
    seg_mod = _ap().seg
    nmin = seg_mod.min_valid_samples(lc.segments, list(kernels), sfreq)
    return seg_mod.build_segment_info(eeg_v, sfreq, joins, lc.segments, min_valid_samples=nmin)


def with_extra_bad(seg, spans: np.ndarray, reason: str):
    """A copy of ``seg`` with ``spans`` added as unusable spans (valid spans re-derived)."""
    sm = _ap().seg
    sp = np.asarray(spans, np.int64).reshape(-1, 2)
    un = np.concatenate([seg.unusable_spans, sp]) if len(sp) else seg.unusable_spans
    reasons = tuple(seg.unusable_reasons) + (reason,) * len(sp)
    return sm.SegmentInfo.build(
        seg.sfreq,
        seg.n_times,
        seg.joins,
        seg.rail_spans,
        seg.flat_spans,
        un,
        reasons,
        seg.first_samp,
        seg.meta,
        short_spans=seg.short_spans,
    )


def annotate(raw, seg) -> None:
    _ap().seg.annotate_raw(raw, seg)


def zero_bad(raw, seg, picks=None) -> None:
    _ap().seg.zero_spans(raw, seg.bad_spans, picks)


def filter_per_span(raw, seg, spec, picks) -> None:
    _ap().filt.filter_raw_per_segment(raw, seg, spec, picks)


def notch_per_span(raw, seg, freqs, picks) -> None:
    _ap().filt.notch_raw_per_segment(raw, seg, list(freqs), picks)


def fir_taps(spec, sfreq: float) -> int:
    return int(len(_ap().filt.fir_kernel(spec, sfreq)))


def bad_channels(
    hp_raw,
    raw_eeg_v: np.ndarray,
    seg,
    names: Sequence[str],
    lc: StepConfigs,
    seed: int,
    lowpass_hz: float | None = None,
):
    """PyPREP robust-reference loop on the 1-100 Hz copy (decimated to 256 Hz), optionally low-passed per valid
    span at ``lowpass_hz`` first; returns the detection result. ``seg`` may carry extra detection-only
    exclusions; the filter runs on the spans of the recording's own segments (via the annotations of ``hp_raw``).
    """
    bc = _ap().bad
    sat = bc.saturation_fractions(raw_eeg_v, seg, names, lc.bad_channels.saturation_threshold_v)
    src = hp_raw
    if lowpass_hz:
        src = hp_raw.copy()
        spec = _ap().cfg.FilterSpec(l_freq=None, h_freq=float(lowpass_hz))
        src.filter(
            picks=list(names),
            skip_by_annotation=list(_ap().seg.SKIP_BY_ANNOTATION),
            verbose="error",
            **spec.mne_kwargs(),
        )
    data = bc.detection_data(src, seg, lc.bad_channels, names)
    sf = hp_raw.info["sfreq"] / lc.bad_channels.decimate
    return bc.find_bad_channels(data, sf, names, lc.bad_channels, seed=seed, saturation_frac=sat)


def run_ica(hp_raw, seg, good: Sequence[str], lc: StepConfigs, seed: int, montage: str):
    """Picard + ICLabel followed by the posterior-alpha veto; returns (ICAResult, selection dict)."""
    res = _ap().ica.run_ica(hp_raw, seg, list(good), lc.ica, seed, montage=montage, n_threads=1)
    sel = _ap().sel.run_selection(hp_raw, res.fit, res.ica, res.excluded, lc.selection)
    return res, sel


def reselect(
    hp_raw, seg, good: Sequence[str], ica, probs: np.ndarray, lc: StepConfigs, montage: str, keep_eye=None
) -> dict:
    """ICLabel rule on stored probabilities, eye validation (``keep_eye(component) -> bool`` decides argmax-eye
    components below the strict threshold), then the posterior-alpha veto on the rebuilt fit data."""
    excl, _ = _ap().ica.select_exclusions(probs, lc.ica.iclabel_rule, lc.ica.iclabel_exclude)
    eye = _ap().cfg.ICLABEL_CLASSES.index("eye blink")
    rejected: list[int] = []
    if keep_eye is not None:
        top = np.argmax(probs, axis=1)
        rejected = [int(i) for i in excl if top[i] == eye and not keep_eye(int(i))]
        excl = [i for i in excl if i not in rejected]
    fit = _ap().ica.ica_fit_data(hp_raw, seg, list(good), lc.ica, montage)
    sel = _ap().sel.run_selection(hp_raw, fit, ica, excl, lc.selection)
    out = {k: sel[k] for k in ("iclabel_excluded", "vetoed", "excluded")}
    out["eye_not_validated"] = rejected
    return out


def iclabel_index(name: str) -> int:
    """Column of ICLabel class ``name`` in the probability matrix."""
    return int(_ap().cfg.ICLABEL_CLASSES.index(name))


def component_sources(ica, data: np.ndarray) -> np.ndarray:
    """ICA sources of ``data`` (ICA channel order, volts) exactly as MNE computes them."""
    return _ap().sel.component_sources(ica, data)


def clean_from_parts(raw, ica, exclude: Sequence[int]):
    """Apply ICA (``exclude``), interpolate ``info['bads']``, robust average reference."""
    return _ap().io.clean_from_parts(raw, ica, apply_ica=True, interpolate=True, exclude=list(exclude))


def line_noise_db(data: np.ndarray, sfreq: float, freq: float) -> dict:
    return _ap().qc.line_noise_db(data, sfreq, freq)


def valid_concat(data: np.ndarray, seg, decimate: int = 1) -> np.ndarray:
    return _ap().bad.valid_concat(data, seg, decimate)
