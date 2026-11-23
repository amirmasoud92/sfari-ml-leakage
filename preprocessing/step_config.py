"""Configuration dataclasses of the preprocessing steps (filters, segments, bad-channel detection, ICA) and the
ICLabel class names. They are built from ``config/analysis.yaml`` by :func:`preprocessing.adapter.step_configs`.
"""

from __future__ import annotations

from dataclasses import dataclass

BAD_CHANNEL_CRITERIA = ("nan_flat", "deviation", "hf_noise", "correlation", "snr", "psd", "ransac")
ICLABEL_CLASSES = ("brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other")


class ConfigError(ValueError):
    """Raised for a missing key, a wrong type or an inconsistent value in a configuration."""


@dataclass(frozen=True)
class FilterSpec:
    """Zero-phase FIR band-pass parameters (passed to MNE ``filter``)."""

    l_freq: float | None
    h_freq: float | None
    l_trans_bandwidth: float | str = "auto"
    h_trans_bandwidth: float | str = "auto"
    filter_length: str | int = "auto"
    fir_window: str = "hamming"
    fir_design: str = "firwin"
    phase: str = "zero"
    pad: str = "reflect_limited"

    def mne_kwargs(self) -> dict:
        """Keyword arguments of :meth:`mne.io.Raw.filter` / :func:`mne.filter.filter_data`."""
        return {"l_freq": self.l_freq, "h_freq": self.h_freq, "l_trans_bandwidth": self.l_trans_bandwidth,
                "h_trans_bandwidth": self.h_trans_bandwidth, "filter_length": self.filter_length,
                "fir_window": self.fir_window, "fir_design": self.fir_design, "phase": self.phase,
                "method": "fir", "pad": self.pad}


@dataclass(frozen=True)
class SpanRule:
    """Rule turning a per-sample mask into padded, merged spans."""

    pad_s: float
    merge_gap_s: float
    min_duration_s: float


@dataclass(frozen=True)
class SegmentsConfig:
    """``segments`` section: join annotation, rail / flat span rules, minimum valid-span length."""

    join_annotation: str
    rail_threshold_v: float
    rail: SpanRule
    flat_max_abs_diff_uv: float
    flat: SpanRule
    zero_skip_spans: bool
    min_valid_span_s: float | str = 0.0     # "auto" = the longest per-span FIR kernel; 0 = no minimum


@dataclass(frozen=True)
class RansacConfig:
    n_samples: int
    sample_prop: float
    corr_thresh: float
    frac_bad: float
    corr_window_secs: float
    channel_wise: bool


@dataclass(frozen=True)
class BadChannelsConfig:
    method: str
    criteria: tuple[str, ...]
    deviation_threshold: float
    hf_noise_threshold: float
    correlation_secs: float
    correlation_threshold: float
    correlation_frac_bad: float
    ransac: RansacConfig
    saturation_threshold_v: float
    saturation_max_frac: float | None
    notch: bool
    decimate: int
    initial_reference: str
    max_iterations: int
    max_bad_fraction_warn: float
    manual_bads_from_known_issues: bool
    ransac_pass: str = "every"     # 'every' robust-reference pass, or only the 'final' one
    correlation_min_bad_windows: int = 1   # correlation / dropout also need at least this many bad windows


@dataclass(frozen=True)
class ICAConfig:
    enabled: bool
    method: str
    fit_params: dict
    max_iter: int
    n_components: str | int
    max_components: int | None
    min_samples_factor: float
    decimate: int
    fit_reject_ptp_uv: float | None
    fit_reject_chunk_s: float
    iclabel_rule: str
    iclabel_exclude: dict[str, float]
    apply_in_primary: bool
