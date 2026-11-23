"""Typed, immutable configuration of the analysis.

``config/analysis.yaml`` holds every analysis parameter (paths, block definitions, preprocessing, features,
identification, variance components, factor model, prediction, encoder). It is parsed into immutable pydantic models
that reject unknown keys. A file may ``inherit`` another one and override some of its keys (``muscle_variant.yaml``
inherits ``analysis.yaml``). Relative paths in the ``paths`` section are resolved against the repository root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repository root


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


Band = tuple[float, float]


class ProjectCfg(_Frozen):
    name: str
    seed: int


class PathsCfg(_Frozen):
    data_root: Path
    derivatives: Path
    results: Path

    def resolve(self, key: str) -> Path:
        """Absolute path of ``key``; relative paths are taken from the repository root."""
        p = Path(getattr(self, key))
        return p if p.is_absolute() else PROJECT_ROOT / p


class DatasetCfg(_Frozen):
    name_contains: str
    expected_version_doi: str
    seated_paradigms: tuple[str, ...]
    motor_paradigm: str
    group_map: dict[str, str]
    phenotype_columns: tuple[str, ...]
    outcome_columns: tuple[str, ...]
    restart_min_fraction: float = Field(gt=0, lt=1)
    expected_trials: dict[str, int]
    trial_patterns: dict[str, str]
    response_pattern: str


class BlockRule(_Frozen):
    rule: Literal["marker", "trial_count", "separator", "category_chunks", "rest_marker", "equal_duration"]
    marker: str | None = None
    size: int | None = None
    category_regex: str | None = None
    window_s: float | None = None
    max_blocks: int | None = None

    @model_validator(mode="after")
    def _check(self) -> BlockRule:
        need = {
            "marker": "marker",
            "separator": "marker",
            "trial_count": "size",
            "equal_duration": "window_s",
            "rest_marker": "marker",
            "category_chunks": "category_regex",
        }
        if self.rule == "category_chunks" and self.size is None:
            raise ValueError("block rule 'category_chunks' needs 'size'")
        key = need.get(self.rule)
        if key is not None and getattr(self, key) is None:
            raise ValueError(f"block rule {self.rule!r} needs {key!r}")
        return self


class BlocksCfg(_Frozen):
    ASSR: BlockRule
    AVSRT: BlockRule
    Beepflash: BlockRule
    FAST: BlockRule
    IC: BlockRule
    Restingstate: BlockRule
    min_trials_per_block: int
    min_rest_block_s: float
    pad_s: float
    alternative: BlockRule

    def rule_for(self, paradigm: str) -> BlockRule:
        return getattr(self, paradigm)


class RansacCfg(_Frozen):
    n_samples: int
    sample_prop: float
    corr_thresh: float
    frac_bad: float
    corr_window_secs: float


class SaturationCfg(_Frozen):
    threshold_v: float
    max_frac: float


class ExcludeBlinksCfg(_Frozen):
    enabled: bool
    channels: tuple[str, ...]
    z: float
    pad_s: float
    max_fraction: float


class ExcludeGlobalCfg(_Frozen):
    enabled: bool
    window_s: float
    z: float
    pad_s: float


class BadChannelsCfg(_Frozen):
    criteria: tuple[str, ...]
    deviation_threshold: float
    correlation_secs: float
    correlation_threshold: float
    correlation_frac_bad: float
    correlation_min_bad_windows: int
    ransac: RansacCfg
    saturation: SaturationCfg
    detection_band: Band
    detection_lowpass_hz: float | None = None
    max_iterations: int
    max_bad_fraction: float
    exclude_blinks: ExcludeBlinksCfg
    exclude_global_artefacts: ExcludeGlobalCfg


class AlphaVetoCfg(_Frozen):
    band_hz: Band
    min_peak_db: float
    aperiodic_fit_hz: Band
    min_posterior_share: float
    posterior_channels: tuple[str, ...]


class EyeValidationCfg(_Frozen):
    strict_p: float
    blink_ratio: float
    heog_r: float


class IcaCfg(_Frozen):
    eye_validation: EyeValidationCfg
    method: str
    fit_params: dict[str, Any]
    max_iter: int
    fit_band: Band
    min_samples_factor: int
    fit_reject_ptp_uv: float
    iclabel_rule: Literal["argmax_and_threshold", "threshold"]
    iclabel_exclude: dict[str, float]
    alpha_veto: AlphaVetoCfg


class AsrCfg(_Frozen):
    enabled: bool
    cutoff: float
    calibration_s: float
    window_s: float
    repaired_threshold: float


class BandpassCfg(_Frozen):
    l_freq: float
    h_freq: float
    fir_design: str
    fir_window: str
    phase: str


class ExgUnconnectedCfg(_Frozen):
    rail_v: float
    max_sd_uv: float


class EpochsCfg(_Frozen):
    tmin: float
    tmax: float
    baseline: Band
    reject_ptp_uv: float


class MinimumRetainedCfg(_Frozen):
    rest_s: float
    evoked_trials: int


class SpanRuleCfg(_Frozen):
    pad_s: float
    merge_gap_s: float
    min_duration_s: float


class SegmentsCfg(_Frozen):
    rail_threshold_v: float
    rail: SpanRuleCfg
    flat_max_abs_diff_uv: float
    flat: SpanRuleCfg
    join_bit: int
    min_valid_span_s: float | Literal["auto"]


class ResidualCfg(_Frozen):
    window_s: float
    ptp_uv: float


class PreprocessingCfg(_Frozen):
    resample_sfreq: float
    bandpass: BandpassCfg
    line_freq: float
    montage: str
    exg_channels: tuple[str, ...]
    exg_unconnected: ExgUnconnectedCfg
    crop_margin_s: float
    segments: SegmentsCfg
    residual: ResidualCfg
    bad_channels: BadChannelsCfg
    ica: IcaCfg
    asr: AsrCfg
    epochs: EpochsCfg
    minimum_retained: MinimumRetainedCfg
    diagnostics_subjects: int


class SpecparamCfg(_Frozen):
    freq_range: Band
    peak_width_limits: Band
    max_n_peaks: int
    min_peak_height: float
    peak_threshold: float
    aperiodic_mode: Literal["auto", "fixed", "knee"]
    knee_rule: dict[str, float]
    min_r2: float


class SpectralCfg(_Frozen):
    welch_window_s: float
    welch_overlap: float
    fmin: float
    fmax: float
    logpow_bin_hz: float
    specparam: SpecparamCfg
    alpha_band: Band
    beta_band: Band


class CovarianceCfg(_Frozen):
    estimator: str
    shrinkage_floor: float
    reference: str


class WpliCfg(_Frozen):
    debiased: bool
    segment_s: float


class FeaturesCfg(_Frozen):
    exclude_band_hz: Band
    segment_s: float
    max_gap_s: float
    spectral: SpectralCfg
    bands: dict[str, Band]
    covariance: CovarianceCfg
    broadband: Band
    wpli: WpliCfg
    evoked: dict[str, Any]
    variability: dict[str, Any]

    @field_validator("bands")
    @classmethod
    def _bands_outside_exclusion(cls, v: dict[str, Band]) -> dict[str, Band]:
        for name, (lo, hi) in v.items():
            if lo >= hi:
                raise ValueError(f"band {name}: low >= high")
        return v

    @model_validator(mode="after")
    def _no_band_in_exclusion(self) -> FeaturesCfg:
        lo, hi = self.exclude_band_hz
        for name, (a, b) in {**self.bands, "broadband": self.broadband}.items():
            if b > lo and a < hi:
                raise ValueError(f"band {name} overlaps the excluded {lo}-{hi} Hz range")
        if self.spectral.fmax > lo:
            raise ValueError("spectral.fmax reaches into the excluded band")
        return self


class IdentificationCfg(_Frozen):
    standardise: Literal["zscore_within_paradigm", "none"]
    n_permutations: int
    split_half: str
    regress_covariates: tuple[str, ...]
    primary_family: str
    floor_bootstrap: int
    floor_signflip: int
    h2_bootstrap: int
    equivalence_bound_g: float


class VarianceCfg(_Frozen):
    estimator: str
    n_bootstrap: int
    fdr_alpha: float
    h1b_features: tuple[str, ...]
    h1b_roi: tuple[str, ...]


class MotorCfg(_Frozen):
    asr_cutoff: float
    walking_min_s: float
    min_block_s: float


class ReproductionCfg(_Frozen):
    rest_channels: tuple[str, ...]
    rest_block: Literal["first_valid", "all"]
    ordinal_codes: dict[str, int]


class LatentCfg(_Frozen):
    method: Literal["mofa", "concat_pca"]
    max_factors: int
    drop_factor_threshold: float
    active_r2: float
    active_r2_sensitivity: tuple[float, ...]
    stability_match_r: float
    max_failed_fraction: float
    view_pca_components: int
    n_bootstrap: int
    convergence_mode: str
    fallback: str


class PredictionCfg(_Frozen):
    outer_folds: int
    repeats: int
    inner_folds: int
    n_permutations: int
    permutation_repeats: int
    n_bootstrap: int
    logistic_c: tuple[float, ...]
    ridge_alpha: tuple[float, ...]
    targets: tuple[str, ...]
    covariates: tuple[str, ...]
    calibration_bins: int


class EncoderCfg(_Frozen):
    """Subject-contrastive encoder (Additional file 1, Supplementary Methods S3)."""

    segments_per_view: int
    embedding_dim: int
    projection_dim: int
    temperature: float
    steps: int
    batch_children: int
    lr: float
    weight_decay: float
    noise_sd: float
    repeats: int


class StatsCfg(_Frozen):
    ci_level: float
    h2_permutations: int


class RuntimeCfg(_Frozen):
    n_jobs: int
    threads_per_worker: int
    log_level: str


class Config(_Frozen):
    """The resolved configuration (see the module docstring)."""

    project: ProjectCfg
    paths: PathsCfg
    dataset: DatasetCfg
    blocks: BlocksCfg
    preprocessing: PreprocessingCfg
    features: FeaturesCfg
    identification: IdentificationCfg
    variance: VarianceCfg
    latent: LatentCfg
    prediction: PredictionCfg
    encoder: EncoderCfg
    stats: StatsCfg
    motor: MotorCfg
    reproduction: ReproductionCfg
    runtime: RuntimeCfg
    source: str = ""

    def hash(self) -> str:
        return config_hash(self)

    def path(self, key: str) -> Path:
        return self.paths.resolve(key)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("inherits", None)
    if parent is None:
        return raw
    return _merge(_read((path.parent / parent).resolve()), raw)


def load_config(path: str | Path | None = None) -> Config:
    """Load, merge and validate a config file (default: ``config/analysis.yaml``)."""
    p = Path(path) if path is not None else PROJECT_ROOT / "config" / "analysis.yaml"
    if not p.is_absolute():
        p = (PROJECT_ROOT / p) if not p.exists() else p.resolve()
    data = _read(p)
    return Config(**data, source=str(p))


def config_hash(cfg: Config, length: int = 12) -> str:
    """SHA-256 (first ``length`` hex digits) of the resolved config without ``runtime`` and ``source``."""
    payload = cfg.model_dump(mode="json", exclude={"runtime", "source"})
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:length]
