"""ICA re-selection on the fitted ICA solution: posterior-alpha veto and fit-rejected spans (no refit; Additional
file 1, Supplementary Methods S1: muscle components with a posterior alpha peak are spared).

1. Fit data. :func:`~preprocessing.ica.ica_fit_data` rebuilds the data the ICA was fitted on (the 1-100 Hz
   per-span copy of the good EEG channels, average-referenced, valid spans only, 1-s chunks above
   ``ica.fit_reject_ptp_uv`` left out). :func:`check_fit_reproduction` compares it with the fitted solution: the
   sample count must equal ``ica.n_samples_`` and the mean of the pre-whitened data must equal ``ica.pca_mean_``
   (relative difference <= 1e-9). Otherwise the step raises: the ICA was not fitted on this data.
2. Posterior-alpha veto. For every component:

   - ``alpha_peak_db``: the source time course (``ica._transform`` of the average-referenced good channels) is
     cut into the contiguous runs that went into the fit (512 Hz, before any decimation). Welch spectra (Hann,
     ``welch_window_s`` windows, ``welch_overlap``, mean over windows) are averaged over runs, weighted by their
     number of windows; runs shorter than one window are skipped. A straight line is fitted by least squares to
     ``log10(power)`` against ``log10(f)`` over ``aperiodic_fit_hz`` with the ``band_hz`` bins left out (the
     aperiodic fit). The residual is ``10 * (log10(power) - fit)`` dB. ``alpha_peak_db`` is the largest residual
     among the *local maxima* of the residual (a bin at least as high as both neighbours on the full frequency
     grid) whose frequency lies in ``band_hz``; NaN if there is none (e.g. a residual that only rises towards a
     band edge). ``alpha_peak_hz`` is its frequency.
   - ``posterior_share``: with ``a`` the component's pattern (its column of ``ica.get_components()`` over the
     ICA's good channels), ``sum(a[c]**2 for c in posterior_channels) / sum(a**2)``.
   - A component of the ICLabel exclusion set is vetoed (kept in the data) if ``alpha_peak_db >=
     min_peak_db`` and ``posterior_share >= min_posterior_share``. Final exclusions = ICLabel set minus vetoed.

3. Fit-rejected spans: the chunks left out of the fit, merged, as half-open sample spans (ICA back-projection
   can amplify such gross transients).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from preprocessing.step_config import ConfigError

FORMAT = "ica_selection"
FORMAT_VERSION = 1
FIT_CHECK_RTOL = 1e-9


@dataclass(frozen=True)
class AlphaVetoConfig:
    """``ica_selection.alpha_veto`` (see the module docstring)."""

    enabled: bool
    band_hz: tuple[float, float]
    min_peak_db: float
    aperiodic_fit_hz: tuple[float, float]
    welch_window_s: float
    welch_overlap: float
    min_posterior_share: float
    posterior_channels: tuple[str, ...]


@dataclass(frozen=True)
class SelectionConfig:
    """``ica_selection`` section of ``configs/preprocessing.yaml``."""

    alpha_veto: AlphaVetoConfig
    mark_fit_rejected: bool
    fit_rejected_description: str

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_POSTERIOR = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "Pz", "PO3", "PO4", "PO7", "PO8",
                     "POz", "O1", "O2", "Oz")


def parse_selection_config(raw: Mapping[str, Any] | None, where: str = "preprocessing.yaml:ica_selection"
                           ) -> SelectionConfig:
    """Validate the ``ica_selection`` mapping (strict keys; an empty mapping gives the veto disabled)."""
    raw = dict(raw or {})
    allowed = {"alpha_veto", "mark_fit_rejected", "fit_rejected_description"}
    bad = set(raw) - allowed
    if bad:
        raise ConfigError(f"{where}: unknown keys {sorted(bad)}; allowed {sorted(allowed)}")
    av = dict(raw.get("alpha_veto") or {})
    akeys = {"enabled", "band_hz", "min_peak_db", "aperiodic_fit_hz", "welch_window_s", "welch_overlap",
             "min_posterior_share", "posterior_channels"}
    bad = set(av) - akeys
    if bad:
        raise ConfigError(f"{where}.alpha_veto: unknown keys {sorted(bad)}; allowed {sorted(akeys)}")

    def num(v, key, lo=None, hi=None):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{where}.{key}: expected a number, got {v!r}")
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise ConfigError(f"{where}.{key}: {v} outside [{lo}, {hi}]")
        return float(v)

    def pair(v, key):
        if not isinstance(v, list) or len(v) != 2:
            raise ConfigError(f"{where}.{key}: expected [low, high]")
        a, b = num(v[0], key, 0), num(v[1], key, 0)
        if not a < b:
            raise ConfigError(f"{where}.{key}: need low < high")
        return (a, b)

    enabled = av.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError(f"{where}.alpha_veto.enabled: expected true or false")
    band = pair(av.get("band_hz", [7.0, 13.0]), "alpha_veto.band_hz")
    fitr = pair(av.get("aperiodic_fit_hz", [2.0, 30.0]), "alpha_veto.aperiodic_fit_hz")
    if not (fitr[0] < band[0] and band[1] < fitr[1]):
        raise ConfigError(f"{where}.alpha_veto: band_hz must lie strictly inside aperiodic_fit_hz")
    post = av.get("posterior_channels", list(DEFAULT_POSTERIOR))
    if not isinstance(post, list) or not post or not all(isinstance(c, str) and c for c in post):
        raise ConfigError(f"{where}.alpha_veto.posterior_channels: expected a non-empty list of channel names")
    mark = raw.get("mark_fit_rejected", False)
    if not isinstance(mark, bool):
        raise ConfigError(f"{where}.mark_fit_rejected: expected true or false")
    desc = raw.get("fit_rejected_description", "BAD_icafit")
    if not isinstance(desc, str) or not desc.upper().startswith("BAD"):
        raise ConfigError(f"{where}.fit_rejected_description must start with 'BAD' (MNE / feature rule)")
    return SelectionConfig(
        alpha_veto=AlphaVetoConfig(
            enabled=enabled, band_hz=band, min_peak_db=num(av.get("min_peak_db", 3.0), "alpha_veto.min_peak_db"),
            aperiodic_fit_hz=fitr,
            welch_window_s=num(av.get("welch_window_s", 2.0), "alpha_veto.welch_window_s", 0.25),
            welch_overlap=num(av.get("welch_overlap", 0.5), "alpha_veto.welch_overlap", 0.0, 0.9),
            min_posterior_share=num(av.get("min_posterior_share", 0.5), "alpha_veto.min_posterior_share", 0.0, 1.0),
            posterior_channels=tuple(post)),
        mark_fit_rejected=mark, fit_rejected_description=desc)


def component_sources(ica, data: np.ndarray) -> np.ndarray:
    """Sources of ``data`` (``(n_ica_channels, n)`` volts, the ICA's channel order) exactly as MNE computes them."""
    return ica._transform(np.array(data, dtype=np.float64, copy=True))


def run_spectra(ica, hp_data: np.ndarray, runs: np.ndarray, sfreq: float, window_s: float, overlap: float
                ) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean Welch spectrum of every component over the contiguous ``runs`` of ``hp_data``.

    ``hp_data`` holds the ICA's channels (rows in ``ica.ch_names`` order) of the 1-100 Hz copy; each run is
    average-referenced over these channels, as the fit data were. Returns (freqs, psd ``(n_comp, n_freq)``, number
    of Welch windows used).
    """
    from scipy.signal import welch

    nper = int(round(window_s * sfreq))
    nover = int(round(overlap * nper))
    step = nper - nover
    acc = None
    n_win = 0
    freqs = None
    for a, b in np.asarray(runs, np.int64).reshape(-1, 2):
        if b - a < nper:
            continue
        x = hp_data[:, a:b]
        x = x - x.mean(axis=0, keepdims=True)
        s = component_sources(ica, x)
        k = 1 + (b - a - nper) // step
        freqs, p = welch(s, fs=sfreq, window="hann", nperseg=nper, noverlap=nover, detrend="constant",
                         average="mean", axis=-1)
        acc = p * k if acc is None else acc + p * k
        n_win += k
    if acc is None:
        return np.zeros(0), np.zeros((int(ica.n_components_), 0)), 0
    return freqs, acc / n_win, n_win


def alpha_peak(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float], fit_range: tuple[float, float]
               ) -> tuple[np.ndarray, np.ndarray]:
    """Per row of ``psd``: (peak height in dB above the log-log aperiodic fit, peak frequency) - module docstring."""
    psd = np.atleast_2d(np.asarray(psd, float))
    n = psd.shape[0]
    db = np.full(n, np.nan)
    hz = np.full(n, np.nan)
    if freqs.size == 0:
        return db, hz
    in_fit = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    use = in_fit & ~in_band
    lf = np.log10(freqs[in_fit])
    xs = np.log10(freqs[use])
    idx_fit = np.flatnonzero(in_fit)
    for i in range(n):
        p = psd[i]
        if not np.all(np.isfinite(p[in_fit])) or np.any(p[in_fit] <= 0):
            continue
        ly = np.log10(p[use])
        slope, icpt = np.polyfit(xs, ly, 1)
        r = 10.0 * (np.log10(p[in_fit]) - (icpt + slope * lf))
        best = -np.inf
        best_f = np.nan
        for j in range(1, r.size - 1):                       # local maxima on the fit grid
            if not in_band[idx_fit[j]]:
                continue
            if r[j] >= r[j - 1] and r[j] >= r[j + 1] and r[j] > best:
                best, best_f = r[j], freqs[idx_fit[j]]
        if np.isfinite(best):
            db[i], hz[i] = best, best_f
    return db, hz


def posterior_share(patterns: np.ndarray, ch_names: Sequence[str], posterior: Sequence[str]) -> np.ndarray:
    """Share of the squared pattern weights on the ``posterior`` channels, per component (columns of ``patterns``)."""
    pat = np.asarray(patterns, float)
    post = np.array([c in set(posterior) for c in ch_names], bool)
    tot = (pat ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (pat[post] ** 2).sum(axis=0) / tot, np.nan)


def check_fit_reproduction(fit, ica) -> dict:
    """Compare rebuilt fit data with the stored ICA (``n_samples_`` and ``pca_mean_``); see the module docstring."""
    x = fit.raw.get_data(picks=list(ica.ch_names))
    out: dict[str, Any] = {"n_fit_samples": int(x.shape[1]), "ica_n_samples": int(getattr(ica, "n_samples_", -1)),
                           "n_chunks": int(fit.n_chunks), "n_rejected_chunks": int(fit.n_rejected_chunks)}
    pw = np.asarray(ica.pre_whitener_, float).reshape(-1, 1)
    mean = (x / pw).mean(axis=1)
    ref = np.asarray(ica.pca_mean_, float)
    scale = float(np.max(np.abs(ref))) or 1.0
    out["pca_mean_max_abs_diff_rel"] = float(np.max(np.abs(mean - ref)) / scale) if ref.shape == mean.shape \
        else float("inf")
    out["ok"] = bool(out["n_fit_samples"] == out["ica_n_samples"]
                     and out["pca_mean_max_abs_diff_rel"] <= FIT_CHECK_RTOL)
    return out


class FitReproductionError(RuntimeError):
    """The rebuilt ICA fit data do not match the stored ICA solution."""


def run_selection(hp_raw, fit, ica, iclabel_excluded: Sequence[int], cfg: SelectionConfig) -> dict:
    """Re-selection record for one recording (JSON-ready; see the module docstring).

    Parameters
    ----------
    hp_raw
        The 1-100 Hz per-span copy (EEG channels) the ICA fit data were built from (512 Hz).
    fit
        :class:`~preprocessing.ica.ICAFitData` built from ``hp_raw`` (fresh or the one of the fit).
    ica
        The fitted (stored) ICA.
    iclabel_excluded
        The ICLabel-rule exclusions (``ica.exclude`` of the stored solution).
    cfg
        Parsed ``ica_selection`` config.

    Raises
    ------
    FitReproductionError
        If ``fit`` is not the data ``ica`` was fitted on.
    """
    t0 = time.perf_counter()
    chk = check_fit_reproduction(fit, ica)
    if not chk["ok"]:
        raise FitReproductionError(f"rebuilt ICA fit data do not match the stored ICA: {chk}")
    av = cfg.alpha_veto
    n = int(ica.n_components_)
    sf = float(hp_raw.info["sfreq"])
    names = list(ica.ch_names)
    idx = [hp_raw.ch_names.index(c) for c in names]
    freqs, psd, n_win = run_spectra(ica, hp_raw._data[idx], fit.kept_spans, sf, av.welch_window_s, av.welch_overlap)
    peak_db, peak_hz = alpha_peak(freqs, psd, av.band_hz, av.aperiodic_fit_hz)
    share = posterior_share(ica.get_components(), names, av.posterior_channels)
    eligible = (peak_db >= av.min_peak_db) & (share >= av.min_posterior_share)
    excl0 = sorted(int(i) for i in iclabel_excluded)
    vetoed = [i for i in excl0 if av.enabled and bool(eligible[i])]
    final = [i for i in excl0 if i not in vetoed]
    valid_n = int(fit.n_valid_samples)
    spans = np.asarray(fit.rejected_spans, np.int64).reshape(-1, 2)
    rej_n = int((spans[:, 1] - spans[:, 0]).sum()) if len(spans) else 0

    def rnd(v):
        return [None if not np.isfinite(x) else round(float(x), 4) for x in v]

    return {
        "format": FORMAT, "version": FORMAT_VERSION, "status": "ok",
        "config": cfg.to_dict(),
        "fit_check": chk,
        "n_components": n,
        "iclabel_excluded": excl0,
        "vetoed": vetoed,
        "excluded": final,
        "components": {"alpha_peak_db": rnd(peak_db), "alpha_peak_hz": rnd(peak_hz),
                       "posterior_share": rnd(share), "veto_eligible": [bool(x) for x in eligible]},
        "spectra": {"n_welch_windows": int(n_win), "freq_resolution_hz": float(freqs[1] - freqs[0])
                    if freqs.size > 1 else None},
        "icafit": {"description": cfg.fit_rejected_description, "mark": bool(cfg.mark_fit_rejected),
                   "spans": spans.tolist(), "n_spans": int(len(spans)), "samples": rej_n,
                   "seconds": rej_n / sf, "fraction_of_valid": rej_n / valid_n if valid_n else 0.0},
        "seconds": round(time.perf_counter() - t0, 2),
    }
