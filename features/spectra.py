"""Per-recording spectral pass: every block's Welch spectrum and band cross-spectral quantities, cached as one
``.npz`` per recording (Methods, Preprocessing and features; Additional file 1, Supplementary Methods S1).

Band-limited covariance, debiased wPLI and coherence all come from the FFT of the same non-overlapping 2-s Hann
segments. For a band B with one-sided cross-spectra S_xy(f):

- covariance: sample covariance of the band-limited reconstruction (inverse FFT of the in-band bins) of the same
  segments, shrunk towards the trace-scaled identity with gamma = max(Ledoit-Wolf gamma, shrinkage floor 0.05),
- debiased wPLI^2 (Vinck et al. 2011) from the per-segment imaginary cross-spectra, averaged over the bins of B,
- coherence |<S_xy>|^2 / (<S_xx><S_yy>), averaged over the bins of B.

The 35-45 Hz range (ASSR stimulation) is excluded in every paradigm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config.loader import Config
from features.signals import CleanRecording, welch_from_segments
from utils.logging_utils import get_logger

log = get_logger(__name__)

BANDS_ORDER = ("theta", "alpha", "beta", "broadband")


def bank_tag(variant: str = "primary", block_mode: str = "primary") -> str:
    """Folder name of a feature bank: the preprocessing variant, plus ``__altblocks`` for the alternative blocks or
    ``__standing`` for the bank of the standing motor runs (cross-system identification)."""
    return {"primary": variant, "alternative": f"{variant}__altblocks", "standing": f"{variant}__standing"}[
        block_mode
    ]


def spectra_path(
    cfg: Config, child: str, paradigm: str, variant: str = "primary", block_mode: str = "primary"
) -> Path:
    return (
        cfg.path("derivatives")
        / "features"
        / bank_tag(variant, block_mode)
        / "spectra"
        / f"{child}_task-{paradigm}.npz"
    )


def band_bins(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return np.flatnonzero((freqs >= band[0]) & (freqs <= band[1]))


def shrunk_covariance(X: np.ndarray, floor: float) -> tuple[np.ndarray, float]:
    """Covariance of ``X`` (ch x samples, zero-mean) shrunk with gamma = max(Ledoit-Wolf, floor)."""
    from sklearn.covariance import ledoit_wolf_shrinkage

    S = X @ X.T / X.shape[1]
    g = max(float(ledoit_wolf_shrinkage(X.T, assume_centered=True)), floor)
    mu = np.trace(S) / S.shape[0]
    return (1 - g) * S + g * mu * np.eye(S.shape[0]), g


def cross_spectral(
    S: np.ndarray,
    sf: float,
    bands: dict[str, tuple[float, float]],
    floor: float = 0.05,
    connectivity: bool = True,
) -> dict[str, np.ndarray]:
    """Band covariance, debiased wPLI and coherence from segments ``(n, ch, t)`` (see the module docstring)."""
    n, nch, w = S.shape
    win = np.hanning(w)
    scale = 1.0 / (sf * (win**2).sum())
    F = np.fft.rfft((S - S.mean(axis=2, keepdims=True)) * win, axis=2)  # (n, ch, f)
    freqs = np.fft.rfftfreq(w, 1 / sf)
    iu = np.triu_indices(nch, 1)
    out: dict[str, np.ndarray] = {}
    for name, band in bands.items():
        idx = band_bins(freqs, band)
        mask = np.zeros(freqs.size)
        mask[idx] = 1.0
        Xb = np.fft.irfft(F * mask, n=w, axis=2)  # band-limited segments
        cov, g = shrunk_covariance(Xb.transpose(1, 0, 2).reshape(nch, -1), floor)
        out[f"cov_{name}"] = cov
        out[f"shrinkage_{name}"] = np.array(g)
        if not connectivity:
            continue
        wpli = np.zeros(iu[0].size)
        coh = np.zeros(iu[0].size)
        for k in idx:
            X = F[:, :, k]
            Sxy = X[:, :, None] * np.conj(X[:, None, :]) * scale * 2  # (n, ch, ch), one-sided
            mean = Sxy.mean(axis=0)
            im = Sxy.imag[:, iu[0], iu[1]]
            s_im, s_abs, s_sq = im.sum(0), np.abs(im).sum(0), (im**2).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                wpli += np.where(s_abs**2 - s_sq > 0, (s_im**2 - s_sq) / (s_abs**2 - s_sq), 0.0)
            pxx = mean.real.diagonal()
            coh += np.abs(mean[iu]) ** 2 / (pxx[iu[0]] * pxx[iu[1]])
        out[f"wpli_{name}"] = wpli / max(idx.size, 1)
        out[f"coh_{name}"] = coh / max(idx.size, 1)
    return out


def recording_pass(
    cfg: Config,
    child: str,
    paradigm: str,
    variant: str = "primary",
    force: bool = False,
    block_mode: str = "primary",
) -> Path | None:
    """Compute and cache the per-block spectra of one recording; returns the cache path (``None`` if unusable)."""
    out = spectra_path(cfg, child, paradigm, variant, block_mode)
    r = CleanRecording(cfg, child, paradigm, variant, block_mode)
    if not r.paths["qc"].is_file() or not r.usable:
        return None
    # the cache is valid only for the cleaned recording it was computed from
    prep = str(r.qc.get("preprocessing_hash", ""))
    if out.is_file() and not force:
        with np.load(out) as z:
            if "preprocessing_hash" in z.files and str(z["preprocessing_hash"]) == prep:
                return out
        log.info("spectra cache %s is stale (preprocessing changed); recomputed", out.name)
    fc = cfg.features
    bands = {**{k: tuple(v) for k, v in fc.bands.items()}, "broadband": tuple(fc.broadband)}
    exg = [c for c in r.qc.get("exg_connected", []) if c in r.exg_names]
    keep_exg = exg if all(c in exg for c in ("EXG1", "EXG2", "EXG3", "EXG4")) else []
    keep_exg = [c for c in ("EXG1", "EXG2", "EXG3", "EXG4") if c in keep_exg]
    rows: dict[str, list] = {"block": [], "n_psd_segments": [], "n_cs_segments": [], "clean_s": []}
    psd, exg_psd, cs, exg_cov = [], [], {}, []
    freqs = None
    for _, b in r.blocks.sort_values("block").iterrows():
        S1 = r.segments(
            b,
            None,
            seg_s=fc.spectral.welch_window_s,
            step_s=fc.spectral.welch_window_s * (1 - fc.spectral.welch_overlap),
        )
        S2 = r.segments(b, None, seg_s=fc.wpli.segment_s, step_s=fc.wpli.segment_s)
        if len(S2) < 5:  # fewer than 10 s of clean data: the block gives no features
            continue
        f, P = welch_from_segments(S1, r.sfreq)
        sel = (f >= 0.5) & (f <= 45.0)
        freqs = f[sel]
        psd.append(P[:, sel].astype(np.float32))
        floor = fc.covariance.shrinkage_floor
        for k, v in cross_spectral(S2, r.sfreq, bands, floor).items():
            if k.startswith(("wpli_broadband", "coh_broadband")):
                continue
            cs.setdefault(k, []).append(np.asarray(v, np.float32))
        if keep_exg:
            E1 = r.segments(
                b,
                keep_exg,
                seg_s=fc.spectral.welch_window_s,
                step_s=fc.spectral.welch_window_s * (1 - fc.spectral.welch_overlap),
            )
            E1 = E1 - E1.mean(axis=1, keepdims=True)  # EXG referenced to their own mean
            _, PE = welch_from_segments(E1, r.sfreq)
            exg_psd.append(PE[:, sel].astype(np.float32))
            E2 = r.segments(b, keep_exg, seg_s=fc.wpli.segment_s, step_s=fc.wpli.segment_s)
            E2 = E2 - E2.mean(axis=1, keepdims=True)
            exg_cov.append(
                cross_spectral(E2, r.sfreq, {"broadband": tuple(fc.broadband)}, floor, connectivity=False)[
                    "cov_broadband"
                ].astype(np.float32)
            )
        rows["block"].append(int(b["block"]))
        rows["n_psd_segments"].append(len(S1))
        rows["n_cs_segments"].append(len(S2))
        rows["clean_s"].append(float(b.get("clean_s", np.nan)))
    if not psd:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.stack(v) for k, v in cs.items()}
    if exg_psd:
        arrays["exg_psd"] = np.stack(exg_psd)
        arrays["exg_cov"] = np.stack(exg_cov)
    np.savez_compressed(
        out,
        freqs=freqs,
        psd=np.stack(psd),
        eeg_names=np.array(r.eeg_names),
        exg_names=np.array(keep_exg),
        preprocessing_hash=np.array(prep),
        **{k: np.array(v) for k, v in rows.items()},
        **arrays,
    )
    return out
