"""QC metrics computed on valid spans only (no join, rail, flat or unusable sample), including the line-noise peak
recorded in the QC record."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from preprocessing.segments import SegmentInfo


def chunk_bounds(seg: SegmentInfo, chunk: int) -> np.ndarray:
    """Non-overlapping full chunks of ``chunk`` samples inside each valid span (``(k, 2)``)."""
    out = []
    for a, b in seg.valid_spans:
        for c in range(a, b - chunk + 1, chunk):
            out.append((c, c + chunk))
    return np.asarray(out, np.int64).reshape(-1, 2)


def chunk_ptp_fraction(data: np.ndarray, seg: SegmentInfo, chunk: int, thr_v: float) -> tuple[int, float]:
    """(number of chunks, fraction of chunks in which any row of ``data`` exceeds ``thr_v`` peak-to-peak)."""
    cb = chunk_bounds(seg, chunk)
    if len(cb) == 0 or data.shape[0] == 0:
        return int(len(cb)), float("nan")
    bad = np.array([np.ptp(data[:, a:b], axis=1).max() > thr_v for a, b in cb])
    return int(len(cb)), float(bad.mean())


def valid_sd_uv(data: np.ndarray, seg: SegmentInfo) -> np.ndarray:
    """Per-row SD (uV) over the valid samples (each span demeaned)."""
    n = 0
    ss = np.zeros(data.shape[0])
    for a, b in seg.valid_spans:
        x = data[:, a:b]
        ss += ((x - x.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        n += b - a
    return np.sqrt(ss / max(n - 1, 1)) * 1e6


def residual_metrics(raw, seg: SegmentInfo, roi: Sequence[str], thr_uv: float, chunk_s: float,
                     frontal: Sequence[str]) -> dict:
    """Artefact-load metrics of a (cleaned) Raw: chunk peak-to-peak exceedance, channel SDs, blink proxy."""
    import mne

    bads = set(raw.info["bads"])
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[]) if raw.ch_names[i] not in bads]
    roi_ok = [c for c in roi if c in eeg]
    chunk = int(round(chunk_s * raw.info["sfreq"]))
    idx = [raw.ch_names.index(c) for c in eeg]
    ridx = [raw.ch_names.index(c) for c in roi_ok]
    n_ch, frac_any = chunk_ptp_fraction(raw._data[idx], seg, chunk, thr_uv * 1e-6)
    _, frac_roi = chunk_ptp_fraction(raw._data[ridx], seg, chunk, thr_uv * 1e-6)
    sd = valid_sd_uv(raw._data[idx], seg) if idx else np.array([np.nan])
    fr = [c for c in frontal if c in eeg]
    fsd = float(valid_sd_uv(raw._data[[raw.ch_names.index(c) for c in fr]].mean(axis=0, keepdims=True), seg)[0]) \
        if fr else float("nan")
    return {"n_chunks": n_ch, "chunk_s": chunk_s, "threshold_uv": thr_uv,
            "frac_chunks_any_eeg_ptp_gt_thr": frac_any, "frac_chunks_roi_ptp_gt_thr": frac_roi,
            "n_roi_good": len(roi_ok), "median_channel_sd_uv": float(np.median(sd)),
            "p95_channel_sd_uv": float(np.percentile(sd, 95)), "frontal_proxy_sd_uv": fsd}


def line_noise_db(data: np.ndarray, sfreq: float, freq: float, flank: tuple[float, float] = (2.0, 5.0)) -> dict:
    """Per-channel peak at ``freq`` in dB over the mean of the flanking bins (Welch, 4-s Hann, median average)."""
    from scipy.signal import welch

    if data.shape[1] < int(4 * sfreq) or freq >= sfreq / 2:
        return {"freq": freq, "median_db": None, "max_db": None, "n_channels_gt_3db": None}
    f, p = welch(data, fs=sfreq, nperseg=int(4 * sfreq), average="median", axis=-1)
    k = int(np.argmin(np.abs(f - freq)))
    fl = (np.abs(f - freq) >= flank[0]) & (np.abs(f - freq) <= flank[1])
    db = 10 * np.log10(p[:, k] / p[:, fl].mean(axis=1))
    return {"freq": freq, "median_db": float(np.median(db)), "max_db": float(db.max()),
            "n_channels_gt_3db": int((db > 3).sum())}
