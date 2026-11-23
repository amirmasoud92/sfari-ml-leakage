"""Array functions used by the preprocessors: span and mask algebra, cropping bounds, blink spans, whole-head
artefact windows, connection check of the external channels, the cleanest calibration windows of artefact
subspace reconstruction, the repaired fraction, residual peak-to-peak windows and the artefact-free trial rule.

All sample spans are ``(k, 2)`` int arrays of half-open ``[start, stop)`` indices.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def spans_to_mask(spans: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in np.asarray(spans, np.int64).reshape(-1, 2):
        m[max(a, 0) : min(b, n)] = True
    return m


def mask_to_spans(mask: np.ndarray) -> np.ndarray:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1).astype(np.int64)


def crop_bounds(onsets_s: Sequence[float], n_times: int, sfreq: float, margin_s: float) -> tuple[int, int]:
    """Sample bounds ``[first event - margin, last event + margin)`` clipped to the run (whole run if no events)."""
    on = np.asarray(onsets_s, float)
    on = on[np.isfinite(on)]
    if on.size == 0:
        return 0, int(n_times)
    a = max(int(np.floor((on.min() - margin_s) * sfreq)), 0)
    b = min(int(np.ceil((on.max() + margin_s) * sfreq)), int(n_times))
    return a, b


def window_starts(valid_spans: np.ndarray, win: int) -> np.ndarray:
    """Start samples of non-overlapping ``win``-sample windows lying entirely inside one valid span."""
    out = [
        np.arange(a, b - win + 1, win)
        for a, b in np.asarray(valid_spans, np.int64).reshape(-1, 2)
        if b - a >= win
    ]
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def cleanest_windows(X: np.ndarray, valid_spans: np.ndarray, win: int, n_windows: int) -> np.ndarray:
    """Starts of the ``n_windows`` cleanest windows, sorted by time.

    Each window's score is the maximum over channels of the robust z (median / 1.4826 MAD across windows) of its
    log RMS. The lowest scores are the cleanest (ASR calibration).
    """
    starts = window_starts(valid_spans, win)
    if starts.size == 0:
        return starts
    rms = np.stack([np.sqrt(np.mean(X[:, s : s + win] ** 2, axis=1)) for s in starts], axis=1)  # (ch, w)
    lr = np.log(np.maximum(rms, 1e-30))
    med = np.median(lr, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(lr - med), axis=1, keepdims=True)
    z = (lr - med) / np.where(mad > 0, mad, 1.0)
    score = z.max(axis=0)
    keep = np.argsort(score, kind="stable")[: min(n_windows, starts.size)]
    return np.sort(starts[keep])


def repaired_fraction(before: np.ndarray, after: np.ndarray, valid: np.ndarray, rel: float) -> float:
    """Fraction of valid samples where ``||after - before|| > rel * ||before||`` (norms over channels)."""
    if not valid.any():
        return 0.0
    d = np.linalg.norm(after[:, valid] - before[:, valid], axis=0)
    n = np.linalg.norm(before[:, valid], axis=0)
    return float(np.mean(d > rel * np.maximum(n, 1e-30)))


def residual_spans(X: np.ndarray, valid_spans: np.ndarray, win: int, ptp_v: float) -> np.ndarray:
    """``win``-sample windows (tiling each valid span; a short tail is its own window) with any channel's
    peak-to-peak above ``ptp_v``."""
    bad = []
    for a, b in np.asarray(valid_spans, np.int64).reshape(-1, 2):
        for s in range(a, b, win):
            e = min(s + win, b)
            if e - s > 1 and np.ptp(X[:, s:e], axis=1).max() > ptp_v:
                bad.append((s, e))
    if not bad:
        return np.zeros((0, 2), np.int64)
    m = spans_to_mask(np.array(bad), int(np.asarray(valid_spans).max()))
    return mask_to_spans(m)


def trial_ok(
    samples: np.ndarray,
    X: np.ndarray,
    good_mask: np.ndarray,
    sfreq: float,
    tmin: float,
    tmax: float,
    ptp_v: float,
) -> np.ndarray:
    """Whether each epoch ``[s + tmin, s + tmax]`` lies wholly in good samples and stays below ``ptp_v``."""
    s = np.asarray(samples, np.int64)
    a = s + int(round(tmin * sfreq))
    b = s + int(round(tmax * sfreq)) + 1
    n = X.shape[1]
    ok = (a >= 0) & (b <= n)
    out = np.zeros(s.size, bool)
    for i in np.flatnonzero(ok):
        if good_mask[a[i] : b[i]].all():
            out[i] = np.ptp(X[:, a[i] : b[i]], axis=1).max() <= ptp_v
    return out


def blink_spans(proxy: np.ndarray, valid: np.ndarray, sfreq: float, z: float, pad_s: float) -> np.ndarray:
    """Spans where |proxy| (50-ms moving average) exceeds median + z * 1.4826 MAD over valid samples, padded by
    ``pad_s`` on both sides."""
    w = max(int(round(0.05 * sfreq)), 1)
    s = np.convolve(np.abs(proxy), np.ones(w) / w, mode="same")
    v = s[valid]
    if v.size == 0:
        return np.zeros((0, 2), np.int64)
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    hit = (s > med + z * max(mad, 1e-30)) & valid
    if not hit.any():
        return np.zeros((0, 2), np.int64)
    pad = int(round(pad_s * sfreq))
    sp = mask_to_spans(hit)
    sp[:, 0] = np.maximum(sp[:, 0] - pad, 0)
    sp[:, 1] = np.minimum(sp[:, 1] + pad, proxy.size)
    return mask_to_spans(spans_to_mask(sp, proxy.size))


def global_artefact_windows(
    X: np.ndarray, valid: np.ndarray, sfreq: float, window_s: float, z: float
) -> tuple[np.ndarray, np.ndarray]:
    """Whole-head artefact windows: per non-overlapping ``window_s`` window inside valid samples, each
    channel's log RMS is robust-z-scored across windows (median, 1.4826 MAD); the window score is the median of
    these z over channels, so it is high only when most channels are disturbed at once (movement), not when one
    channel is noisy (a bad channel must stay detectable). Returns ``(spans, scores)`` of the windows with score
    > ``z``, highest score first."""
    w = int(round(window_s * sfreq))
    n = X.shape[1]
    starts = np.arange(0, n - w + 1, w)
    c = np.concatenate([[0], np.cumsum(valid)])
    starts = starts[(c[starts + w] - c[starts]) == w]
    if starts.size < 10:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    rms = np.stack([np.sqrt(np.mean(X[:, s : s + w] ** 2, axis=1)) for s in starts], axis=1)
    lr = np.log(np.maximum(rms, 1e-30))
    med = np.median(lr, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(lr - med), axis=1, keepdims=True)
    score = np.median((lr - med) / np.where(mad > 0, mad, 1.0), axis=0)
    hit = score > z
    order = np.argsort(-score[hit])
    s = starts[hit][order]
    return np.stack([s, s + w], axis=1).astype(np.int64), score[hit][order]


def exg_connected(raw_v: np.ndarray, filt_v: np.ndarray, rail_v: float, min_sd_uv: float) -> np.ndarray:
    """Per EXG channel: median raw level above the rail and 1-40 Hz SD at least ``min_sd_uv``."""
    med = np.median(raw_v, axis=1)
    sd = np.std(filt_v, axis=1) * 1e6
    return (med > rail_v) & (sd >= min_sd_uv)
