"""Zero-phase FIR filtering, run separately inside every valid span.

:func:`filter_raw_per_segment` checks that the annotations of the Raw object reproduce
:attr:`SegmentInfo.valid_spans` (:func:`~preprocessing.segments.check_annotations_match`) and then calls MNE's
``raw.filter`` with ``skip_by_annotation=SKIP_BY_ANNOTATION``, which filters each contiguous span on its own (edge
padding ``reflect_limited`` at both span ends). Samples inside BAD spans are not filtered.

:func:`filter_segments_explicit` is an equivalent explicit implementation (a loop calling
:func:`mne.filter.filter_data` on each span).
"""

from __future__ import annotations

import numpy as np

from preprocessing.segments import SKIP_BY_ANNOTATION, SegmentInfo, as_spans, check_annotations_match
from preprocessing.step_config import FilterSpec


def filter_raw_per_segment(raw, seg: SegmentInfo, spec: FilterSpec, picks, *, check: bool = True, n_jobs: int = 1):
    """Band-pass ``raw`` in place, separately in each valid span; returns ``raw``.

    Parameters
    ----------
    raw
        Preloaded Raw annotated with :func:`~preprocessing.segments.annotate_raw`.
    seg
        Segment info of the recording.
    spec
        Filter parameters.
    picks
        Channels to filter (names or indices).
    check
        Verify the annotation-based segmentation first (raises ``ValueError``).
    """
    if check:
        check_annotations_match(raw, seg)
    raw.filter(picks=picks, skip_by_annotation=list(SKIP_BY_ANNOTATION), n_jobs=n_jobs, verbose="error",
               **spec.mne_kwargs())
    return raw


def notch_raw_per_segment(raw, seg: SegmentInfo, freqs, picks, *, check: bool = True, n_jobs: int = 1):
    """FIR notch (zero phase, firwin, Hamming) at ``freqs`` in place, separately in each valid span."""
    freqs = [f for f in freqs if f < raw.info["sfreq"] / 2]
    if not freqs:
        return raw
    if check:
        check_annotations_match(raw, seg)
    raw.notch_filter(freqs, picks=picks, method="fir", phase="zero", fir_window="hamming", fir_design="firwin",
                     skip_by_annotation=list(SKIP_BY_ANNOTATION), n_jobs=n_jobs, verbose="error")
    return raw


def filter_segments_explicit(data: np.ndarray, sfreq: float, spans: np.ndarray, spec: FilterSpec) -> np.ndarray:
    """Reference implementation: filter each ``[start, stop)`` span of ``data`` on its own (returns a copy).

    Samples outside ``spans`` are returned unchanged.
    """
    from mne.filter import filter_data

    out = np.array(data, dtype=np.float64, copy=True)
    kw = spec.mne_kwargs()
    for a, b in as_spans(spans):
        if b <= a:
            continue
        out[:, a:b] = filter_data(out[:, a:b], sfreq, copy=True, verbose="error", **kw)
    return out


def fir_kernel(spec: FilterSpec, sfreq: float) -> np.ndarray:
    """The FIR kernel MNE designs for ``spec`` at ``sfreq``."""
    from mne.filter import create_filter

    kw = spec.mne_kwargs()
    kw.pop("pad")
    kw.pop("method")
    return create_filter(None, sfreq, verbose="error", **kw)


def filter_half_length_s(spec: FilterSpec, sfreq: float) -> float:
    """Half-length of the FIR kernel in seconds."""
    return (len(fir_kernel(spec, sfreq)) // 2) / sfreq


def filter_gain_db(spec: FilterSpec, sfreq: float, freqs) -> np.ndarray:
    """Magnitude response (dB) of the designed kernel at ``freqs`` (zero-phase: the kernel's own magnitude)."""
    h = fir_kernel(spec, sfreq)
    f = np.atleast_1d(np.asarray(freqs, float))
    n = np.arange(len(h)) - (len(h) - 1) / 2
    H = np.abs(np.exp(-2j * np.pi * np.outer(f, n) / sfreq) @ h)
    return 20 * np.log10(np.maximum(H, 1e-300))
