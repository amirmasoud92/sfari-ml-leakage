"""Recording segments: joins, rail / flat / unusable spans and the valid spans between them.

A BioSemi task file is a concatenation of separately saved blocks. The only marker of a block boundary is a rising
edge of status bit 16 (a *join*, :mod:`preprocessing.status_channel`). The raw signal jumps at a join, and the cap
can be unplugged while triggers continue (all EEG at the rail). Every filter therefore runs separately inside each
*valid span*:

- the recording ``[0, n_times)`` is split at every join;
- rail spans (per-sample median over EEG channels below ``rail.threshold_v``), flat spans (per-sample median
  |first difference| below ``flat.max_abs_diff_uv``) and *unusable* spans are removed;
- a remaining piece shorter than the preprocessing FIR kernel (``segments.min_valid_span_s``; ``auto`` = the
  longest kernel) cannot be filtered without distortion; it becomes a *short* span (``BAD_short``) and is removed
  as well;
- what remains, as half-open sample intervals ``[start, stop)``, is :attr:`SegmentInfo.valid_spans`.

All sample positions are 0-based indices into the recording's data array. Spans are ``(k, 2)`` int64 arrays of
half-open ``[start, stop)`` intervals, sorted and non-overlapping.

The same structure is written to the Raw object as annotations (:func:`annotate_raw`): ``EDGE boundary`` (zero
duration) at every join and ``BAD_rail`` / ``BAD_flat`` / ``BAD_unusable`` / ``BAD_short`` spans. MNE's filters split
the data at the annotations named in :data:`SKIP_BY_ANNOTATION` (case-insensitive prefix match), and
:func:`check_annotations_match` verifies that MNE's segmentation equals :attr:`SegmentInfo.valid_spans`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from preprocessing.step_config import SegmentsConfig, SpanRule

#: Annotation descriptions.
RAIL_DESC = "BAD_rail"
FLAT_DESC = "BAD_flat"
UNUSABLE_DESC = "BAD_unusable"
SHORT_DESC = "BAD_short"
DEFAULT_JOIN_DESC = "EDGE boundary"
#: Prefixes (MNE: case-insensitive ``startswith``) at which every filter must split the data.
SKIP_BY_ANNOTATION: tuple[str, ...] = ("edge", RAIL_DESC.lower(), FLAT_DESC.lower(), UNUSABLE_DESC.lower(),
                                       SHORT_DESC.lower())
#: Our annotation descriptions (replaced, not duplicated, by :func:`annotate_raw`).
OWN_DESCRIPTIONS = (RAIL_DESC, FLAT_DESC, UNUSABLE_DESC, SHORT_DESC)


def as_spans(spans: Iterable[Sequence[int]] | np.ndarray | None) -> np.ndarray:
    """``(k, 2)`` int64 array from any iterable of (start, stop) pairs (empty -> shape ``(0, 2)``)."""
    if spans is None:
        return np.zeros((0, 2), np.int64)
    a = np.asarray(list(spans) if not isinstance(spans, np.ndarray) else spans, dtype=np.int64)
    if a.size == 0:
        return np.zeros((0, 2), np.int64)
    a = a.reshape(-1, 2)
    if np.any(a[:, 1] < a[:, 0]):
        raise ValueError("span stop < start")
    return a


def mask_to_spans(mask: np.ndarray) -> np.ndarray:
    """Half-open spans of the True runs of a boolean mask."""
    m = np.asarray(mask, bool)
    if m.size == 0:
        return np.zeros((0, 2), np.int64)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return np.stack([starts, stops], axis=1).astype(np.int64)


def spans_to_mask(spans: np.ndarray, n: int) -> np.ndarray:
    """Boolean mask of length ``n`` that is True inside ``spans``."""
    m = np.zeros(int(n), bool)
    for a, b in as_spans(spans):
        m[max(a, 0):min(b, n)] = True
    return m


def merge_spans(spans: np.ndarray, gap: int = 0) -> np.ndarray:
    """Sort spans and merge those that overlap or are separated by fewer than ``gap`` samples."""
    s = as_spans(spans)
    s = s[s[:, 1] > s[:, 0]]
    if len(s) == 0:
        return np.zeros((0, 2), np.int64)
    s = s[np.argsort(s[:, 0], kind="stable")]
    out = [list(s[0])]
    for a, b in s[1:]:
        if a - out[-1][1] < gap or a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.asarray(out, np.int64)


def subtract_spans(base: np.ndarray, remove: np.ndarray) -> np.ndarray:
    """``base`` minus the union of ``remove`` (both half-open span arrays)."""
    base = merge_spans(base)
    rem = merge_spans(remove)
    out: list[tuple[int, int]] = []
    for a, b in base:
        cur = a
        for c, d in rem:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return as_spans(out)


def split_spans_at(spans: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Split spans at the given sample positions (a point ``p`` starts a new span at ``p``)."""
    pts = np.unique(np.asarray(points, np.int64))
    out: list[tuple[int, int]] = []
    for a, b in as_spans(spans):
        inner = pts[(pts > a) & (pts < b)]
        edges = np.concatenate([[a], inner, [b]])
        out.extend(zip(edges[:-1], edges[1:]))
    return as_spans(out)


def span_total(spans: np.ndarray) -> int:
    """Number of samples covered by (merged) ``spans``."""
    s = merge_spans(spans)
    return int((s[:, 1] - s[:, 0]).sum()) if len(s) else 0


def apply_span_rule(mask: np.ndarray, sfreq: float, rule: SpanRule) -> np.ndarray:
    """Mask -> spans: drop runs shorter than ``min_duration_s``, merge across gaps < ``merge_gap_s``, pad."""
    n = len(mask)
    s = mask_to_spans(mask)
    if len(s) == 0:
        return s
    min_len = int(round(rule.min_duration_s * sfreq))
    if min_len > 0:
        s = s[(s[:, 1] - s[:, 0]) >= min_len]
    s = merge_spans(s, gap=int(round(rule.merge_gap_s * sfreq)))
    pad = int(round(rule.pad_s * sfreq))
    if pad and len(s):
        s = np.stack([np.maximum(s[:, 0] - pad, 0), np.minimum(s[:, 1] + pad, n)], axis=1)
    return merge_spans(s)


@dataclass(frozen=True)
class SegmentInfo:
    """Segment structure of one recording (sample indices, 0-based, half-open spans).

    Attributes
    ----------
    sfreq
        Sampling rate (Hz).
    n_times
        Samples in the recording (equal to the BDF's ``n_records * 512``).
    first_samp
        ``raw.first_samp`` of the source Raw (0 for BDF), stored for checking.
    joins
        Sorted samples of Status bit-16 rising edges (``0 < join < n_times``). A
        join is the first sample of a new saved block.
    rail_spans, flat_spans, unusable_spans
        ``(k, 2)`` spans of each bad kind (see module docstring).
    unusable_reasons
        One label per row of ``unusable_spans``.
    valid_spans
        ``[0, n_times)`` split at the joins, minus every bad span: the units in
        which filters run, and the only data that bad-channel detection, ICA and
        windows may use.
    short_spans
        Pieces that would otherwise be valid but are shorter than the minimum
        valid-span length (``BAD_short``; empty for most recordings).
    """

    sfreq: float
    n_times: int
    first_samp: int
    joins: np.ndarray
    rail_spans: np.ndarray
    flat_spans: np.ndarray
    unusable_spans: np.ndarray
    unusable_reasons: tuple[str, ...]
    valid_spans: np.ndarray
    meta: dict = field(default_factory=dict, compare=False)
    short_spans: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))

    @classmethod
    def build(cls, sfreq: float, n_times: int, joins: Iterable[int] = (), rail_spans=None, flat_spans=None,
              unusable_spans=None, unusable_reasons: Sequence[str] = (), first_samp: int = 0,
              meta: dict | None = None, short_spans=None, min_valid_samples: int = 0) -> SegmentInfo:
        """Assemble a :class:`SegmentInfo` and derive :attr:`valid_spans`.

        ``short_spans`` are given explicitly (e.g. read back from JSON). With ``min_valid_samples > 0`` every piece
        that would be valid but is shorter than that becomes a short span as well.
        """
        n = int(n_times)
        j = np.unique(np.asarray(list(joins), np.int64))
        if j.size and (j.min() <= 0 or j.max() >= n):
            raise ValueError(f"joins must lie in (0, n_times={n}); got {j.min()}..{j.max()}")
        rail = merge_spans(np.clip(as_spans(rail_spans), 0, n))
        flat = merge_spans(np.clip(as_spans(flat_spans), 0, n))
        un = np.clip(as_spans(unusable_spans), 0, n)
        reasons = tuple(unusable_reasons)
        if len(reasons) != len(un):
            raise ValueError("unusable_reasons must have one entry per unusable span")
        short = merge_spans(np.clip(as_spans(short_spans), 0, n))
        bad = merge_spans(np.concatenate([rail, flat, un, short]))
        valid = split_spans_at(subtract_spans(as_spans([(0, n)]) if n > 0 else as_spans(None), bad), j)
        valid = valid[valid[:, 1] > valid[:, 0]]
        if min_valid_samples > 0 and len(valid):
            too_short = (valid[:, 1] - valid[:, 0]) < int(min_valid_samples)
            if too_short.any():
                short = merge_spans(np.concatenate([short, valid[too_short]]))
                valid = valid[~too_short]
        return cls(sfreq=float(sfreq), n_times=n, first_samp=int(first_samp), joins=j, rail_spans=rail,
                   flat_spans=flat, unusable_spans=un, unusable_reasons=reasons, valid_spans=valid,
                   meta=dict(meta or {}), short_spans=as_spans(short))

    @property
    def bad_spans(self) -> np.ndarray:
        """Union of rail, flat, unusable and short spans."""
        return merge_spans(np.concatenate([self.rail_spans, self.flat_spans, self.unusable_spans,
                                           self.short_spans]))

    @property
    def n_segments(self) -> int:
        """Number of valid spans (filter segments)."""
        return int(len(self.valid_spans))

    @property
    def segment_lengths_s(self) -> np.ndarray:
        return (self.valid_spans[:, 1] - self.valid_spans[:, 0]) / self.sfreq

    def valid_mask(self) -> np.ndarray:
        """Boolean mask over samples: True inside a valid span."""
        return spans_to_mask(self.valid_spans, self.n_times)

    def fraction(self, kind: str) -> float:
        """Fraction of samples covered by ``kind`` in {rail, flat, unusable, bad, valid}."""
        spans = {"rail": self.rail_spans, "flat": self.flat_spans, "unusable": self.unusable_spans,
                 "short": self.short_spans, "bad": self.bad_spans, "valid": self.valid_spans}[kind]
        return span_total(spans) / self.n_times if self.n_times else 0.0

    def segment_of(self, samples: np.ndarray | Sequence[int] | int) -> np.ndarray:
        """Index into :attr:`valid_spans` of each sample (-1 if the sample is not in a valid span)."""
        s = np.atleast_1d(np.asarray(samples, np.int64))
        v = self.valid_spans
        if len(v) == 0:
            return np.full(s.shape, -1, np.int64)
        i = np.searchsorted(v[:, 0], s, side="right") - 1
        ok = (i >= 0) & (s < v[np.clip(i, 0, None), 1])
        return np.where(ok, i, -1).astype(np.int64)

    def window_valid(self, start: np.ndarray | int, stop: np.ndarray | int, margin: int = 0) -> np.ndarray:
        """Whether each window ``[start, stop)`` lies inside ONE valid span, with ``margin`` samples to spare.

        With ``margin`` equal to the FIR half-length this is an edge-validity rule: no join, file edge or bad
        span within ``margin`` samples of either window edge.
        """
        a = np.atleast_1d(np.asarray(start, np.int64))
        b = np.atleast_1d(np.asarray(stop, np.int64))
        if a.shape != b.shape:
            raise ValueError("start and stop must have the same shape")
        seg = self.segment_of(a)
        ok = seg >= 0
        v = self.valid_spans
        idx = np.clip(seg, 0, None)
        if len(v):
            ok &= (a - v[idx, 0] >= margin) & (v[idx, 1] - b >= margin) & (b > a)
        return ok

    def summary(self) -> dict:
        """Flat QC summary."""
        L = self.segment_lengths_s
        return {
            "n_times": self.n_times, "sfreq": self.sfreq, "duration_s": self.n_times / self.sfreq if self.sfreq else 0,
            "n_joins": int(self.joins.size), "n_segments": self.n_segments,
            "segment_s_min": float(L.min()) if L.size else None, "segment_s_median": float(np.median(L)) if L.size else None,
            "n_rail_spans": int(len(self.rail_spans)), "rail_fraction": self.fraction("rail"),
            "n_flat_spans": int(len(self.flat_spans)), "flat_fraction": self.fraction("flat"),
            "n_unusable_spans": int(len(self.unusable_spans)), "unusable_fraction": self.fraction("unusable"),
            "unusable_reasons": list(self.unusable_reasons), "valid_fraction": self.fraction("valid"),
            "n_short_spans": int(len(self.short_spans)), "short_fraction": self.fraction("short"),
        }

    def to_dict(self) -> dict:
        """JSON-ready dict. ``short_spans`` is written only when there is one."""
        d = {"sfreq": self.sfreq, "n_times": self.n_times, "first_samp": self.first_samp,
             "joins": self.joins.tolist(), "rail_spans": self.rail_spans.tolist(),
             "flat_spans": self.flat_spans.tolist(), "unusable_spans": self.unusable_spans.tolist(),
             "unusable_reasons": list(self.unusable_reasons), "valid_spans": self.valid_spans.tolist(),
             "meta": self.meta}
        if len(self.short_spans):
            d["short_spans"] = self.short_spans.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SegmentInfo:
        """Inverse of :meth:`to_dict`; :attr:`valid_spans` is re-derived and checked against the stored value."""
        si = cls.build(d["sfreq"], d["n_times"], d["joins"], d["rail_spans"], d["flat_spans"], d["unusable_spans"],
                       d.get("unusable_reasons", ()), d.get("first_samp", 0), d.get("meta"),
                       short_spans=d.get("short_spans"))
        if "valid_spans" in d and not np.array_equal(as_spans(d["valid_spans"]), si.valid_spans):
            raise ValueError("stored valid_spans differ from the ones derived from joins and bad spans")
        return si

    def to_json(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> SegmentInfo:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _channel_median(data: np.ndarray, chunk: int = 1 << 16) -> np.ndarray:
    """Per-sample median over channels, computed in chunks (bounded memory)."""
    n = data.shape[1]
    out = np.empty(n, np.float64)
    for a in range(0, n, chunk):
        out[a:a + chunk] = np.median(data[:, a:a + chunk], axis=0)
    return out


def detect_rail_spans(eeg: np.ndarray, sfreq: float, threshold_v: float, rule: SpanRule) -> np.ndarray:
    """Spans where the per-sample median over EEG channels is below ``threshold_v`` (volts)."""
    if eeg.shape[1] == 0:
        return as_spans(None)
    return apply_span_rule(_channel_median(eeg) < threshold_v, sfreq, rule)


def detect_flat_spans(eeg: np.ndarray, sfreq: float, max_abs_diff_v: float, rule: SpanRule) -> np.ndarray:
    """Spans where the per-sample median |first difference| over EEG channels is below ``max_abs_diff_v``."""
    n = eeg.shape[1]
    if n < 2:
        return as_spans(None)
    med = np.empty(n, np.float64)
    chunk = 1 << 16
    for a in range(1, n, chunk):
        b = min(a + chunk, n)
        med[a:b] = np.median(np.abs(eeg[:, a:b] - eeg[:, a - 1:b - 1]), axis=0)
    med[0] = med[1]
    return apply_span_rule(med < max_abs_diff_v, sfreq, rule)


def build_segment_info(eeg: np.ndarray, sfreq: float, joins: Iterable[int], cfg: SegmentsConfig, *,
                       unusable_spans=None, unusable_reasons: Sequence[str] = (), first_samp: int = 0,
                       min_valid_samples: int = 0) -> SegmentInfo:
    """Detect rail and flat spans on the raw (unfiltered, volts) EEG and assemble the :class:`SegmentInfo`.

    Parameters
    ----------
    eeg
        ``(n_eeg, n_times)`` raw EEG in volts (CMS reference, DC-coupled).
    sfreq
        Sampling rate.
    joins
        Join samples (``DecodedStatus.joins``).
    cfg
        ``segments`` section of the preprocessing config.
    unusable_spans, unusable_reasons
        Additional spans to exclude, with one reason each.
    min_valid_samples
        Valid pieces shorter than this become ``BAD_short`` (see :func:`min_valid_samples`).
    """
    rail = detect_rail_spans(eeg, sfreq, cfg.rail_threshold_v, cfg.rail)
    flat = detect_flat_spans(eeg, sfreq, cfg.flat_max_abs_diff_uv * 1e-6, cfg.flat)
    return SegmentInfo.build(sfreq, eeg.shape[1], joins, rail, flat, unusable_spans, unusable_reasons, first_samp,
                             min_valid_samples=min_valid_samples)


def min_valid_samples(cfg: SegmentsConfig, kernels_samples: Sequence[int], sfreq: float) -> int:
    """Minimum valid-span length in samples: ``min_valid_span_s`` or, for ``"auto"``, the longest FIR kernel.

    ``kernels_samples`` are the lengths (taps) of the FIR kernels that run per valid span (the 1-40 Hz analysis
    filter and the 1-100 Hz copy; 1691 taps each at 512 Hz). A span shorter than its kernel is filtered with
    distortion (MNE: "filter_length is longer than the signal").
    """
    v = cfg.min_valid_span_s
    if v == "auto":
        return int(max(kernels_samples)) if len(kernels_samples) else 0
    return int(round(float(v) * sfreq))


def _onset_s(raw, samples: np.ndarray) -> np.ndarray:
    """Annotation onsets (s, in the convention of ``raw.annotations``) of data-array sample indices."""
    base = raw.first_time if raw.annotations.orig_time is not None else 0.0
    return base + np.asarray(samples, np.float64) / raw.info["sfreq"]


def effective_joins(seg: SegmentInfo) -> np.ndarray:
    """Joins that are annotated: those not inside (or on the edge of) a bad span.

    Such a join changes nothing (the span is not filtered) and would make MNE pass an empty segment to the filter.
    """
    bad = seg.bad_spans
    j = seg.joins
    if len(bad) and j.size:
        inside = np.zeros(j.size, bool)
        for a, b in bad:
            inside |= (j >= a) & (j <= b)
        j = j[~inside]
    return j


def annotate_raw(raw, seg: SegmentInfo, join_description: str = DEFAULT_JOIN_DESC):
    """Add join (``EDGE boundary``) and bad-span annotations to ``raw`` in place and return it.

    Existing annotations with our descriptions (and existing join annotations
    at the same samples) are replaced, so the call is idempotent.
    """
    import mne

    if raw.n_times != seg.n_times or raw.first_samp != seg.first_samp:
        raise ValueError(f"raw ({raw.n_times} samples, first_samp {raw.first_samp}) does not match the segment info "
                         f"({seg.n_times}, {seg.first_samp})")
    old = raw.annotations
    keep = [i for i, d in enumerate(old.description) if d not in OWN_DESCRIPTIONS and d != join_description]
    on, du, de = [], [], []
    j = effective_joins(seg)
    on.extend(_onset_s(raw, j))
    du.extend([0.0] * j.size)
    de.extend([join_description] * j.size)
    for spans, desc in ((seg.rail_spans, RAIL_DESC), (seg.flat_spans, FLAT_DESC), (seg.unusable_spans, UNUSABLE_DESC),
                        (seg.short_spans, SHORT_DESC)):
        for a, b in merge_spans(spans):
            on.append(_onset_s(raw, np.array([a]))[0])
            du.append((b - a) / raw.info["sfreq"])
            de.append(desc)
    new = mne.Annotations(onset=np.concatenate([old.onset[keep], np.asarray(on, float)]),
                          duration=np.concatenate([old.duration[keep], np.asarray(du, float)]),
                          description=np.concatenate([old.description[keep], np.asarray(de, dtype=object)]),
                          orig_time=old.orig_time)
    raw.set_annotations(new)
    return raw


def mne_segments(raw, skip_by_annotation: Sequence[str] = SKIP_BY_ANNOTATION) -> np.ndarray:
    """The ``[start, stop)`` spans in which MNE's ``raw.filter(skip_by_annotation=...)`` filters separately."""
    from mne.annotations import _annotations_starts_stops

    on, off = _annotations_starts_stops(raw, list(skip_by_annotation), invert=True)
    s = as_spans(np.stack([on, off], axis=1)) if len(on) else as_spans(None)
    return s[s[:, 1] > s[:, 0]]


def check_annotations_match(raw, seg: SegmentInfo, skip_by_annotation: Sequence[str] = SKIP_BY_ANNOTATION) -> None:
    """Raise ``ValueError`` unless MNE's annotation-based segmentation equals ``seg.valid_spans``."""
    got = mne_segments(raw, skip_by_annotation)
    if not np.array_equal(got, seg.valid_spans):
        raise ValueError(f"MNE segments from annotations ({len(got)}) differ from SegmentInfo.valid_spans "
                         f"({seg.n_segments}); first: {got[:3].tolist()} vs {seg.valid_spans[:3].tolist()}")


def zero_spans(raw, spans: np.ndarray, picks=None) -> None:
    """Set ``raw`` data to 0 inside ``spans`` for ``picks`` (default: every non-stim channel), in place."""
    import mne

    if picks is None:
        picks = mne.pick_types(raw.info, eeg=True, eog=True, misc=True, emg=True, ecg=True, exclude=[])
    picks = np.asarray(picks, int)
    for a, b in merge_spans(spans):
        raw._data[picks, a:b] = 0.0
