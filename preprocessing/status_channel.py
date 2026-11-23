"""BioSemi status channel: header and status reading, and decoding of the trigger codes and of the pause/resume
joins (rising edges of status bit 16) that :mod:`preprocessing.segments` uses to split the filter segments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StatusConfig:
    """Status-channel decoding parameters."""

    channel: str
    event_mask: int
    join_bit: int
    shortest_event: int
    stuck_code_min_samples: int
    status_only_marker_codes: tuple[int, ...]
    sample_onset_tolerance: int


class BdfFormatError(ValueError):
    """Raised when a file is not a readable BioSemi BDF (bad magic or inconsistent header)."""


class BdfTruncatedError(BdfFormatError):
    """Raised when a BDF holds fewer bytes than its header's record count promises."""


@dataclass(frozen=True)
class BdfHeader:
    """Minimal BDF header."""

    path: Path
    header_bytes: int
    n_records: int
    record_duration_s: float
    n_signals: int
    labels: tuple[str, ...]
    n_samples: tuple[int, ...]
    startdate: str
    starttime: str

    @property
    def record_bytes(self) -> int:
        return 3 * sum(self.n_samples)

    def sfreq(self, label: str) -> float:
        return self.n_samples[self.labels.index(label)] / self.record_duration_s


def read_bdf_header(path: str | os.PathLike[str]) -> BdfHeader:
    """Parse the fixed and per-signal BDF header fields needed to locate a channel.

    Raises
    ------
    BdfFormatError
        If the file is not a BioSemi BDF or its header is inconsistent.
    BdfTruncatedError
        If the file is shorter than ``header_bytes + n_records * record_bytes``
        (a truncated copy or an interrupted download).
    """
    p = Path(path)
    with open(p, "rb") as fh:
        h = fh.read(256)
        if len(h) < 256 or h[:8] != b"\xffBIOSEMI":
            raise BdfFormatError(f"{p.name} is not a BioSemi BDF file")
        ns = int(h[252:256].decode("ascii").strip())
        sh = fh.read(256 * ns)
    def field_(offset_per_signal: int, width: int) -> list[str]:
        base = offset_per_signal * ns
        return [sh[base + i * width: base + (i + 1) * width].decode("latin-1").strip() for i in range(ns)]
    labels = field_(0, 16)
    # label16 transducer80 physdim8 physmin8 physmax8 digmin8 digmax8 prefilter80 -> nsamp8 at 216
    nsamp = [int(x) for x in field_(216, 8)]
    header_bytes = int(h[184:192].decode("ascii").strip())
    n_rec = int(h[236:244].decode("ascii").strip())
    rec_dur = float(h[244:252].decode("ascii").strip())
    rec_bytes = 3 * sum(nsamp)
    size = p.stat().st_size
    if rec_bytes <= 0 or header_bytes != 256 * (ns + 1):
        raise BdfFormatError(f"{p.name}: inconsistent BDF header (header bytes {header_bytes}, {ns} signals, "
                             f"record bytes {rec_bytes})")
    if n_rec < 0:  # unknown in header: derive from the file size
        n_rec = (size - header_bytes) // rec_bytes
    expected = header_bytes + n_rec * rec_bytes
    if size < expected:
        raise BdfTruncatedError(
            f"{p.name}: truncated BDF: the header promises {n_rec} data records ({expected:,} bytes) but the file has "
            f"{size:,} bytes ({max(size - header_bytes, 0) // rec_bytes} complete records)")
    return BdfHeader(path=p, header_bytes=header_bytes, n_records=n_rec, record_duration_s=rec_dur,
                     n_signals=ns, labels=tuple(labels), n_samples=tuple(nsamp),
                     startdate=h[168:176].decode("latin-1"), starttime=h[176:184].decode("latin-1"))


@dataclass(frozen=True)
class StatusData:
    """Raw Status samples (unsigned 24-bit integers)."""

    values: np.ndarray
    sfreq: float
    record_samples: int
    path: Path

    @property
    def n_times(self) -> int:
        return int(self.values.size)


def read_status_channel(path: str | os.PathLike[str], channel: str = "Status") -> StatusData:
    """Read only the Status channel of a BDF via a memory map.

    Parameters
    ----------
    path
        BDF file.
    channel
        Status channel label (case-insensitive match on the start of the label).

    Returns
    -------
    StatusData
    """
    hdr = read_bdf_header(path)
    idx = [i for i, lab in enumerate(hdr.labels) if lab.lower().startswith(channel.lower())]
    if not idx:
        raise KeyError(f"{path}: no channel labelled {channel!r}")
    i = idx[-1]
    n = hdr.n_samples[i]
    start = 3 * sum(hdr.n_samples[:i])
    mm = np.memmap(hdr.path, dtype=np.uint8, mode="r", offset=hdr.header_bytes,
                   shape=(hdr.n_records, hdr.record_bytes))
    try:
        b = np.ascontiguousarray(mm[:, start: start + 3 * n]).reshape(hdr.n_records, n, 3).astype(np.int32)
    finally:
        del mm
    vals = (b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)).reshape(-1)
    return StatusData(values=vals, sfreq=hdr.sfreq(hdr.labels[i]), record_samples=n, path=hdr.path)


@dataclass(frozen=True)
class DecodedStatus:
    """Decoded Status channel.

    Attributes
    ----------
    events
        One row per onset: ``sample``, ``value`` (masked code), ``prev_value``
        (masked value just before), ``width`` (samples the code is held).
    joins
        Samples of join-bit rising edges after sample 0 (segment joins).
    join_bit_high_at_start
        Whether the join bit is high at sample 0.
    joins_off_record_boundary
        Number of joins not on a data-record boundary (expected 0).
    upper_bits_seen
        OR of all bits outside ``mask`` and the join bit (informational).
    """

    events: pd.DataFrame
    joins: np.ndarray
    join_bit_high_at_start: bool
    joins_off_record_boundary: int
    upper_bits_seen: int
    n_times: int
    sfreq: float
    record_samples: int
    mask: int = 0xFFFF
    join_bit: int = 16
    meta: dict = field(default_factory=dict)


def decode_status(
    values: np.ndarray,
    sfreq: float,
    *,
    mask: int = 0xFFFF,
    join_bit: int = 16,
    record_samples: int | None = None,
    shortest_event: int = 1,
) -> DecodedStatus:
    """Decode raw Status samples into trigger onsets and segment joins.

    Parameters
    ----------
    values
        Raw Status samples (integers; bits above 23 are ignored).
    sfreq
        Sampling rate of the Status channel.
    mask
        Bit mask applied to obtain trigger codes.
    join_bit
        Bit whose rising edges mark segment joins.
    record_samples
        Samples per BDF data record (default ``round(sfreq)``).
    shortest_event
        Onsets whose code is held for fewer samples are dropped.

    Returns
    -------
    DecodedStatus
    """
    v_all = np.asarray(values).astype(np.int64)
    if v_all.ndim != 1:
        raise ValueError("values must be one-dimensional")
    rec = int(record_samples) if record_samples else int(round(sfreq))
    n = v_all.size
    v = v_all & mask
    if n == 0:
        empty = pd.DataFrame({"sample": np.array([], np.int64), "value": np.array([], np.int64),
                              "prev_value": np.array([], np.int64), "width": np.array([], np.int64)})
        return DecodedStatus(empty, np.array([], np.int64), False, 0, 0, 0, float(sfreq), rec, mask, join_bit)
    prev = np.concatenate([[0], v[:-1]])
    change = np.flatnonzero(v != prev)
    onset = change[v[change] != 0]
    nxt = np.concatenate([change, [n]])
    pos = np.searchsorted(nxt, onset, side="right")
    width = nxt[pos] - onset
    keep = width >= shortest_event
    ev = pd.DataFrame({"sample": onset[keep].astype(np.int64), "value": v[onset[keep]].astype(np.int64),
                       "prev_value": prev[onset[keep]].astype(np.int64), "width": width[keep].astype(np.int64)})
    b = (v_all >> join_bit) & 1
    rises = (np.flatnonzero((b[1:] == 1) & (b[:-1] == 0)) + 1).astype(np.int64)
    upper = int(np.bitwise_or.reduce(v_all & ~(mask | (1 << join_bit)) & 0xFFFFFF)) if n else 0
    return DecodedStatus(
        events=ev, joins=rises, join_bit_high_at_start=bool(b[0]),
        joins_off_record_boundary=int(np.sum(rises % rec != 0)), upper_bits_seen=upper,
        n_times=int(n), sfreq=float(sfreq), record_samples=rec, mask=mask, join_bit=join_bit,
    )


def decode_status_file(path: str | os.PathLike[str], cfg: StatusConfig) -> DecodedStatus:
    """Read and decode the Status channel of one BDF using the events config."""
    st = read_status_channel(path, cfg.channel)
    dec = decode_status(st.values, st.sfreq, mask=cfg.event_mask, join_bit=cfg.join_bit,
                        record_samples=st.record_samples, shortest_event=cfg.shortest_event)
    dec.meta["path"] = str(st.path)
    return dec


def extract_markers(decoded: DecodedStatus, codes: set[int] | list[int] | tuple[int, ...]) -> pd.DataFrame:
    """Onsets whose code is in ``codes`` (``sample``, ``value``, ``width``)."""
    ev = decoded.events
    return ev.loc[ev["value"].isin(list(codes)), ["sample", "value", "width"]].reset_index(drop=True)


def stuck_codes(decoded: DecodedStatus, min_samples: int) -> pd.DataFrame:
    """Onsets whose code is held for at least ``min_samples`` samples."""
    ev = decoded.events
    return ev.loc[ev["width"] >= min_samples].reset_index(drop=True)


def joins_between(joins: np.ndarray, start: np.ndarray | int, stop: np.ndarray | int) -> np.ndarray:
    """Number of joins with ``start < join <= stop`` (vectorised over start/stop)."""
    j = np.asarray(joins, np.int64)
    return np.searchsorted(j, np.asarray(stop), side="right") - np.searchsorted(j, np.asarray(start), side="right")


@dataclass(frozen=True)
class CrosscheckReport:
    """events.tsv vs Status, matched on exact (sample, value).

    Attributes
    ----------
    n_tsv
        Number of events.tsv rows.
    n_matched
        Rows with a Status onset at the same sample and with the same code.
    tsv_only
        events.tsv rows without such an onset.
    status_only
        Status onsets absent from events.tsv.
    """

    n_tsv: int
    n_matched: int
    tsv_only: pd.DataFrame
    status_only: pd.DataFrame
    n_joins: int
    joins_off_record_boundary: int
    join_bit_high_at_start: bool

    @property
    def sample_exact(self) -> bool:
        """Every events.tsv row is present in Status at the exact sample."""
        return self.n_matched == self.n_tsv

    def status_only_counts(self) -> dict[int, int]:
        return {int(k): int(v) for k, v in self.status_only["value"].value_counts().sort_index().items()}

    def summary(self) -> dict:
        """Flat summary suitable for one table row."""
        return {
            "n_tsv": self.n_tsv, "n_matched": self.n_matched, "n_tsv_only": len(self.tsv_only),
            "n_status_only": len(self.status_only), "sample_exact": self.sample_exact,
            "status_only_counts": ";".join(f"{k}:{v}" for k, v in self.status_only_counts().items()),
            "n_joins": self.n_joins, "joins_off_record_boundary": self.joins_off_record_boundary,
            "join_bit_high_at_start": self.join_bit_high_at_start,
        }


def crosscheck_events(tsv: pd.DataFrame, decoded: DecodedStatus) -> CrosscheckReport:
    """Compare events.tsv rows (``sample``, ``value``) with decoded Status onsets.

    Parameters
    ----------
    tsv
        events.tsv with integer ``sample`` and ``value`` columns.
    decoded
        Output of :func:`decode_status`.
    """
    t = tsv[["sample", "value"]].astype("int64")
    s = decoded.events
    key_t = pd.MultiIndex.from_frame(t)
    key_s = pd.MultiIndex.from_frame(s[["sample", "value"]])
    in_s = key_t.isin(key_s)
    in_t = key_s.isin(key_t)
    return CrosscheckReport(
        n_tsv=len(t), n_matched=int(in_s.sum()), tsv_only=t.loc[~in_s].reset_index(drop=True),
        status_only=s.loc[~in_t].reset_index(drop=True), n_joins=int(decoded.joins.size),
        joins_off_record_boundary=decoded.joins_off_record_boundary,
        join_bit_high_at_start=decoded.join_bit_high_at_start,
    )
