"""ICA on a 1-Hz high-passed copy, ICLabel classification and a configurable exclusion rule (Methods,
Preprocessing; Additional file 1, Supplementary Methods S1).

- Fit data (:func:`ica_fit_data`): the 1-100 Hz per-segment FIR copy of the *good* EEG channels,
  average-referenced over those channels, restricted to the valid spans (no join, rail, flat or unusable sample),
  with 1-s chunks whose peak-to-peak exceeds ``fit_reject_ptp_uv`` in any channel left out (gross movement; blinks
  stay in, the decomposition needs them). ICLabel expects 1-100 Hz and a common average reference.
- Decomposition: Picard with ``extended=True, ortho=False``, which converges to the extended-infomax solution
  ICLabel was trained on; seeded ``random_state`` (deterministic).
- Number of components (:func:`choose_n_components`): the data rank after the average reference
  (``n_good - 1``), capped so that the fit has at least ``min_samples_factor * n_components**2`` samples (short
  resting runs get fewer components), and by ``max_components``. Every 2nd sample is used (256 Hz) when that still
  allows the cap.
- Labels: ICLabel (mne-icalabel, ONNX backend) on the same fit data. The full 7-class probability matrix is
  stored.
- Exclusion (:func:`select_exclusions`): a component is excluded for class c if its probability for c is at
  least ``exclude[c]`` and, with ``rule = argmax_and_threshold``, c is also its most probable class. The exclusions
  finally applied are the ones of the re-selection step (:mod:`preprocessing.ica_selection`: posterior-alpha veto)
  and of the eye validation in :mod:`preprocessing.biosemi`.
- Fit-rejected chunks (:attr:`ICAFitData.rejected_spans`): kept as sample spans.
- Threads: ICLabel's onnxruntime session can be capped (:func:`onnx_thread_cap`).

Note on referencing: the ICA is fitted on data referenced to the mean of the good channels, so it must be applied
to data on that same reference, *before* interpolation (:func:`preprocessing.cleaning.clean_from_parts`).
"""

from __future__ import annotations

import contextlib
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from preprocessing.segments import SegmentInfo
from preprocessing.step_config import ICLABEL_CLASSES, ICAConfig
from utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class ICAFitData:
    """Data the ICA is fitted on."""

    raw: object                      # mne RawArray of the good EEG channels, average-referenced
    n_valid_samples: int
    n_rejected_chunks: int
    n_chunks: int
    decimate: int
    n_components: int
    rank: int
    #: ``(k, 2)`` half-open sample spans (indices of the source Raw) of the chunks left out of the fit, merged
    rejected_spans: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))
    #: ``(k, 2)`` contiguous runs of the source Raw that went into the fit (valid spans minus rejected chunks)
    kept_spans: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))


@dataclass
class ICAResult:
    """A fitted ICA, its ICLabel probabilities and the excluded components."""

    ica: object
    probabilities: np.ndarray        # (n_components, 7), columns ICLABEL_CLASSES
    excluded: list[int]
    excluded_by_label: dict[str, list[int]]
    qc: dict = field(default_factory=dict)
    fit: ICAFitData | None = None

    @property
    def labels(self) -> list[str]:
        return [ICLABEL_CLASSES[i] for i in np.argmax(self.probabilities, axis=1)]

    def table(self):
        """One row per component: ``component``, the 7 probabilities, ``label``, ``excluded``, ``exclusion_label``."""
        import pandas as pd

        df = pd.DataFrame(self.probabilities, columns=[c.replace(" ", "_") for c in ICLABEL_CLASSES])
        df.insert(0, "component", np.arange(len(df), dtype=np.int64))
        df["label"] = self.labels
        df["excluded"] = df["component"].isin(self.excluded)
        lab = {c: k for k, cs in self.excluded_by_label.items() for c in cs}
        df["exclusion_label"] = [lab.get(int(c), "") for c in df["component"]]
        ev = self.qc.get("component_explained_variance")
        if ev is not None and len(ev) == len(df):
            df["explained_variance"] = ev
        return df


def choose_n_components(rank: int, n_samples: int, cfg: ICAConfig) -> int:
    """``min(rank, floor(sqrt(n_samples / k)), max_components)`` (or the fixed ``n_components``)."""
    if cfg.n_components != "auto":
        return int(min(int(cfg.n_components), rank))
    n = min(rank, int(np.floor(np.sqrt(max(n_samples, 0) / cfg.min_samples_factor))))
    if cfg.max_components is not None:
        n = min(n, cfg.max_components)
    return int(max(n, 0))


def ica_fit_data(hp_raw, seg: SegmentInfo, good_names: Sequence[str], cfg: ICAConfig, montage: str = "biosemi64",
                 chunk_s: float | None = None) -> ICAFitData:
    """Build the ICA fit data (see the module docstring) from the per-segment 1-100 Hz copy."""
    import mne

    from preprocessing.reference import robust_average_reference

    sf = float(hp_raw.info["sfreq"])
    idx = [hp_raw.ch_names.index(c) for c in good_names]
    x_all = hp_raw._data[idx]
    chunk = int(round((chunk_s or cfg.fit_reject_chunk_s) * sf))
    thr = cfg.fit_reject_ptp_uv * 1e-6 if cfg.fit_reject_ptp_uv is not None else None
    from preprocessing.segments import as_spans, mask_to_spans, merge_spans

    parts, n_chunks, n_rej, n_valid = [], 0, 0, 0
    rejected: list[tuple[int, int]] = []
    kept: list[np.ndarray] = []
    for a, b in seg.valid_spans:
        x = x_all[:, a:b]
        x = x - x.mean(axis=0, keepdims=True)              # average reference over the good channels
        n_valid += b - a
        keep = np.ones(b - a, bool)
        for c0 in range(0, b - a, chunk):
            c1 = min(c0 + chunk, b - a)
            n_chunks += 1
            if thr is not None and np.ptp(x[:, c0:c1], axis=1).max() > thr:
                keep[c0:c1] = False
                n_rej += 1
                rejected.append((int(a + c0), int(a + c1)))
        if keep.any():
            parts.append(x[:, keep])
            kept.append(mask_to_spans(keep) + int(a))
    data = np.concatenate(parts, axis=1) if parts else np.zeros((len(idx), 0))
    rank = max(len(idx) - 1, 0)
    cap = rank if cfg.max_components is None else min(rank, cfg.max_components)
    dec = 1
    if cfg.decimate > 1 and choose_n_components(rank, data.shape[1] // cfg.decimate, cfg) >= cap:
        dec = cfg.decimate
    data = data[:, ::dec]
    info = mne.create_info(list(good_names), sf / dec, "eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_montage(montage, on_missing="raise", verbose="error")
    robust_average_reference(raw)                        # no-op on the data; marks the average reference
    with raw.info._unlock():
        raw.info["highpass"] = float(hp_raw.info["highpass"])
        raw.info["lowpass"] = float(hp_raw.info["lowpass"])
    return ICAFitData(raw=raw, n_valid_samples=int(n_valid), n_rejected_chunks=int(n_rej), n_chunks=int(n_chunks),
                      decimate=dec, n_components=choose_n_components(rank, data.shape[1], cfg), rank=rank,
                      rejected_spans=merge_spans(as_spans(rejected)),
                      kept_spans=np.concatenate(kept).astype(np.int64) if kept else as_spans(None))


def fit_ica(fit: ICAFitData, cfg: ICAConfig, seed: int):
    """Fit the ICA on ``fit.raw``; returns (ica, warnings)."""
    from mne.preprocessing import ICA

    ica = ICA(n_components=fit.n_components, method=cfg.method, fit_params=dict(cfg.fit_params),
              max_iter=cfg.max_iter, random_state=seed, verbose="error")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ica.fit(fit.raw, picks="eeg", reject_by_annotation=False, verbose="error")
    msgs = sorted({str(x.message) for x in w if "converge" in str(x.message).lower()})
    return ica, msgs


@contextlib.contextmanager
def onnx_thread_cap(n_threads: int | None):
    """Cap the intra-op threads of the onnxruntime sessions mne-icalabel creates (``None``: no cap).

    mne-icalabel builds ``onnxruntime.InferenceSession(network_file)`` without session options, so its pool uses
    every core whatever ``threadpoolctl`` says; inside this context the session gets ``intra_op_num_threads =
    n_threads`` and ``inter_op_num_threads = 1``. The ICLabel probabilities do not depend on it (unit test
    ``test_iclabel_thread_cap_is_bit_identical``).
    """
    if not n_threads:
        yield
        return
    import onnxruntime as ort

    orig = ort.InferenceSession

    def capped(*args, **kwargs):
        so = kwargs.get("sess_options") or ort.SessionOptions()
        so.intra_op_num_threads = int(n_threads)
        so.inter_op_num_threads = 1
        kwargs["sess_options"] = so
        return orig(*args, **kwargs)

    ort.InferenceSession = capped  # type: ignore[assignment,misc]
    try:
        yield
    finally:
        ort.InferenceSession = orig  # type: ignore[misc]


def iclabel_probabilities(fit_raw, ica, n_threads: int | None = None) -> np.ndarray:
    """ICLabel 7-class probabilities (columns :data:`ICLABEL_CLASSES`); ``n_threads`` caps onnxruntime."""
    from mne_icalabel.iclabel import iclabel_label_components

    with warnings.catch_warnings(), onnx_thread_cap(n_threads):
        warnings.simplefilter("ignore")
        p = iclabel_label_components(fit_raw, ica, inplace=True, backend="onnx")
    return np.asarray(p, dtype=np.float64)


def select_exclusions(probs: np.ndarray, rule: str, thresholds: dict[str, float]) -> tuple[list[int], dict[str, list[int]]]:
    """Components to exclude under the configured rule; returns (sorted indices, label -> indices)."""
    probs = np.asarray(probs, float)
    if probs.ndim != 2 or probs.shape[1] != len(ICLABEL_CLASSES):
        raise ValueError(f"probabilities must be (n, {len(ICLABEL_CLASSES)})")
    top = np.argmax(probs, axis=1) if len(probs) else np.zeros(0, int)
    by: dict[str, list[int]] = {}
    for label, thr in thresholds.items():
        k = ICLABEL_CLASSES.index(label)
        hit = probs[:, k] >= thr
        if rule == "argmax_and_threshold":
            hit &= top == k
        elif rule != "threshold":
            raise ValueError(f"unknown rule {rule!r}")
        if hit.any():
            by[label] = [int(i) for i in np.flatnonzero(hit)]
    excl = sorted({i for v in by.values() for i in v})
    return excl, by


def component_explained_variance(ica, fit_raw) -> np.ndarray:
    """Fraction of the fit data's EEG variance explained by each component on its own.

    Same definition as MNE's ``ICA.get_explained_variance_ratio(components=[i])``
    (1 - mean over time of the across-channel variance of the residual, over that
    of the data), computed for all components at once from the back-projection
    ``x_hat = pre_whitener * (pca_mean + b_i * s_i(t))``.
    """
    X = fit_raw.get_data(picks=ica.ch_names)
    n = ica.n_components_
    S = ica.get_sources(fit_raw).get_data()[:n]                          # (n, T)
    B = ica.pca_components_[:n].T @ ica.mixing_matrix_                     # (C, n)
    w = np.asarray(ica.pre_whitener_).reshape(-1, 1)                      # MNE: whitened = X / w, then - pca_mean
    C = w * B                                                              # channel-space maps (volts per source unit)
    Y = X - w * np.asarray(ica.pca_mean_).reshape(-1, 1)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Cc = C - C.mean(axis=0, keepdims=True)
    nch = X.shape[0]
    base = (Yc ** 2).sum(axis=0).mean() / nch
    cross = ((Cc.T @ Yc) * S).mean(axis=1) / nch
    quad = (Cc ** 2).sum(axis=0) * (S ** 2).mean(axis=1) / nch
    resid = base - 2 * cross + quad
    orig = X.var(axis=0).mean()
    return 1.0 - resid / orig


def run_ica(hp_raw, seg: SegmentInfo, good_names: Sequence[str], cfg: ICAConfig, seed: int,
            montage: str = "biosemi64", n_threads: int | None = None) -> ICAResult:
    """Fit data -> ICA -> ICLabel -> exclusions, with QC numbers (``ICAResult.fit`` keeps the fit data)."""
    t0 = time.perf_counter()
    fit = ica_fit_data(hp_raw, seg, good_names, cfg, montage)
    if fit.n_components < 2:
        raise ValueError(f"too little data for ICA ({fit.raw.n_times} samples, rank {fit.rank})")
    ica, conv = fit_ica(fit, cfg, seed)
    t_fit = time.perf_counter() - t0
    probs = iclabel_probabilities(fit.raw, ica, n_threads=n_threads)
    excl, by = select_exclusions(probs, cfg.iclabel_rule, cfg.iclabel_exclude)
    ica.exclude = list(excl)
    pca_var = np.asarray(ica.pca_explained_variance_, float)
    comp_ev = component_explained_variance(ica, fit.raw)
    excl_ev = (float(ica.get_explained_variance_ratio(fit.raw, components=excl, ch_type="eeg")["eeg"])
               if excl else 0.0)
    top = np.argmax(probs, axis=1)
    qc = {
        "method": cfg.method, "fit_params": dict(cfg.fit_params), "n_components": int(ica.n_components_),
        "rank": fit.rank, "n_good_channels": len(good_names), "n_iter": int(getattr(ica, "n_iter_", -1)),
        "convergence_warnings": conv, "fit_sfreq": float(fit.raw.info["sfreq"]), "fit_decimate": fit.decimate,
        "fit_samples": int(fit.raw.n_times), "fit_seconds_of_data": fit.raw.n_times / fit.raw.info["sfreq"],
        "valid_samples": fit.n_valid_samples, "fit_chunks": fit.n_chunks, "fit_rejected_chunks": fit.n_rejected_chunks,
        "fit_rejected_fraction": fit.n_rejected_chunks / fit.n_chunks if fit.n_chunks else 0.0,
        "pca_variance_retained": float(pca_var[: ica.n_components_].sum() / pca_var.sum()) if pca_var.sum() else None,
        "explained_variance_all_components": float(ica.get_explained_variance_ratio(
            fit.raw, components=None, ch_type="eeg")["eeg"]),
        "explained_variance_excluded": excl_ev,
        "excluded": excl, "n_excluded": len(excl),
        "excluded_by_label": {k: len(v) for k, v in by.items()},
        "label_counts": {c: int((top == i).sum()) for i, c in enumerate(ICLABEL_CLASSES)},
        "rule": cfg.iclabel_rule, "thresholds": dict(cfg.iclabel_exclude),
        "component_explained_variance": [float(v) for v in comp_ev],
        "seconds_fit": round(t_fit, 2), "seconds_total": round(time.perf_counter() - t0, 2),
    }
    return ICAResult(ica=ica, probabilities=probs, excluded=excl, excluded_by_label=by, qc=qc, fit=fit)
