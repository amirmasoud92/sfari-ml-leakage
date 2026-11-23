"""Robust average reference.

The reference signal is the per-sample mean of the *good* EEG channels (EEG channels not in ``info['bads']`` and
not in ``exclude``). It is subtracted from every EEG channel, bad channels included, so a bad channel that is
later interpolated is on the same reference. Non-EEG channels are never changed.

"Robust" refers to the reference set: bad channels are found by the PREP-style loop of
:mod:`preprocessing.bad_channels` (median-referenced first pass, then the mean of the not-yet-bad channels) and are
excluded from the mean. This differs from MNE's ``set_eeg_reference('average')``, which leaves bad channels
un-referenced.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def eeg_picks(info) -> np.ndarray:
    """Indices of every EEG channel (bads included)."""
    import mne

    return mne.pick_types(info, eeg=True, exclude=[])


def good_eeg_picks(info, exclude: Iterable[str] = ()) -> np.ndarray:
    """Indices of EEG channels not in ``info['bads']`` and not in ``exclude``."""
    bad = set(info["bads"]) | set(exclude)
    return np.array([i for i in eeg_picks(info) if info["ch_names"][i] not in bad], int)


def robust_average_reference(raw, exclude: Iterable[str] = (), *, copy: bool = False):
    """Re-reference the EEG channels of ``raw`` to the mean of its good EEG channels.

    Parameters
    ----------
    raw
        Preloaded Raw (modified in place unless ``copy``).
    exclude
        Extra channel names to leave out of the reference (in addition to ``info['bads']``).
    copy
        Work on a copy.

    Returns
    -------
    raw, ref
        The re-referenced Raw and the subtracted reference signal (volts).
    """
    from mne.io.constants import FIFF

    if copy:
        raw = raw.copy()
    good = good_eeg_picks(raw.info, exclude)
    if good.size == 0:
        raise ValueError("no good EEG channel left for the average reference")
    allp = eeg_picks(raw.info)
    ref = raw._data[good].mean(axis=0)
    raw._data[allp] -= ref[np.newaxis, :]
    with raw.info._unlock():
        raw.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON
    return raw, ref


def median_reference(data: np.ndarray) -> np.ndarray:
    """``data`` minus its per-sample median over channels (PREP's robust first-pass reference)."""
    return data - np.median(data, axis=0, keepdims=True)
