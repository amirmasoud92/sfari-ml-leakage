"""Cleaning of the analysis copy from its parts (:func:`clean_from_parts`):

1. robust average reference (mean of the good EEG channels, subtracted from every EEG channel), the reference the
   ICA was fitted on;
2. removal of the excluded independent components (good channels only);
3. spherical-spline interpolation of the bad EEG channels;
4. robust average reference again, over the originally good channels (a no-op on them; it puts interpolated
   channels on the same reference).

The sample axis is never changed.
"""

from __future__ import annotations


class PreprocessingOutputError(RuntimeError):
    """Raised when stored outputs are missing or inconsistent."""


def clean_from_parts(raw, ica=None, *, apply_ica: bool = True, interpolate: bool = True,
                     exclude: list[int] | None = None, origin: str = "auto"):
    """Clean a pre-ICA analysis copy in place (steps in the module docstring); returns ``raw``.

    ``raw.info['temp']`` receives ``{"interpolated": [...], "ica_excluded": [...],
    "ica_applied": bool, "reference_channels": [...]}``.
    """
    from preprocessing.reference import eeg_picks, good_eeg_picks, robust_average_reference

    eeg = [raw.ch_names[i] for i in eeg_picks(raw.info)]
    bad_eeg = [c for c in raw.info["bads"] if c in eeg]
    other_bads = [c for c in raw.info["bads"] if c not in eeg]
    good = [raw.ch_names[i] for i in good_eeg_picks(raw.info)]
    robust_average_reference(raw)
    excluded: list[int] = []
    if apply_ica and ica is not None:
        miss = sorted(set(ica.ch_names) - set(good))
        if miss:
            raise PreprocessingOutputError(f"ICA channels not among the good EEG channels: {miss}")
        excluded = sorted(int(i) for i in (ica.exclude if exclude is None else exclude))
        if excluded:
            # MNE's apply(exclude=...) removes the UNION of `exclude` and ica.exclude, so a smaller (vetoed) set must be
            # put into ica.exclude itself for the call
            saved = list(ica.exclude)
            ica.exclude = list(excluded)
            try:
                ica.apply(raw, verbose="error")
            finally:
                ica.exclude = saved
    interpolated: list[str] = []
    if interpolate and bad_eeg:
        raw.interpolate_bads(reset_bads=True, mode="accurate", origin=origin, verbose="error")
        interpolated = bad_eeg
        with raw.info._unlock():
            raw.info["bads"] = list(other_bads)      # non-EEG bads (e.g. a faulty EXG) stay marked
    robust_average_reference(raw, exclude=interpolated)
    with raw.info._unlock():
        raw.info["temp"] = {"interpolated": interpolated, "ica_excluded": excluded,
                            "ica_applied": bool(apply_ica and ica is not None), "reference_channels": good}
    return raw
