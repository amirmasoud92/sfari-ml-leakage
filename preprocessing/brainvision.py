"""``BrainVisionPreprocessor``: the standing motor paradigm recorded with a mobile Brain Products system, used
only to test identification across amplifier and electrode systems (Additional file 1, Table S3; Fig. 2a).

Same steps as :class:`~preprocessing.biosemi.BiosemiPreprocessor` except: BrainVision loading (64 EEG channels, no
external channels, no BioSemi status joins: segments split at run boundaries only); one block per run of at least
``motor.min_block_s``; a more aggressive ASR cutoff (``motor.asr_cutoff``) for gait-related motion, calibrated on
the cleanest 60 s; walking runs left out of the bad-channel detection data. Retention: at least
``minimum_retained.rest_s`` of clean data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.loader import Config
from preprocessing.biosemi import Assembled, BiosemiPreprocessor
from utils.dataset import Recording


class BrainVisionPreprocessor(BiosemiPreprocessor):
    def __init__(
        self, cfg: Config, variant: str = "primary", asr: bool | None = None, diagnostics: bool = False
    ):
        p = cfg.preprocessing
        cfg_m = cfg.model_copy(
            update={
                "preprocessing": p.model_copy(
                    update={"asr": p.asr.model_copy(update={"cutoff": cfg.motor.asr_cutoff})}
                )
            }
        )
        super().__init__(cfg_m, variant=variant, asr=asr, diagnostics=diagnostics)

    def detection_exclusions(self, A: Assembled) -> np.ndarray:
        """Walking-aware: channel quality is judged on the standing runs; walking runs are left out of the
        PyPREP detection data (motion breaks the correlation criterion in far more than 1% of windows). If no
        standing run exists, all data are used."""
        b = A.blocks
        if b.empty or not (~b["walking"]).any():
            return np.zeros((0, 2), np.int64)
        w = b[b["walking"]]
        return w[["start_sample", "stop_sample"]].to_numpy(np.int64)

    def assemble(self, rec: Recording) -> Assembled:
        import mne

        names = list(mne.channels.make_standard_montage("biosemi64").ch_names)
        datas, joins, evs, blks, runs = [], [], [], [], []
        offset, sf = 0, None
        for rf in rec.runs:
            raw = mne.io.read_raw_brainvision(rf.eeg, preload=False, verbose="error")
            sf = float(raw.info["sfreq"])
            missing = [c for c in names if c not in raw.ch_names]
            if missing:
                raise ValueError(f"{rf.eeg.name}: EEG channels missing {missing[:5]}")
            x = raw.get_data(picks=names)
            n = x.shape[1]
            if offset > 0:
                joins.append(offset)
            ev = rf.read_events()
            e = ev.copy()
            e["sample_rec"] = np.rint(e["onset"].to_numpy(float) * sf).astype(np.int64) + offset
            e["run"] = rf.run
            evs.append(e)
            dur = n / sf
            if dur >= self.cfg.motor.min_block_s:
                blks.append(
                    {
                        "child": rec.subject,
                        "paradigm": rec.paradigm,
                        "run": rf.run,
                        "block": len(blks),
                        "start_s": 0.0,
                        "stop_s": dur,
                        "duration_s": dur,
                        "n_trials": 0,
                        "n_responses": int((ev["trial_type"] == "button_press").sum()),
                        "walking": bool(dur >= self.cfg.motor.walking_min_s),
                        "start_sample": offset,
                        "stop_sample": offset + n,
                    }
                )
            runs.append({"run": rf.run, "file_start_sample": 0, "file_stop_sample": n, "offset": offset})
            datas.append(x)
            offset += n
        data = np.concatenate(datas, axis=1)
        return Assembled(
            data,
            float(sf),
            names,
            [],
            np.array(joins, np.int64),
            pd.concat(evs, ignore_index=True),
            pd.DataFrame(blks),
            runs,
        )
