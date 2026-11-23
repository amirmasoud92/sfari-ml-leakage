"""Paradigm-specific evoked indices and behaviour, one recording at a time, with the data authors' published
definitions (Methods, Reliability and validity of candidate markers): 40- and 27-Hz ASSR inter-trial phase
coherence and transient response (180-250 ms), AVSRT multisensory ERP index, race-model gain and reaction times,
FAST N170 latency and face-inversion lateralisation, the illusory-contour effect and cue-locked alpha modulation in
Beep-Flash.

Every index needs at least 20 artefact-free trials per condition entering it; otherwise it is missing. Evoked
indices use the average reference except the ASSR transient response (TP7/TP8 reference, as published). Behaviour
uses all trials.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config.loader import Config
from features.signals import CleanRecording

MIN_TRIALS = 20


def _mean_window(E: np.ndarray, t: np.ndarray, w) -> np.ndarray:
    m = (t >= w[0]) & (t <= w[1])
    return E[..., m].mean(axis=-1)


def _itpc(E: np.ndarray, t: np.ndarray, sf: float, band, window) -> float:
    """Mean over band frequencies (1-Hz steps) and channels of |mean_trials exp(i phase)| (Hann-tapered DFT)."""
    m = (t >= window[0]) & (t <= window[1])
    x = E[:, :, m] * np.hanning(int(m.sum()))
    tt = t[m]
    vals = []
    for f in np.arange(band[0], band[1] + 1e-9, 1.0):
        z = (x * np.exp(-2j * np.pi * f * tt)).sum(axis=2)
        vals.append(np.abs((z / np.abs(z)).mean(axis=0)).mean())
    return float(np.mean(vals))


def _responses(r: CleanRecording) -> np.ndarray:
    return np.sort(r.events.loc[r.events["trial_type"] == "Response_button", "sample"].to_numpy(np.int64))


def _behaviour(r: CleanRecording, targets: np.ndarray, nontargets: np.ndarray, cfg: Config) -> dict:
    lo, hi = cfg.features.evoked["rt_window_s"]
    sf = r.sfreq
    resp = _responses(r)
    rt = np.full(targets.size, np.nan)
    for i, s in enumerate(targets):
        k = np.searchsorted(resp, s + lo * sf)
        if k < resp.size and resp[k] <= s + hi * sf:
            rt[i] = (resp[k] - s) / sf
    out = {
        "hit_rate": float(np.isfinite(rt).mean()) if rt.size else np.nan,
        "rt_mean": float(np.nanmean(rt)) if np.isfinite(rt).any() else np.nan,
    }
    out["rt_icv"] = float(np.nanstd(rt) / out["rt_mean"]) if np.isfinite(rt).sum() > 2 else np.nan
    if nontargets.size:
        # a response not within 1,500 ms after a target is a false alarm. Responses outside
        # the task (before the first stimulus or after the last one + the window) are not counted. Rate: false
        # alarms per non-target stimulus.
        stim = np.sort(np.concatenate([targets, nontargets]))
        r_in = resp[(resp >= stim[0]) & (resp <= stim[-1] + hi * sf)]
        tg = np.sort(targets)
        k = np.searchsorted(tg, r_in, side="right") - 1
        near = (
            (k >= 0) & (r_in - tg[np.clip(k, 0, None)] <= hi * sf) if tg.size else np.zeros(r_in.size, bool)
        )
        out["fa_rate"] = float((~near).sum() / nontargets.size)
    return out


def assr(r: CleanRecording, cfg: Config) -> dict:
    e = cfg.features.evoked["assr"]
    chans, ref = list(e["channels"]), list(e["reference"])
    out = {}
    for freq in ("27", "40"):
        s = r.trials(f"^{freq}_Hz_Standard$")["sample"].to_numpy()
        E, t = r.epochs(s, -0.2, 0.8, chans + ref)
        if len(E) < MIN_TRIALS:
            continue
        B = E[:, : len(chans)] - E[:, len(chans) :].mean(axis=1, keepdims=True)
        B = B - _mean_window(B, t, e["baseline_s"])[..., None]
        out[f"broadband_180_250_{freq}hz"] = float(
            _mean_window(B.mean(0), t, e["broadband_window_s"]).mean() * 1e6
        )
        out[f"itpc_{freq}hz"] = _itpc(
            E[:, : len(chans)], t, r.sfreq, e["itpc_bands_hz"][freq], e["itpc_window_s"]
        )
    ev = r.events
    tg = ev.loc[ev.trial_type.str.match(r"^(27|40)_Hz_Oddball$"), "sample"].to_numpy(np.int64)
    nt = ev.loc[ev.trial_type.str.match(r"^(27|40)_Hz_Standard$"), "sample"].to_numpy(np.int64)
    out.update({f"beh_{k}": v for k, v in _behaviour(r, tg, nt, cfg).items()})
    return out


def _race_gain(rt: dict[str, np.ndarray]) -> float:
    """Area of positive violation of the Raab bound F_A + F_V - F_A F_V over RT quantiles 0.05-0.95."""
    allrt = np.concatenate([v for v in rt.values() if v.size])
    if any(v.size < MIN_TRIALS for v in rt.values()):
        return np.nan
    qs = np.quantile(allrt, np.arange(0.05, 0.951, 0.05))

    def cdf(x, q):
        return np.searchsorted(np.sort(x), q, side="right") / x.size

    fa, fv, fav = cdf(rt["A"], qs), cdf(rt["V"], qs), cdf(rt["AV"], qs)
    viol = fav - (fa + fv - fa * fv)
    return float(np.clip(viol, 0, None).mean())


def avsrt(r: CleanRecording, cfg: Config) -> dict:
    e = cfg.features.evoked["avsrt_msi"]
    lo, hi = cfg.features.evoked["rt_window_s"]
    sf = r.sfreq
    ev = r.trials(artefact_free=False).sort_values("sample")
    resp = _responses(r)
    on = ev["sample"].to_numpy(np.int64)
    cond = (
        ev["trial_type"].map({"Auditory_stim": "A", "Visual_stim": "V", "Audiovisual_stim": "AV"}).to_numpy()
    )
    rt = np.full(on.size, np.nan)
    for i, s in enumerate(on):
        stop = on[i + 1] if i + 1 < on.size else s + int(hi * sf)
        k = np.flatnonzero((resp > s) & (resp < stop))
        if k.size and lo * sf <= resp[k[0]] - s <= hi * sf and k.size <= 2:
            rt[i] = (resp[k[0]] - s) / sf
    out = {
        "beh_hit_rate": float(np.isfinite(rt).mean()),
        "beh_rt_mean": float(np.nanmean(rt)),
        "beh_rt_icv": float(np.nanstd(rt) / np.nanmean(rt)),
    }
    prev = np.concatenate([[None], cond[:-1]])
    repeat = cond == prev
    out["race_model_gain"] = _race_gain(
        {c: rt[(cond == c) & repeat & np.isfinite(rt)] for c in ("A", "V", "AV")}
    )
    sw = []
    for c in ("A", "V"):
        rep = rt[(cond == c) & repeat]
        swi = rt[(cond == c) & ~repeat & (prev != "AV") & (prev != None)]  # noqa: E711 - object array
        if np.isfinite(rep).sum() >= MIN_TRIALS and np.isfinite(swi).sum() >= MIN_TRIALS:
            sw.append(np.nanmean(swi) - np.nanmean(rep))
    out["switch_cost_s"] = float(np.mean(sw)) if sw else np.nan
    af = set(r.trials()["sample"].tolist())
    erp = {}
    for c in ("A", "V", "AV"):
        s = on[(cond == c) & np.isfinite(rt)]
        s = np.array([x for x in s if x in af])
        E, t = r.epochs(s, -0.1, 0.4, list(e["channels"]), tuple(e["baseline_s"]))
        if len(E) < MIN_TRIALS:
            return out
        erp[c] = float(_mean_window(E.mean(0), t, e["window_s"]).mean() * 1e6)
    out["erp_msi_145_165"] = erp["AV"] - (erp["A"] + erp["V"])
    return out


def fast(r: CleanRecording, cfg: Config) -> dict:
    e = cfg.features.evoked["fast_n170"]
    L, R = e["channels"]["left"][0], e["channels"]["right"][0]
    out = {}
    erps = {}
    for c in ("upright", "inverted"):
        s = r.trials(f"^Face_{c}$")["sample"].to_numpy()
        E, t = r.epochs(s, -0.2, 0.5, [L, R], tuple(e["baseline_s"]))
        if len(E) < MIN_TRIALS:
            erps = {}
            break
        erps[c] = (E.mean(0), t)
    if erps:
        up, t = erps["upright"]
        w = (t >= e["latency_window_s"][0]) & (t <= e["latency_window_s"][1])
        out["n170_latency_s"] = float(t[w][np.argmin(up.mean(0)[w])])
        amp = {c: _mean_window(erps[c][0], t, e["amplitude_window_s"]) * 1e6 for c in erps}
        fie = amp["inverted"] - amp["upright"]
        out.update(
            {
                "fie_left_uv": float(fie[0]),
                "fie_right_uv": float(fie[1]),
                "fie_lateralisation_uv": float(fie[1] - fie[0]),
            }
        )
    ev = r.events
    tg = ev.loc[ev.trial_type.str.endswith("_shadow"), "sample"].to_numpy(np.int64)
    nt = ev.loc[ev.trial_type.str.match(r"^(Face|Object)_(upright|inverted)$"), "sample"].to_numpy(np.int64)
    out.update({f"beh_{k}": v for k, v in _behaviour(r, tg, nt, cfg).items()})
    return out


def ic(r: CleanRecording, cfg: Config) -> dict:
    e = cfg.features.evoked["ic_effect"]
    out = {}
    erp = {}
    for c, tt in (("ic", "Illusory_contours"), ("nc", "Non_Illusory_contours")):
        s = r.trials(f"^{tt}$")["sample"].to_numpy()
        E, t = r.epochs(s, -0.1, 0.4, list(e["channels"]), tuple(e["baseline_s"]))
        if len(E) < MIN_TRIALS:
            erp = {}
            break
        erp[c] = E.mean(0).mean(0)
    if erp:
        d = (erp["ic"] - erp["nc"]) * 1e6
        w = (t >= e["window_s"][0]) & (t <= e["window_s"][1])
        k = int(np.argmin(d[w]))
        out.update({"ic_effect_uv": float(d[w][k]), "ic_effect_latency_s": float(t[w][k])})
    ev = r.events
    tg = ev.loc[ev.trial_type == "Dot_flip", "sample"].to_numpy(np.int64)
    nt = r.trials(artefact_free=False)["sample"].to_numpy(
        np.int64
    )  # contour stimuli are the non-targets
    out.update({f"beh_{k}": v for k, v in _behaviour(r, tg, nt, cfg).items()})
    return out


_BF = re.compile(r"^(Visual|Auditory)_Cue_trial_(Visual|Auditory)_S2_(.*)$")


def _bf_attended_target(tt: str) -> bool | None:
    """Whether a Beep-Flash S2 event is a target in the cued modality (None for non-S2 events)."""
    m = _BF.match(tt)
    if not m:
        return None
    cue, mod, rest = m.groups()
    if cue != mod:
        return False
    if rest.startswith("Unisensory"):
        return rest.endswith("_Target") and "Non" not in rest
    if "Visual_and_Auditory_Target" in rest:
        return True
    return f"{mod}_Target" in rest


def beepflash(r: CleanRecording, cfg: Config) -> dict:
    e = cfg.features.evoked["beepflash_alpha"]
    out = {}
    pw = {}
    a0, b0 = (int(round(x * r.sfreq)) for x in e["cue_window_s"])
    good = r.good_mask
    for c in ("Auditory", "Visual"):
        s = r.trials(f"^{c}_Cue$")["sample"].to_numpy()
        # the artefact-free flag covers -0.2-0.8 s; the alpha window must be clean too
        s = np.array([k for k in s if k + b0 < good.size and good[k + a0 : k + b0 + 1].all()], np.int64)
        E, t = r.epochs(s, 0.0, 1.0, list(e["channels"]))
        if len(E) < MIN_TRIALS:
            pw = {}
            break
        m = (t >= e["cue_window_s"][0]) & (t <= e["cue_window_s"][1])
        x = E[:, :, m] * np.hanning(int(m.sum()))
        f = np.fft.rfftfreq(int(m.sum()), 1 / r.sfreq)
        P = np.abs(np.fft.rfft(x, axis=2)) ** 2
        b = (f >= e["band_hz"][0]) & (f <= e["band_hz"][1])
        pw[c] = float(P[:, :, b].mean())
    if pw:
        out["alpha_modulation_index"] = (pw["Auditory"] - pw["Visual"]) / (pw["Auditory"] + pw["Visual"])
    ev = r.events
    att = ev["trial_type"].map(_bf_attended_target)
    tg = ev.loc[att == True, "sample"].to_numpy(np.int64)  # noqa: E712 - None/False/True column
    nt = ev.loc[att == False, "sample"].to_numpy(np.int64)  # noqa: E712
    out.update({f"beh_{k}": v for k, v in _behaviour(r, tg, nt, cfg).items()})
    return out


EVOKED = {"ASSR": assr, "AVSRT": avsrt, "FAST": fast, "IC": ic, "Beepflash": beepflash}


def recording_evoked(cfg: Config, child: str, paradigm: str, variant: str = "primary") -> pd.DataFrame:
    """Tidy table (child, paradigm, block=-1, feature, value) of one recording's evoked indices and behaviour."""
    fn = EVOKED.get(paradigm)
    r = CleanRecording(cfg, child, paradigm, variant)
    if fn is None or not r.paths["qc"].is_file() or not r.usable:
        return pd.DataFrame(columns=["child", "paradigm", "block", "feature", "value"])
    vals = fn(r, cfg)
    return pd.DataFrame(
        {
            "child": child,
            "paradigm": paradigm,
            "block": -1,
            "feature": list(vals),
            "value": [float(v) for v in vals.values()],
        }
    )
