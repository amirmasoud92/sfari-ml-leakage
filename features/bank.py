"""The feature bank (Methods, Preprocessing and features; Additional file 1, Supplementary Methods S1).

Steps: (1) per-recording spectral pass (:mod:`features.spectra`); (2) label-free choice of the specparam aperiodic
mode on paradigm-level spectra; (3) block-level specparam (aperiodic exponent and offset, periodic alpha and beta
power, alpha centre frequency); (4) Welch log power in 1-Hz bins from 2 to 34 Hz at 64 channels (2,112 features),
debiased wPLI and coherence, and theta, alpha, beta and broadband covariances projected to the tangent space at the
Riemannian mean of all blocks (one reference per band); (5) evoked indices and behaviour; (6) assembly. The external
electrodes form a separate family (spectral and covariance features of EXG1-4).

Storage: one wide parquet per family, ``<derivatives>/features/<variant>/bank/<family>.parquet``, one row per child
x paradigm x block (``block = -1`` for recording-level families) and one float32 column per feature, plus
``features_meta.parquet`` (``family, feature, channel, channel_b, band, type``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config.loader import Config
from features.evoked import recording_evoked
from features.signals import bin_log_power, fit_specparam, largest_peak, periodic_band
from features.spectra import recording_pass
from utils.base import FeatureFamily
from utils.logging_utils import get_logger

log = get_logger(__name__)

SHARED_FAMILIES = (
    "spectral_logpow",
    "aperiodic",
    "periodic",
    "tangent_theta",
    "tangent_alpha",
    "tangent_beta",
    "wpli_theta",
    "wpli_alpha",
    "wpli_beta",
)
FLOOR_FAMILIES = ("broadband_cov", "exg_only")
CONTROL_FAMILIES = ("coherence_theta", "coherence_alpha", "coherence_beta")


class _BankFamily(FeatureFamily):
    """A family whose values are produced by :class:`FeatureBank`; ``compute`` returns its tidy rows for one
    recording from the bank (the bank computes all families of a recording in one pass)."""

    families: tuple[str, ...] = ()

    def compute(self, child, paradigm, data, blocks):  # noqa: D401 - see class docstring
        bank: FeatureBank = data
        frames = [bank.tidy(f, child=child, paradigm=paradigm) for f in self.families]
        meta = bank.meta()
        return pd.concat(frames, ignore_index=True), meta[meta.family.isin(self.families)]


class SpectralParam(_BankFamily):
    families = ("spectral_logpow", "aperiodic", "periodic", "alpha_cf")


class TangentSpaceCov(_BankFamily):
    families = ("tangent_theta", "tangent_alpha", "tangent_beta")


class WPLI(_BankFamily):
    families = ("wpli_theta", "wpli_alpha", "wpli_beta", *CONTROL_FAMILIES)


class BroadbandCovFloor(_BankFamily):
    families = ("broadband_cov",)


class ExgOnly(_BankFamily):
    families = ("exg_only",)


class EvokedIndices(_BankFamily):
    shared = False
    families = ("evoked",)


class Behaviour(_BankFamily):
    shared = False
    families = ("behaviour",)


def _load(p: Path) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _specparam_rows(cfg: Config, p: Path, mode: str, child: str, paradigm: str) -> dict:
    """Block-level aperiodic / periodic / alpha CF for EEG, plus EXG logpow and aperiodic (one recording)."""
    z = _load(p)
    sp = cfg.features.spectral
    f = z["freqs"]
    names = [str(x) for x in z["eeg_names"]]
    out = {
        "aperiodic": [],
        "periodic": [],
        "alpha_cf": [],
        "exg_only": [],
        "fit_r2": [],
        "n_fail": 0,
        "n_fit": 0,
    }
    for i, blk in enumerate(z["block"]):
        fits = fit_specparam(f, z["psd"][i].astype(float), cfg, mode)
        ap, pe, cf, r2 = {}, {}, {}, []
        for c, ft in zip(names, fits, strict=True):
            out["n_fit"] += 1
            if ft is None or ft.r2 < sp.specparam.min_r2:
                out["n_fail"] += 1
                ap[f"{c}_exponent"] = ap[f"{c}_offset"] = np.nan
                pe[f"{c}_alpha"] = pe[f"{c}_beta"] = cf[f"{c}_alphacf"] = np.nan
                continue
            r2.append(ft.r2)
            ap[f"{c}_exponent"], ap[f"{c}_offset"] = ft.exponent, ft.offset
            row = z["psd"][i, names.index(c)].astype(float)
            pe[f"{c}_alpha"] = periodic_band(f, row, ft, tuple(sp.alpha_band))
            pe[f"{c}_beta"] = periodic_band(f, row, ft, tuple(sp.beta_band))
            cf[f"{c}_alphacf"] = largest_peak(ft, tuple(sp.alpha_band))[0]
        key = {"child": child, "paradigm": paradigm, "block": int(blk)}
        out["aperiodic"].append({**key, **ap})
        out["periodic"].append({**key, **pe})
        out["alpha_cf"].append({**key, **cf})
        out["fit_r2"].append(float(np.median(r2)) if r2 else np.nan)
        if "exg_psd" in z:
            enames = [str(x) for x in z["exg_names"]]
            centres, lp = bin_log_power(
                f, z["exg_psd"][i].astype(float), sp.fmin, sp.fmax - 0.5, sp.logpow_bin_hz
            )
            ex = {f"{c}_{int(cc)}Hz": lp[k, j] for k, c in enumerate(enames) for j, cc in enumerate(centres)}
            for c, ft in zip(enames, fit_specparam(f, z["exg_psd"][i].astype(float), cfg, mode), strict=True):
                ok = ft is not None and ft.r2 >= sp.specparam.min_r2
                ex[f"{c}_exponent"] = ft.exponent if ok else np.nan
                ex[f"{c}_offset"] = ft.offset if ok else np.nan
            out["exg_only"].append({**key, **ex})
    return out


def _mode_fit(cfg: Config, p: Path) -> tuple[list[float], list[float]]:
    """R^2 of fixed and knee fits on the block-averaged spectrum of every channel (label-free mode choice)."""
    z = _load(p)
    P = z["psd"].astype(float).mean(axis=0)
    fx = fit_specparam(z["freqs"], P, cfg, "fixed")
    kn = fit_specparam(z["freqs"], P, cfg, "knee")
    return ([a.r2 if a else np.nan for a in fx], [b.r2 if b else np.nan for b in kn])


class FeatureBank:
    """Builds and reads the feature bank (module docstring)."""

    def __init__(self, cfg: Config, variant: str = "primary", block_mode: str = "primary") -> None:
        from features.spectra import bank_tag

        self.cfg = cfg
        self.variant = variant
        self.block_mode = block_mode
        self.root = cfg.path("derivatives") / "features" / bank_tag(variant, block_mode) / "bank"

    def build(self, children: list[str], paradigms: list[str] | None = None, force: bool = False) -> dict:
        cfg = self.cfg
        pars = paradigms or list(cfg.dataset.seated_paradigms)
        pairs = [(c, p) for c in children for p in pars]
        nj = cfg.runtime.n_jobs
        self.root.mkdir(parents=True, exist_ok=True)

        log.info("features 1/5: spectral pass on %d recordings", len(pairs))
        paths = Parallel(n_jobs=nj)(
            delayed(recording_pass)(cfg, c, p, self.variant, force, self.block_mode) for c, p in pairs
        )
        avail = [(c, p, q) for (c, p), q in zip(pairs, paths, strict=True) if q is not None]
        log.info("  %d recordings with spectra", len(avail))

        log.info("features 2/5: specparam mode choice (label-free)")
        mode, mode_info = self._choose_mode(avail)

        log.info("features 3/5: block-level specparam (%s)", mode)
        res = Parallel(n_jobs=nj)(delayed(_specparam_rows)(cfg, q, mode, c, p) for c, p, q in avail)
        n_fit = sum(r["n_fit"] for r in res)
        n_fail = sum(r["n_fail"] for r in res)
        for fam in ("aperiodic", "periodic", "alpha_cf", "exg_only"):
            rows = [x for r in res for x in r[fam]]
            self._write(fam, pd.DataFrame(rows))

        log.info("features 4/5: log power, connectivity, tangent space")
        tang = self._spectral_and_tangent(avail)

        log.info("features 5/5: evoked indices and behaviour")
        ev = pd.concat(
            Parallel(n_jobs=nj)(delayed(recording_evoked)(cfg, c, p, self.variant) for c, p in pairs),
            ignore_index=True,
        )
        beh = ev[ev.feature.str.startswith("beh_")]
        evo = ev[~ev.feature.str.startswith("beh_")]
        for fam, d in (("evoked", evo), ("behaviour", beh)):
            if d.empty:  # e.g. the motor standing bank: no evoked indices
                log.info("  %s: no rows; not written", fam)
                continue
            w = (
                d.assign(feature=d.paradigm + "__" + d.feature)
                .pivot_table(index=["child", "paradigm", "block"], columns="feature", values="value")
                .reset_index()
            )
            self._write(fam, w)

        summary = {
            "specparam_mode": mode,
            "mode_choice": mode_info,
            "n_recordings": len(avail),
            "specparam_fits": n_fit,
            "specparam_dropped_fraction": n_fail / max(n_fit, 1),
            **tang,
        }
        self._write_meta()
        (self.root / "bank_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _choose_mode(self, avail) -> tuple[str, dict]:
        sp = self.cfg.features.spectral.specparam
        if sp.aperiodic_mode != "auto":
            return sp.aperiodic_mode, {"rule": "fixed by config"}
        r = Parallel(n_jobs=self.cfg.runtime.n_jobs)(delayed(_mode_fit)(self.cfg, q) for _, _, q in avail)
        fx = np.concatenate([np.array(a) for a, _ in r])
        kn = np.concatenate([np.array(b) for _, b in r])
        gain = float(np.nanmedian(kn - fx))
        mode = "knee" if gain >= sp.knee_rule["min_median_r2_gain"] else "fixed"
        info = {
            "median_r2_fixed": float(np.nanmedian(fx)),
            "median_r2_knee": float(np.nanmedian(kn)),
            "median_gain": gain,
            "threshold": sp.knee_rule["min_median_r2_gain"],
            "n_spectra": int(fx.size),
        }
        log.info("  specparam mode %s (%s)", mode, info)
        return mode, info

    def _spectral_and_tangent(self, avail) -> dict:
        from pyriemann.utils.mean import mean_riemann
        from pyriemann.utils.tangentspace import tangent_space

        cfg = self.cfg
        sp = cfg.features.spectral
        logpow, keys, covs, conn = [], [], {}, {}
        exg_cov, exg_keys = [], []
        names = None
        shrink = {}
        for c, p, q in avail:
            z = _load(q)
            names = [str(x) for x in z["eeg_names"]]
            for i, blk in enumerate(z["block"]):
                key = {"child": c, "paradigm": p, "block": int(blk)}
                keys.append(key)
                centres, lp = bin_log_power(
                    z["freqs"], z["psd"][i].astype(float), sp.fmin, sp.fmax - 0.5, sp.logpow_bin_hz
                )
                logpow.append(lp.ravel())
                for b in ("theta", "alpha", "beta", "broadband"):
                    covs.setdefault(b, []).append(z[f"cov_{b}"][i].astype(float))
                    shrink.setdefault(b, []).append(float(z[f"shrinkage_{b}"][i]))
                for b in ("theta", "alpha", "beta"):
                    conn.setdefault(f"wpli_{b}", []).append(z[f"wpli_{b}"][i])
                    conn.setdefault(f"coherence_{b}", []).append(z[f"coh_{b}"][i])
                if "exg_cov" in z:
                    exg_cov.append(z["exg_cov"][i].astype(float))
                    exg_keys.append(key)
        kdf = pd.DataFrame(keys)
        cols = [f"{ch}_{int(cc)}Hz" for ch in names for cc in centres]
        self._write("spectral_logpow", pd.concat([kdf, pd.DataFrame(np.array(logpow), columns=cols)], axis=1))
        iu = np.triu_indices(len(names), 1)
        pair_cols = [f"{names[a]}-{names[b]}" for a, b in zip(*iu, strict=True)]
        for fam, rows in conn.items():
            self._write(fam, pd.concat([kdf, pd.DataFrame(np.array(rows), columns=pair_cols)], axis=1))
        ref_dir = self.root / "tangent_reference"
        ref_dir.mkdir(exist_ok=True)
        it, jt = np.triu_indices(len(names))
        tcols = [f"{names[a]}~{names[b]}" for a, b in zip(it, jt, strict=True)]
        info = {}
        for b, mats in covs.items():
            C = np.stack(mats)
            ref = mean_riemann(C)
            np.save(ref_dir / f"{b}.npy", ref)
            T = tangent_space(C, ref)
            fam = "broadband_cov" if b == "broadband" else f"tangent_{b}"
            self._write(fam, pd.concat([kdf, pd.DataFrame(T.astype(np.float32), columns=tcols)], axis=1))
            ev = np.linalg.eigvalsh(C)
            info[f"{fam}_min_eig_ratio_median"] = float(np.median(ev[:, 0] / ev[:, -1]))
            info[f"{fam}_shrinkage_median"] = float(np.median(shrink[b]))
        if exg_cov:
            E = np.stack(exg_cov)
            ref = mean_riemann(E)
            np.save(ref_dir / "exg_broadband.npy", ref)
            T = tangent_space(E, ref)
            k = int(np.sqrt(2 * T.shape[1] + 0.25) - 0.5)
            ie, je = np.triu_indices(k)
            tc = pd.DataFrame(
                T.astype(np.float32), columns=[f"EXG{a + 1}~EXG{b + 1}" for a, b in zip(ie, je, strict=True)]
            )
            ex = pd.read_parquet(self.root / "exg_only.parquet")
            ex = ex.merge(pd.concat([pd.DataFrame(exg_keys), tc], axis=1), on=["child", "paradigm", "block"])
            self._write("exg_only", ex)
        info["tangent_reference_n_matrices"] = int(len(keys))
        return info

    def _write(self, family: str, df: pd.DataFrame) -> None:
        idx = ["child", "paradigm", "block"]
        if df.empty:  # e.g. exg_only for the motor recordings, which have no external channels
            log.info("  %s: no rows; not written", family)
            return
        df = df.sort_values(idx).reset_index(drop=True)
        df[[c for c in df.columns if c not in idx]] = df[[c for c in df.columns if c not in idx]].astype(
            np.float32
        )
        df.to_parquet(self.root / f"{family}.parquet", index=False)

    def _write_meta(self) -> None:
        rows = []
        for p in sorted(self.root.glob("*.parquet")):
            fam = p.stem
            if fam == "features_meta":
                continue
            import pyarrow.parquet as pq

            for col in pq.read_schema(p).names:
                if col in ("child", "paradigm", "block"):
                    continue
                rows.append({"family": fam, "feature": col, **_describe(fam, col)})
        pd.DataFrame(rows).to_parquet(self.root / "features_meta.parquet", index=False)

    def wide(self, family: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / f"{family}.parquet")

    def meta(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "features_meta.parquet")

    def tidy(self, family: str, child: str | None = None, paradigm: str | None = None) -> pd.DataFrame:
        w = self.wide(family)
        if child is not None:
            w = w[w.child == child]
        if paradigm is not None:
            w = w[w.paradigm == paradigm]
        return w.melt(id_vars=["child", "paradigm", "block"], var_name="feature", value_name="value")


def _describe(family: str, col: str) -> dict:
    band = {"theta": "theta", "alpha": "alpha", "beta": "beta"}
    b = next((v for k, v in band.items() if family.endswith(k)), None)
    if family in ("spectral_logpow",):
        ch, hz = col.rsplit("_", 1)
        return {"channel": ch, "channel_b": None, "band": hz, "type": "log10 power"}
    if family in ("aperiodic", "periodic", "alpha_cf"):
        ch, kind = col.rsplit("_", 1)
        return {
            "channel": ch,
            "channel_b": None,
            "band": kind if family == "periodic" else None,
            "type": kind,
        }
    if family.startswith(("wpli", "coherence")):
        a, c = col.split("-")
        return {"channel": a, "channel_b": c, "band": b, "type": family.split("_")[0]}
    if family.startswith("tangent") or family == "broadband_cov":
        a, c = col.split("~")
        return {"channel": a, "channel_b": c, "band": b or "broadband", "type": "tangent"}
    if family == "exg_only":
        if "~" in col:
            a, c = col.split("~")
            return {"channel": a, "channel_b": c, "band": "broadband", "type": "tangent"}
        ch, kind = col.rsplit("_", 1)
        return {"channel": ch, "channel_b": None, "band": kind, "type": "exg spectral"}
    par, name = col.split("__", 1) if "__" in col else (None, col)
    return {"channel": None, "channel_b": None, "band": None, "type": f"{family}:{par}"}


def main(variant: str = "primary", block_mode: str = "primary") -> dict:
    """Build the feature bank of every usable child from the preprocessed recordings of ``variant``.

    ``block_mode``: ``primary`` (event-defined blocks), ``alternative`` (equal-duration blocks; Table S8) or
    ``standing`` (the standing runs of the motor paradigm; cross-system identification).
    """
    from config.loader import load_config
    from utils.dataset import BidsDataset, Participants

    cfg = load_config()
    kids = Participants.load(BidsDataset.from_config(cfg), cfg).usable_ids
    pars = [cfg.dataset.motor_paradigm] if block_mode == "standing" else None
    return FeatureBank(cfg, variant, block_mode).build(kids, pars)


if __name__ == "__main__":
    main()
