"""Convolutional networks: EEGNet and ShallowConvNet (braindecode) on 2-s segments (Methods, Other classifiers;
Fig. 3d; Table 2; Additional file 1, Supplementary Methods S3 and Table S9b).

Segments: up to 60 clean 2-s segments per child, sampled evenly over the six paradigms' blocks, decimated to 128 Hz
(the data are low-passed at 45 Hz) and z-scored per channel with training statistics. Training: AdamW (learning
rate 1e-3, weight decay 1e-3), 20 epochs, batch 64, seeded. Two evaluations:

- children kept whole: the child folds of :class:`~classification.splits.SplitFactory`; segment probabilities are
  averaged per child and the AUC is computed over children (child-bootstrap interval);
- segments split (leaky): segments of one child fall in both training and test folds; AUC over segments (interval
  from resampling children with all their segments).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from classification.splits import SplitFactory
from config.loader import Config
from features.signals import CleanRecording
from stats.inference import percentile_ci
from stats.rng import generator
from utils.logging_utils import get_logger

log = get_logger(__name__)


def child_segments(cfg: Config, child: str, paradigms: list[str], per_child: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_par = max(per_child // len(paradigms), 1)
    out = []
    for p in paradigms:
        r = CleanRecording(cfg, child, p)
        if not r.paths["qc"].is_file() or not r.usable:
            continue
        segs = [r.segments(b, None, 2.0, 2.0) for _, b in r.blocks.iterrows()]
        segs = [s for s in segs if len(s)]
        if not segs:
            continue
        S = np.concatenate(segs)
        k = rng.choice(len(S), min(per_par, len(S)), replace=False)
        out.append(S[k][:, :, ::2].astype(np.float32))  # 256 -> 128 Hz
    return np.concatenate(out) if out else np.zeros((0, 64, 256), np.float32)


def _model(name: str, n_chans: int, n_times: int):
    from braindecode.models import EEGNet, ShallowFBCSPNet

    if name == "eegnet":
        return EEGNet(n_chans=n_chans, n_outputs=2, n_times=n_times)
    return ShallowFBCSPNet(n_chans=n_chans, n_outputs=2, n_times=n_times, final_conv_length="auto")


def train_predict(name, Xtr, ytr, Xte, seed: int, epochs: int = 20, batch: int = 64) -> np.ndarray:
    import torch

    torch.manual_seed(seed)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-12
    Xtr = torch.tensor((Xtr - mu) / sd)
    Xte = torch.tensor((Xte - mu) / sd)
    y = torch.tensor(ytr, dtype=torch.long)
    net = _model(name, Xtr.shape[1], Xtr.shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = lossf(net(Xtr[idx]), y[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out = torch.cat([torch.softmax(net(Xte[i : i + 256]), 1)[:, 1] for i in range(0, len(Xte), 256)])
    return out.numpy()


def run_deep(
    cfg: Config,
    labels: pd.Series,
    paradigms: list[str],
    per_child: int = 60,
    models: tuple[str, ...] = ("eegnet", "shallow"),
    epochs: int = 20,
) -> pd.DataFrame:
    """``labels``: 1 = ASD, 0 = TD, index child."""
    import torch

    torch.set_num_threads(max(cfg.runtime.n_jobs, 1))
    kids = list(labels.index)
    _, seed = generator(cfg.project.seed, "deep")
    segs = {c: child_segments(cfg, c, paradigms, per_child, seed) for c in kids}
    kids = [c for c in kids if len(segs[c])]
    y = labels.loc[kids]
    rows = []
    for name in models:
        # child-level CV
        prob = pd.Series(np.nan, index=kids)
        for fold in SplitFactory(kids, cfg.project.seed, strata=y.to_numpy()).outer(
            cfg.prediction.outer_folds, 1
        ):
            Xtr = np.concatenate([segs[c] for c in fold.train])
            ytr = np.concatenate([[y[c]] * len(segs[c]) for c in fold.train])
            Xte = np.concatenate([segs[c] for c in fold.test])
            p = train_predict(name, Xtr, ytr, Xte, seed, epochs)
            k = 0
            for c in fold.test:
                prob[c] = float(p[k : k + len(segs[c])].mean())
                k += len(segs[c])
        auc = roc_auc_score(y, prob)
        rng = np.random.default_rng(seed)
        bs = []
        for _ in range(cfg.prediction.n_bootstrap):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y.iloc[i])) == 2:
                bs.append(roc_auc_score(y.iloc[i], prob.iloc[i]))
        lo, hi = percentile_ci(np.array(bs))
        rows.append(
            {
                "model": name,
                "evaluation": "child-level CV (child AUC)",
                "n": len(y),
                "auc": auc,
                "auc_lo": lo,
                "auc_hi": hi,
            }
        )
        # segment-level (leaky) CV
        X = np.concatenate([segs[c] for c in kids])
        ys = np.concatenate([[y[c]] * len(segs[c]) for c in kids])
        pp = np.zeros(len(ys))
        for tr, te in StratifiedKFold(cfg.prediction.outer_folds, shuffle=True, random_state=seed).split(
            X, ys
        ):
            pp[te] = train_predict(name, X[tr], ys[tr], X[te], seed, epochs)
        # segments of one child are not independent: the interval resamples children with all their segments
        owner = np.concatenate([[i] * len(segs[c]) for i, c in enumerate(kids)])
        idx = [np.flatnonzero(owner == i) for i in range(len(kids))]
        bs = []
        for _ in range(cfg.prediction.n_bootstrap):
            j = np.concatenate([idx[i] for i in rng.integers(0, len(kids), len(kids))])
            if len(np.unique(ys[j])) == 2:
                bs.append(roc_auc_score(ys[j], pp[j]))
        lo, hi = percentile_ci(np.array(bs))
        rows.append(
            {
                "model": name,
                "evaluation": "segment-level CV (leaky; segment AUC)",
                "n": len(ys),
                "auc": roc_auc_score(ys, pp),
                "auc_lo": lo,
                "auc_hi": hi,
            }
        )
        log.info("deep %s: %s", name, rows[-2:])
    return pd.DataFrame(rows)
