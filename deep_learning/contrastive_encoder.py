"""Subject-contrastive encoder trained without diagnostic labels (Methods, Other classifiers; Additional file 1,
Supplementary Methods S3 and Table S9c).

Data: up to ``segments_per_view`` clean 2-s segments per child and seated paradigm (a "view"), decimated to 128 Hz,
z-scored per channel with the training children's statistics. No group label or outcome is used in training.

Training (one encoder per outer fold): a ShallowConvNet trunk (braindecode ``ShallowFBCSPNet`` with
``embedding_dim`` = 64 outputs) and a two-layer projection head, normalised temperature-scaled cross-entropy
(NT-Xent) loss. A batch holds ``batch_children`` = 32 children; each contributes two segments from two different
paradigms, which form the positive pair, and every other segment in the batch is a negative; 1,500 steps. The
training set is every child with at least two kept seated paradigms except the fold's test children. The test
children are never seen.

Evaluation, on the autistic-versus-TD folds of the model ladder (same SplitFactory and seed, ``repeats`` x 5 folds):
- a child's representation is the mean over paradigms of its L2-normalised mean segment embedding; it enters the
  ladder's supervised step (z-scoring + L2 logistic or ridge regression, inner CV) as rung (e) does;
- cross-paradigm identification among the test children of each fold (z-scored within paradigm, Pearson r,
  30 ordered pairs), next to log power on the same children;
- the control is the same architecture with the same initial weights and no training.

Dimensional targets (SRS-2, MABC-2, FSIQ) reuse the autistic-versus-TD folds, so that each fold trains one encoder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from classification.model_ladder import Target, bootstrap_metrics, fit_predict
from classification.splits import SplitFactory
from config.loader import Config
from features.signals import CleanRecording
from identification.core import corr_matrix, correct_rows, paradigm_matrix, zscore_columns
from stats.inference import percentile_ci
from stats.rng import derive_seed, generator
from utils.base import Result
from utils.logging_utils import get_logger

log = get_logger(__name__)
MODELS = ("trained", "untrained")


def view_segments(cfg: Config, child: str, paradigm: str) -> np.ndarray:
    """``(k, 64, 256)`` float32 segments at 128 Hz (k <= segments_per_view), sampled without replacement."""
    r = CleanRecording(cfg, child, paradigm)
    if not r.usable:
        return np.zeros((0, 64, 256), np.float32)
    segs = [r.segments(b, None, 2.0, 2.0) for _, b in r.blocks.iterrows()]
    segs = [s for s in segs if len(s)]
    if not segs:
        return np.zeros((0, 64, 256), np.float32)
    S = np.concatenate(segs)
    rng = np.random.default_rng(derive_seed(cfg.project.seed, "encoder_segments", child, paradigm))
    k = rng.choice(len(S), min(cfg.encoder.segments_per_view, len(S)), replace=False)
    return S[np.sort(k)][:, :, ::2].astype(np.float32)  # 256 -> 128 Hz (data low-passed at 45 Hz)


def load_segments(cfg: Config, children: list[str], paradigms: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """child -> paradigm -> segments, for every kept recording (read in parallel; order-independent)."""
    jobs = [(c, p) for c in children for p in paradigms]
    out = Parallel(n_jobs=cfg.runtime.n_jobs)(delayed(view_segments)(cfg, c, p) for c, p in jobs)
    segs: dict[str, dict[str, np.ndarray]] = {}
    for (c, p), s in zip(jobs, out, strict=True):
        if len(s):
            segs.setdefault(c, {})[p] = s
    return segs


def _networks(cfg: Config, seed: int):
    import torch
    from braindecode.models import ShallowFBCSPNet

    e = cfg.encoder
    torch.manual_seed(seed)
    trunk = ShallowFBCSPNet(n_chans=64, n_outputs=e.embedding_dim, n_times=256, final_conv_length="auto")
    head = torch.nn.Sequential(
        torch.nn.Linear(e.embedding_dim, e.embedding_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(e.embedding_dim, e.projection_dim),
    )
    return trunk, head


def nt_xent(z1, z2, tau: float):
    """NT-Xent over 2B embeddings; positives are (i, i + B)."""
    import torch

    z = torch.nn.functional.normalize(torch.cat([z1, z2]), dim=1)
    s = z @ z.T / tau
    n = z1.shape[0]
    s.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return torch.nn.functional.cross_entropy(s, target)


def train_encoder(cfg: Config, segs: dict[str, dict[str, np.ndarray]], children: list[str], seed: int):
    """Contrastive training on ``children`` (each with >= 2 views); returns (trunk, mu, sd, loss per 50 steps)."""
    import torch

    e = cfg.encoder
    pool = [c for c in children if len(segs.get(c, {})) >= 2]
    allx = np.concatenate([s for c in pool for s in segs[c].values()])
    mu = allx.mean(axis=(0, 2), keepdims=True)[0]
    sd = allx.std(axis=(0, 2), keepdims=True)[0] + 1e-12
    del allx
    trunk, head = _networks(cfg, seed)
    params = list(trunk.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=e.lr, weight_decay=e.weight_decay)
    rng = np.random.default_rng(seed)
    B = min(e.batch_children, len(pool))
    losses, run = [], []
    trunk.train()
    head.train()
    for step in range(e.steps):
        kids = rng.choice(len(pool), B, replace=False)
        x1, x2 = [], []
        for i in kids:
            views = segs[pool[i]]
            p, q = rng.choice(list(views), 2, replace=False)
            x1.append(views[p][rng.integers(len(views[p]))])
            x2.append(views[q][rng.integers(len(views[q]))])
        a = (np.stack(x1) - mu) / sd
        b = (np.stack(x2) - mu) / sd
        a = a * rng.uniform(0.9, 1.1, (B, 1, 1)) + rng.normal(0, e.noise_sd, a.shape)
        b = b * rng.uniform(0.9, 1.1, (B, 1, 1)) + rng.normal(0, e.noise_sd, b.shape)
        ta = torch.tensor(a, dtype=torch.float32)
        tb = torch.tensor(b, dtype=torch.float32)
        opt.zero_grad()
        loss = nt_xent(head(trunk(ta)), head(trunk(tb)), e.temperature)
        loss.backward()
        opt.step()
        run.append(loss.item())
        if (step + 1) % 50 == 0:
            losses.append(float(np.mean(run)))
            run = []
    trunk.eval()
    return trunk, mu, sd, losses


def embed(trunk, segs: dict[str, dict[str, np.ndarray]], children: list[str], mu, sd) -> dict:
    """child -> paradigm -> L2-normalised mean of the L2-normalised segment embeddings."""
    import torch

    out: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for c in children:
            for p, s in segs.get(c, {}).items():
                z = trunk(torch.tensor((s - mu) / sd, dtype=torch.float32)).numpy()
                z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
                m = z.mean(0)
                out.setdefault(c, {})[p] = m / (np.linalg.norm(m) + 1e-12)
    return out


def child_matrix(E: dict, children: list[str]) -> np.ndarray:
    """Child representation: mean over the child's paradigms of its view embeddings."""
    return np.stack([np.mean(list(E[c].values()), axis=0) for c in children])


def fold_identification(views: dict[str, np.ndarray], paradigms: list[str]) -> np.ndarray:
    """Per child: fraction of the ordered paradigm pairs in which it is identified among the fold's children.
    ``views[p]`` is a child x feature matrix (same child order in every paradigm)."""
    hits, n = None, 0
    for p in paradigms:
        for q in paradigms:
            if p == q:
                continue
            R = corr_matrix(zscore_columns(views[p]), zscore_columns(views[q]))
            h = correct_rows(R).astype(float)
            hits = h if hits is None else hits + h
            n += 1
    return hits / n


def run_fold(cfg, fold, segs, pool, targets, logpow, paradigms, core_views):
    """Train one encoder on the pool minus the fold's test children; evaluate both models on the fold."""
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(max(1, cfg.runtime.threads_per_worker))
    test = list(fold.test)
    train = list(fold.train)
    seed = derive_seed(cfg.project.seed, "encoder", fold.repeat, fold.fold)
    with threadpool_limits(max(1, cfg.runtime.threads_per_worker)):
        trunk, mu, sd, losses = train_encoder(cfg, segs, [c for c in pool if c not in set(test)], seed)
        untrained, _ = _networks(cfg, seed)
        untrained.eval()
        rows, ident = [], []
        # identification among the test children that have every seated paradigm
        tid = [c for c in test if c in core_views]
        for name, net in (("trained", trunk), ("untrained", untrained)):
            E = embed(net, segs, train + test, mu, sd)
            Xtr, Xte = child_matrix(E, train), child_matrix(E, test)
            for t in targets:
                ok_tr = [i for i, c in enumerate(train) if c in t.y.index and pd.notna(t.y[c])]
                ok_te = [i for i, c in enumerate(test) if c in t.y.index and pd.notna(t.y[c])]
                ytr = t.y.loc[[train[i] for i in ok_tr]].to_numpy()
                kids_tr = tuple(train[i] for i in ok_tr)
                sf = SplitFactory(
                    list(kids_tr), cfg.project.seed, strata=t.strata.reindex(kids_tr).to_numpy()
                )
                strata = ytr if t.kind == "classification" else t.strata.reindex(kids_tr).to_numpy()
                inner = sf.inner(kids_tr, cfg.prediction.inner_folds, strata=strata)
                scoring = "roc_auc" if t.kind == "classification" else "neg_mean_squared_error"
                p = fit_predict(Xtr[ok_tr], ytr, Xte[ok_te], t.kind, cfg, inner, scoring)[0]
                rows += [
                    {
                        "model": name,
                        "target": t.name,
                        "repeat": fold.repeat,
                        "child": test[i],
                        "pred": float(v),
                    }
                    for i, v in zip(ok_te, p, strict=True)
                ]
            if len(tid) >= 3:
                hit = fold_identification({p: np.stack([E[c][p] for c in tid]) for p in paradigms}, paradigms)
                ident += [
                    {"model": name, "repeat": fold.repeat, "fold": fold.fold, "child": c, "hit": float(h)}
                    for c, h in zip(tid, hit, strict=True)
                ]
        if len(tid) >= 3:
            hit = fold_identification({p: logpow[p].loc[tid].to_numpy() for p in paradigms}, paradigms)
            ident += [
                {"model": "log power", "repeat": fold.repeat, "fold": fold.fold, "child": c, "hit": float(h)}
                for c, h in zip(tid, hit, strict=True)
            ]
    curve = pd.DataFrame(
        {"repeat": fold.repeat, "fold": fold.fold, "step": 50 * np.arange(1, len(losses) + 1), "loss": losses}
    )
    return pd.DataFrame(rows), pd.DataFrame(ident), curve


def run_encoder(cfg: Config, inp, meta: pd.DataFrame, core: list[str], targets: list[Target]) -> Result:
    paradigms = list(inp.paradigms)
    qc = inp.qc
    kept = qc[(qc.status == "ok") & qc.paradigm.isin(paradigms)]
    n_views = kept.groupby("child").paradigm.nunique()
    pool = sorted(c for c in n_views.index[n_views >= 2] if c in meta.index)
    segs = load_segments(cfg, pool, paradigms)
    pool = [c for c in pool if len(segs.get(c, {})) >= 2]
    core_views = {c for c in core if len(segs.get(c, {})) == len(paradigms)}
    log.info(
        "encoder: %d children with >= 2 views (%d segments)",
        len(pool),
        sum(len(s) for v in segs.values() for s in v.values()),
    )
    wide = inp.bank.wide(cfg.identification.primary_family)
    logpow = {p: pd.DataFrame(paradigm_matrix(wide, p, core), index=core) for p in paradigms}
    t0 = targets[0]  # ASD vs TD: its folds are used for every target
    kids = list(t0.y.dropna().index)
    sf = SplitFactory(kids, cfg.project.seed, strata=t0.strata.reindex(kids).to_numpy())
    folds = list(sf.outer(cfg.prediction.outer_folds, cfg.encoder.repeats))
    # folds in parallel; each worker uses threads_per_worker threads, so the result does not depend on n_jobs
    out = Parallel(n_jobs=cfg.runtime.n_jobs)(
        delayed(run_fold)(cfg, f, segs, pool, targets, logpow, paradigms, core_views) for f in folds
    )
    preds = pd.concat([o[0] for o in out], ignore_index=True)
    ident = pd.concat([o[1] for o in out], ignore_index=True)
    curves = pd.concat([o[2] for o in out], ignore_index=True)
    rows = []
    for t in targets:
        y = t.y.dropna()
        y = y[[c in set(kids) for c in y.index]]
        prev = float(y.mean()) if t.kind == "classification" else None
        for name in MODELS:
            p = preds[(preds.model == name) & (preds.target == t.name)].groupby("child").pred.mean()
            p = p.reindex(y.index)
            rng, seed = generator(cfg.project.seed, "encoder_boot", t.name, name)
            rows.append(
                {
                    "target": t.name,
                    "model": name,
                    "n": len(y),
                    "repeats": cfg.encoder.repeats,
                    **bootstrap_metrics(
                        y.to_numpy(),
                        p.to_numpy(),
                        t.kind,
                        cfg.prediction.n_bootstrap,
                        rng,
                        cfg.stats.ci_level,
                        prev,
                    ),
                    "boot_seed": seed,
                }
            )
    id_rows = []
    per_child = ident.groupby(["model", "child"]).hit.mean().reset_index()
    for name, g in per_child.groupby("model"):
        rng, seed = generator(cfg.project.seed, "encoder_ident_boot", name)
        h = g.hit.to_numpy()
        draws = [h[rng.integers(0, len(h), len(h))].mean() for _ in range(cfg.prediction.n_bootstrap)]
        lo, hi = percentile_ci(np.array(draws), cfg.stats.ci_level)
        id_rows.append(
            {
                "model": name,
                "n_children": len(h),
                "mean_test_fold_size": float(
                    ident[ident.model == name].groupby(["repeat", "fold"]).size().mean()
                ),
                "accuracy": float(h.mean()),
                "ci_lo": lo,
                "ci_hi": hi,
                "boot_seed": seed,
            }
        )
    summ = {
        "n_pool": len(pool),
        "n_folds": len(folds),
        "final_loss_median": float(curves.groupby(["repeat", "fold"]).loss.last().median()),
    }
    return Result(
        "encoder",
        {
            "ladder": pd.DataFrame(rows),
            "identification": pd.DataFrame(id_rows),
            "oof": preds,
            "identification_per_child": ident,
            "loss": curves,
        },
        summ,
        label="estimation",
    )
