"""Permuted-label control for EEGNet (Methods, Other classifiers; Table 2; Additional file 1, Supplementary
Methods S5 and Fig. S3d).

Same children (70 autistic and TD core children), segments (:func:`deep_learning.convnets.child_segments`: up to 60
2-s segments per child, seed ``generator(project seed, "deep")``), preprocessing (128 Hz, per-channel z-scoring with
training statistics), model, optimiser, learning rate, weight decay, epochs (20), batch size (64) and seed as
:mod:`deep_learning.convnets`. Only the labels change: diagnosis is permuted between children (all segments of a
child share its new label), 20 permutations. Folds are re-drawn on the permuted labels (children kept whole:
:class:`~classification.splits.SplitFactory` with the project seed and the permuted labels as strata; segments split:
StratifiedKFold with the deep seed on segment labels). The true labels are rerun in the same environment as the
reference. Child-level AUC under both schemes = AUC over children of the mean predicted probability of their test
segments; the segment-level AUC is also stored.

The reported run trained on CUDA devices with deterministic cuDNN (PyTorch 2.14.1 with CUDA 12.6, braindecode
2.0.0); GPU arithmetic changes single trainings relative to a CPU run. The batch order comes from a seeded CPU
generator.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from classification.splits import SplitFactory
from deep_learning.convnets import child_segments
from stats.rng import generator
from utils import common as C
from utils.inputs import Inputs

N_PERM = 20


def segments():
    """The segments of :mod:`deep_learning.convnets`: child order, child labels, segment array, owner index."""
    cfg = C.CFG
    inp = Inputs(cfg)
    meta, core = inp.meta(), inp.core()
    asd_td = [k for k in core if meta.loc[k, "group"] in ("ASD", "TD")]
    _, seed = generator(cfg.project.seed, "deep")
    segs = {c: child_segments(cfg, c, inp.paradigms, 60, seed) for c in asd_td}
    kids = [c for c in asd_td if len(segs[c])]
    y = (meta.loc[kids, "group"] == "ASD").astype(int).to_numpy()
    X = np.concatenate([segs[c] for c in kids])
    owner = np.concatenate([[i] * len(segs[c]) for i, c in enumerate(kids)])
    return kids, y, X, owner


def permutations(y_child, n):
    """``n`` permutations of the child labels between children."""
    rng, seed = generator(C.SEED, "revBI_b3")
    return [rng.permutation(y_child) for _ in range(n)], seed


def fold_assignment(kids, y, X, owner, n_perm: int = N_PERM) -> dict:
    """For the true labels (run -1) and each permutation: the child labels, the child-level fold of every child and
    the segment-level fold of every segment."""
    cfg = C.CFG
    _, deep_seed = generator(cfg.project.seed, "deep")
    perms, perm_seed = permutations(y, n_perm)
    runs = [(-1, y)] + [(i, yp) for i, yp in enumerate(perms)]
    labels, child_fold, seg_fold = [], [], []
    for _, lab in runs:
        cf = np.full(len(kids), -1)
        for f, fold in enumerate(
            SplitFactory(kids, cfg.project.seed, strata=lab).outer(cfg.prediction.outer_folds, 1)
        ):
            cf[[kids.index(c) for c in fold.test]] = f
        sf = np.full(len(owner), -1)
        ys = lab[owner]
        for f, (_, te) in enumerate(
            StratifiedKFold(cfg.prediction.outer_folds, shuffle=True, random_state=deep_seed).split(X, ys)
        ):
            sf[te] = f
        assert (cf >= 0).all() and (sf >= 0).all()
        labels.append(lab)
        child_fold.append(cf)
        seg_fold.append(sf)
    return {
        "run": np.array([r for r, _ in runs]),
        "labels": np.array(labels),
        "child_fold": np.array(child_fold),
        "seg_fold": np.array(seg_fold),
        "deep_seed": deep_seed,
        "perm_seed": perm_seed,
    }


def train_predict(Xtr, ytr, Xte, seed, device, epochs=20, batch=64):
    """EEGNet trained as in :func:`deep_learning.convnets.train_predict`, with model and tensors on ``device``."""
    import torch
    from braindecode.models import EEGNet

    torch.manual_seed(seed)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-12
    Xtr = torch.tensor((Xtr - mu) / sd).to(device)
    Xte = torch.tensor((Xte - mu) / sd).to(device)
    y = torch.tensor(ytr, dtype=torch.long).to(device)
    net = EEGNet(n_chans=Xtr.shape[1], n_outputs=2, n_times=Xtr.shape[2]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch].to(device)
            opt.zero_grad()
            loss = lossf(net(Xtr[idx]), y[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out = torch.cat([torch.softmax(net(Xte[i : i + 256]), 1)[:, 1] for i in range(0, len(Xte), 256)])
    return out.cpu().numpy()


def child_auc(prob, owner, y_child):
    p = pd.Series(prob).groupby(owner).mean().sort_index()
    return float(roc_auc_score(y_child[p.index.to_numpy()], p.to_numpy()))


def band(v):
    v = np.asarray(v, float)
    return [float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))]


def main(device: str | None = None, n_perm: int = N_PERM) -> None:
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out = C.out_dir("eegnet_permuted_labels")
    kids, y, X, owner = segments()
    design = fold_assignment(kids, y, X, owner, n_perm)
    seed = int(design["deep_seed"])
    rows = []
    for j in range(len(design["run"])):
        run = int(design["run"][j])
        yc = design["labels"][j]
        ys = yc[owner]
        for sc in ("child", "segment"):
            folds = design["child_fold"][j][owner] if sc == "child" else design["seg_fold"][j]
            prob = np.full(len(ys), np.nan)
            for f in range(C.CFG.prediction.outer_folds):
                tr, te = folds != f, folds == f
                prob[te] = train_predict(X[tr], ys[tr], X[te], seed, device)
            rows.append(
                {
                    "run": run,
                    "labels": "true" if run < 0 else "permuted",
                    "scheme": sc,
                    "auc_child": child_auc(prob, owner, yc),
                    "auc_segment": float(roc_auc_score(ys, prob)),
                }
            )
            print(rows[-1], flush=True)
    D = pd.DataFrame(rows).sort_values(["labels", "run", "scheme"]).reset_index(drop=True)
    D.to_csv(out / "runs.csv", index=False)
    P = D[D.labels == "permuted"]
    summ = {
        "seed": int(design["perm_seed"]),
        "deep_seed": seed,
        "device": str(device),
        "n_perm": int((design["run"] >= 0).sum()),
        "n_children": len(kids),
        "n_segments": len(owner),
        "permuted": {
            f"{sc} split | {m}": band(P[P.scheme == sc][m])
            for sc in ("child", "segment")
            for m in ("auc_child", "auc_segment")
        },
        "permuted_min_max": {
            f"{sc} split | auc_child": [
                float(P[P.scheme == sc].auc_child.min()),
                float(P[P.scheme == sc].auc_child.max()),
            ]
            for sc in ("child", "segment")
        },
        "true": D[D.labels == "true"][["scheme", "auc_child", "auc_segment"]].to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
