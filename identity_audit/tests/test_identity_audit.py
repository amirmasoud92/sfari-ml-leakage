"""Tests of the identity_audit module on synthetic data in which the features carry subject identity and the
labels carry nothing (labels are random per subject)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from identity_audit import identity_audit as IA  # noqa: E402


def knn(seed):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(5))


@pytest.fixture(scope="module")
def identity_only():
    rng = np.random.default_rng(1)
    n_sub, n_blk, n_feat = 40, 20, 30
    offsets = rng.normal(0, 1, (n_sub, n_feat))
    groups = np.repeat(np.arange(n_sub), n_blk)
    X = offsets[groups] + rng.normal(0, 0.5, (n_sub * n_blk, n_feat))
    lab = rng.permutation(np.r_[np.zeros(n_sub // 2), np.ones(n_sub // 2)]).astype(int)
    return X, lab[groups], groups, offsets


def test_leaky_split_recognises_subjects_not_labels(identity_only):
    X, y, g, _ = identity_only
    R = IA.split_contrast(X, y, g, knn, repeats=2, seed=0)
    by_block = R[R.scheme == "by block"].auc_subject.mean()
    by_subject = R[R.scheme == "by subject"].auc_subject.mean()
    assert by_block > 0.95  # labels carry nothing; only identity can produce this
    assert 0.2 < by_subject < 0.8


def test_permuted_labels_keep_subject_structure(identity_only):
    X, y, g, _ = identity_only
    yp = IA.permute_subject_labels(y, g, np.random.default_rng(0))
    for s in np.unique(g):
        assert len(set(yp[g == s])) == 1
    assert yp.sum() == y.sum()
    P = IA.permuted_label_control(X, y, g, knn, n_perm=3, repeats=1, seed=0)
    assert (P[P.scheme == "by block"].auc_subject > 0.9).all()


def test_dose_response_rises_with_leaked_blocks(identity_only):
    X, y, g, _ = identity_only
    D = IA.dose_response(X, y, g, knn, ks=(0, 4, 8), seed=0)
    a = D.set_index("k").auc_subject
    assert a[8] > a[0] + 0.3
    assert 0.2 < a[0] < 0.8


def test_identify_blocks_and_fingerprint(identity_only):
    X, y, g, offsets = identity_only
    res = IA.identify_blocks(X, g, knn, seed=0, n_boot=200)
    assert res["accuracy"] > 0.95 and res["ci"][0] <= res["accuracy"] <= res["ci"][1]
    rng = np.random.default_rng(2)
    A = offsets + rng.normal(0, 0.3, offsets.shape)
    B = offsets + rng.normal(0, 0.3, offsets.shape)
    assert IA.fingerprint_accuracy(A, B) > 0.9
    assert IA.fingerprint_accuracy(A, rng.normal(0, 1, offsets.shape)) < 0.2


def test_error_corrected_auc_and_power():
    assert IA.error_corrected_auc(0.0, 0.8) == pytest.approx(0.5)
    assert IA.error_corrected_auc(0.5, 0.0) == 1.0
    assert IA.error_corrected_auc(0.71, 1.0) == pytest.approx(IA.auc_from_g(0.71))
    assert IA.error_corrected_auc(0.5, 0.5) > IA.error_corrected_auc(0.5, 0.9)
    assert IA.n_per_group(0.5) == 63
    assert np.isnan(IA.error_corrected_auc(np.nan, 0.5))
    with pytest.raises(ValueError):
        list(IA._splits(np.zeros((4, 1)), np.array([0, 1, 0, 1]), np.arange(4), "by trial", 2, 0))


def test_positive_control_signal_survives_subject_split():
    """A label with a real between-subject effect must be detected with subjects kept whole."""
    rng = np.random.default_rng(3)
    n_sub, n_blk, n_feat = 40, 20, 30
    lab = rng.permutation(np.r_[np.zeros(n_sub // 2), np.ones(n_sub // 2)]).astype(int)
    offsets = rng.normal(0, 1, (n_sub, n_feat))
    offsets[:, :5] += 1.5 * lab[:, None]  # group effect on five features
    groups = np.repeat(np.arange(n_sub), n_blk)
    X = offsets[groups] + rng.normal(0, 0.5, (n_sub * n_blk, n_feat))
    R = IA.split_contrast(X, lab[groups], groups, knn, repeats=2, seed=0)
    assert R[R.scheme == "by subject"].auc_subject.mean() > 0.8


def test_negative_control_no_identity_no_leak():
    """Without subject offsets, block splitting must not inflate the AUC of random labels."""
    rng = np.random.default_rng(4)
    n_sub, n_blk, n_feat = 40, 20, 30
    groups = np.repeat(np.arange(n_sub), n_blk)
    X = rng.normal(0, 1, (n_sub * n_blk, n_feat))
    lab = rng.permutation(np.r_[np.zeros(n_sub // 2), np.ones(n_sub // 2)]).astype(int)
    R = IA.split_contrast(X, lab[groups], groups, knn, repeats=2, seed=0)
    assert 0.2 < R[R.scheme == "by block"].auc_subject.mean() < 0.8


def test_dose_response_never_scores_training_blocks(identity_only):
    X, y, g, _ = identity_only
    _, index = IA.dose_response(X, y, g, knn, ks=(0, 2, 8), seed=0, return_index=True)
    for fold in index:
        evs = [set(ev.tolist()) for _, ev in fold.values()]
        assert all(e == evs[0] for e in evs)  # the same evaluation blocks for every k
        for idx, ev in fold.values():
            assert not set(idx.tolist()) & set(ev.tolist())  # no scored block in training


def test_dose_response_cross_condition_leaks_only_other_conditions(identity_only):
    X, y, g, _ = identity_only
    cond = np.tile(np.repeat(np.arange(4), 5), len(np.unique(g)))  # four conditions of five blocks per subject
    _, index = IA.dose_response(X, y, g, knn, ks=(0, 8), seed=0, conditions=cond, return_index=True)
    for fold in index:
        idx0, ev = fold[0]
        idx8, _ = fold[8]
        leaked = np.setdiff1d(idx8, idx0)
        assert len(leaked) > 0
        for s in np.unique(g[ev]):
            ev_c = set(cond[ev][g[ev] == s].tolist())
            assert len(ev_c) == 1  # one scored condition per test subject
            assert not ev_c & set(cond[leaked][g[leaked] == s].tolist())  # leaked blocks from other conditions
