"""Child-level cross-validation splits.

Splits are drawn over *children*, never over rows, so every block, run and segment of a child falls in the same
fold. :meth:`SplitFactory.row_indices` maps child folds onto any long table. :meth:`SplitFactory.sibling_held_out`
gives the variant in which siblings form one held-out test set.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass(frozen=True)
class ChildFold:
    """One outer fold: ``repeat``, ``fold`` and the train / test child IDs."""

    repeat: int
    fold: int
    train: tuple[str, ...]
    test: tuple[str, ...]

    def check(self) -> None:
        overlap = set(self.train) & set(self.test)
        if overlap:
            raise AssertionError(f"children in both train and test: {sorted(overlap)[:5]}")


class SplitFactory:
    """Child-level repeated (stratified) K-fold splits with a recorded seed.

    Parameters
    ----------
    children
        Child IDs (unique).
    seed
        Seed; repeat ``r`` uses ``seed + r``.
    strata
        Optional labels per child for stratification (group, or binned continuous target).
    """

    def __init__(self, children: Sequence[str], seed: int, strata: Sequence | None = None) -> None:
        ids = list(children)
        if len(set(ids)) != len(ids):
            raise ValueError("child IDs must be unique")
        self.children = np.array(ids, dtype=object)
        self.seed = int(seed)
        self.strata = None if strata is None else np.asarray(strata)
        if self.strata is not None and len(self.strata) != len(ids):
            raise ValueError("strata must have one entry per child")

    def outer(self, n_folds: int, repeats: int = 1) -> Iterator[ChildFold]:
        for r in range(repeats):
            if self.strata is not None:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed + r)
                it = cv.split(self.children, self.strata)
            else:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed + r)
                it = cv.split(self.children)
            for f, (tr, te) in enumerate(it):
                fold = ChildFold(r, f, tuple(self.children[tr]), tuple(self.children[te]))
                fold.check()
                yield fold

    def inner(
        self,
        train_children: Sequence[str],
        n_folds: int,
        strata: Sequence | None = None,
        offset: int = 10_000,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Index splits of ``train_children`` for hyperparameter search (indices into that sequence)."""
        n = len(train_children)
        rs = self.seed + offset
        if strata is not None:
            return list(StratifiedKFold(n_folds, shuffle=True, random_state=rs).split(np.zeros(n), strata))
        return list(KFold(n_folds, shuffle=True, random_state=rs).split(np.zeros(n)))

    @staticmethod
    def sibling_held_out(children: Sequence[str], groups: Sequence[str], sib_label: str = "SIB") -> ChildFold:
        """Train on every non-sibling child, test on all siblings."""
        g = np.asarray(groups)
        c = np.asarray(children, dtype=object)
        fold = ChildFold(0, 0, tuple(c[g != sib_label]), tuple(c[g == sib_label]))
        fold.check()
        return fold

    @staticmethod
    def row_indices(row_children: Sequence[str], fold: ChildFold) -> tuple[np.ndarray, np.ndarray]:
        """Row indices of a long table (one child ID per row) for the fold's train and test children."""
        rc = np.asarray(row_children, dtype=object)
        tr = np.flatnonzero(np.isin(rc, fold.train))
        te = np.flatnonzero(np.isin(rc, fold.test))
        if np.intersect1d(rc[tr], rc[te]).size:
            raise AssertionError("a child has rows in both train and test")
        return tr, te
