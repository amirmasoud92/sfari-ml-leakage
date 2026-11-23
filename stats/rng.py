"""Seeded random generators derived from the master seed (``project.seed``) and a key, so that every permutation,
bootstrap and fold assignment is reproducible.

The key strings passed by the analyses (for example the name of an analysis and a target) are opaque labels. They
enter the seed derivation and are therefore part of the numerical specification: changing a key changes the
random draws of that analysis.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(master: int, *keys: object) -> int:
    """A 32-bit seed from ``master`` and any keys (stable across Python sessions)."""
    blob = "|".join([str(master), *map(str, keys)]).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little")


def generator(master: int, *keys: object) -> tuple[np.random.Generator, int]:
    """``(Generator, seed)`` for the key; the seed is stored with the results."""
    s = derive_seed(master, *keys)
    return np.random.default_rng(s), s
