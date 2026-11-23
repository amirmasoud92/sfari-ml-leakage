"""Base classes shared by the methods: the :class:`Result` container returned by the analyses (named tables
plus scalar summaries, written as parquet and JSON) and the abstract interfaces :class:`Preprocessor`,
:class:`FeatureFamily` and :class:`Analysis`."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

if TYPE_CHECKING:
    from config.loader import Config
    from utils.dataset import Recording

#: Columns of every feature table (one row per child x paradigm x block x feature).
FEATURE_COLUMNS = ("child", "paradigm", "block", "feature", "value")


@dataclass
class Result:
    """Typed analysis output: named tables plus scalar summaries.

    Attributes
    ----------
    name
        Output name (file stem under ``results/<stage>/``).
    tables
        Name -> tidy DataFrame (written as parquet).
    summary
        JSON-serialisable scalars and small structures (written as ``<name>.json``).
    label
        ``confirmatory``, ``estimation`` or ``exploratory`` (stored with every table).
    """

    name: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    label: str = "confirmatory"

    def write(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for key, df in self.tables.items():
            p = out_dir / f"{self.name}__{key}.parquet"
            df.assign(analysis_label=self.label).to_parquet(p, index=False)
            paths.append(p)
        p = out_dir / f"{self.name}.json"
        p.write_text(
            json.dumps({"label": self.label, **self.summary}, indent=2, default=_json_default),
            encoding="utf-8",
        )
        paths.append(p)
        return paths

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Preprocessor(ABC):
    """Turns one recording into a cleaned continuous Raw plus a QC record."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def run(self, recording: Recording) -> Any:
        """Process one recording; returns a result with the cleaned data path and the QC record."""


class FeatureFamily(ABC):
    """Computes one family of features per block, as a tidy table with :data:`FEATURE_COLUMNS`.

    Subclasses set ``paradigms`` (the paradigms the family applies to; ``None`` = every seated paradigm) and
    ``shared`` (whether the family has the same definition in every paradigm, which the identification and
    variance analyses require).
    """

    paradigms: ClassVar[tuple[str, ...] | None] = None
    shared: ClassVar[bool] = True

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def compute(
        self, child: str, paradigm: str, data: Any, blocks: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(features, metadata)``: features with :data:`FEATURE_COLUMNS`; metadata one row per feature
        name with at least ``feature, channel, band, type``."""


class Analysis(ABC):
    """One analysis; reads the feature bank (and the labels) and returns a :class:`Result`."""

    label: ClassVar[str] = "confirmatory"
    needs_labels: ClassVar[bool] = False

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def run(self, **inputs: Any) -> Result:
        """Run the analysis."""
