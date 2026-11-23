"""ResultsStore: writes :class:`~utils.base.Result` objects under ``<results>/<stage>/`` and reads them back."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.loader import Config
from utils.base import Result


class ResultsStore:
    """Read and write results under ``<results>/<stage>/``."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.root = cfg.path("results")

    def write(self, result: Result, stage: str, **_: Any) -> list[Path]:
        """Write the tables (parquet) and the summary (JSON) of ``result`` to ``<results>/<stage>/``."""
        return result.write(self.root / stage)

    def summary(self, stage: str, name: str) -> dict[str, Any]:
        return json.loads((self.root / stage / f"{name}.json").read_text(encoding="utf-8"))

    def table(self, stage: str, name: str, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / stage / f"{name}__{key}.parquet")
