from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import RESULTS_CSV
from core.schemas import HistoryRecord

logger = logging.getLogger(__name__)

COLUMNS = HistoryRecord.CSV_COLUMNS


class ResultRepository:
    def __init__(self, path: Path | str = RESULTS_CSV) -> None:
        self.path = Path(path)

    def initialise(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            pd.DataFrame(columns=list(COLUMNS)).to_csv(self.path, index=False)

    def save(self, record: HistoryRecord) -> None:
        self.initialise()
        row = pd.DataFrame([record.as_row()], columns=list(COLUMNS))
        row.to_csv(self.path, mode="a", header=False, index=False, quoting=1)

    def load(self) -> pd.DataFrame:
        self.initialise()

        try:
            frame = pd.read_csv(self.path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            logger.warning("Unreadable database at %s, starting fresh.", self.path)
            frame = pd.DataFrame(columns=list(COLUMNS))

        for column in COLUMNS:
            if column not in frame.columns:
                frame[column] = ""

        return frame[list(COLUMNS)].fillna("")

    def count(self) -> int:
        return len(self.load())

    def clear(self) -> None:
        self.initialise()
        pd.DataFrame(columns=list(COLUMNS)).to_csv(self.path, index=False)

    def to_csv_bytes(self) -> bytes:
        return self.load().to_csv(index=False).encode("utf-8")
