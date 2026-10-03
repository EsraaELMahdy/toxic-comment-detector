from __future__ import annotations

from datetime import datetime

from core.database import COLUMNS, ResultRepository
from core.schemas import HistoryRecord


def make_record(text: str = "some comment", classification: str = "toxic") -> HistoryRecord:
    return HistoryRecord(
        created_at=datetime(2026, 10, 3, 14, 30, 5),
        input_type="text",
        input_text=text,
        model_label="LSTM",
        classification=classification,
        probabilities='{"toxic": 0.91}',
        latency_ms=12.5,
    )


class TestRepository:
    def test_creates_file_on_init(self, tmp_path):
        repository = ResultRepository(tmp_path / "results.csv")
        repository.initialise()
        assert repository.path.exists()

    def test_load_on_missing_file_returns_empty_frame(self, tmp_path):
        frame = ResultRepository(tmp_path / "missing.csv").load()
        assert frame.empty
        assert list(frame.columns) == list(COLUMNS)

    def test_save_appends_rows(self, tmp_path):
        repository = ResultRepository(tmp_path / "results.csv")
        repository.save(make_record("first"))
        repository.save(make_record("second"))

        frame = repository.load()
        assert len(frame) == 2
        assert list(frame["input"]) == ["first", "second"]

    def test_saved_row_carries_every_column(self, tmp_path):
        repository = ResultRepository(tmp_path / "results.csv")
        repository.save(make_record("hello"))

        row = repository.load().iloc[0]
        assert row["date"] == "2026-10-03"
        assert row["time"] == "14:30:05"
        assert row["model"] == "LSTM"
        assert row["classification"] == "toxic"

    def test_commas_in_input_do_not_break_the_csv(self, tmp_path):
        repository = ResultRepository(tmp_path / "results.csv")
        repository.save(make_record("hello, world, and again"))

        assert repository.load().iloc[0]["input"] == "hello, world, and again"

    def test_clear_keeps_headers(self, tmp_path):
        repository = ResultRepository(tmp_path / "results.csv")
        repository.save(make_record())
        repository.clear()

        frame = repository.load()
        assert frame.empty
        assert list(frame.columns) == list(COLUMNS)

    def test_count(self, tmp_path):
        repository = ResultRepository(tmp_path / "results.csv")
        for _ in range(3):
            repository.save(make_record())
        assert repository.count() == 3

    def test_recovers_from_corrupt_file(self, tmp_path):
        path = tmp_path / "results.csv"
        path.write_text('a,b\n"unclosed,quote\n', encoding="utf-8")

        frame = ResultRepository(path).load()
        assert list(frame.columns) == list(COLUMNS)

    def test_backfills_columns_from_older_files(self, tmp_path):
        path = tmp_path / "results.csv"
        path.write_text("date,time,input\n2026-01-01,00:00:00,old row\n", encoding="utf-8")

        frame = ResultRepository(path).load()
        assert list(frame.columns) == list(COLUMNS)
        assert frame.iloc[0]["input"] == "old row"

    def test_export_bytes_round_trip(self, tmp_path):
        repository = ResultRepository(tmp_path / "results.csv")
        repository.save(make_record("exported"))

        assert b"exported" in repository.to_csv_bytes()
