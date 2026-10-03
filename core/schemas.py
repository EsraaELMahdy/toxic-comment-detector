from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from config import LABEL_DISPLAY_NAMES


@dataclass(frozen=True)
class LabelPrediction:
    label: str
    probability: float
    threshold: float

    @property
    def positive(self) -> bool:
        return self.probability >= self.threshold

    @property
    def display_name(self) -> str:
        return LABEL_DISPLAY_NAMES.get(self.label, self.label.title())


@dataclass(frozen=True)
class ClassificationResult:
    model_id: str
    model_label: str
    input_text: str
    predictions: Sequence[LabelPrediction]
    latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def detected(self) -> list[LabelPrediction]:
        return [prediction for prediction in self.predictions if prediction.positive]

    @property
    def is_clean(self) -> bool:
        return not self.detected

    @property
    def verdict(self) -> str:
        if self.is_clean:
            return "non-toxic"
        return ", ".join(prediction.label for prediction in self.detected)

    @property
    def severity(self) -> float:
        if self.is_clean:
            return 0.0
        return max(prediction.probability for prediction in self.detected)

    @property
    def probabilities_json(self) -> str:
        import json

        return json.dumps(
            {prediction.label: round(prediction.probability, 4) for prediction in self.predictions}
        )


@dataclass(frozen=True)
class HistoryRecord:
    created_at: datetime
    input_type: str
    input_text: str
    model_label: str
    classification: str
    probabilities: str
    latency_ms: float = 0.0

    CSV_COLUMNS = (
        "date",
        "time",
        "input_type",
        "input",
        "model",
        "classification",
        "probabilities",
        "latency_ms",
    )

    def as_row(self) -> list[str]:
        return [
            self.created_at.strftime("%Y-%m-%d"),
            self.created_at.strftime("%H:%M:%S"),
            self.input_type,
            self.input_text,
            self.model_label,
            self.classification,
            self.probabilities,
            f"{self.latency_ms:.1f}",
        ]
