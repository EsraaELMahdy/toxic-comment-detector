from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter

from core.schemas import ClassificationResult, LabelPrediction


class ModelStatus(str, Enum):
    READY = "ready"
    MISSING_WEIGHTS = "missing_weights"
    UNAVAILABLE = "unavailable"


@dataclass
class RawOutput:
    predictions: list[LabelPrediction] = field(default_factory=list)


class BaseClassifier(ABC):
    model_id = "base"
    display_name = "Base"
    description = ""

    def __init__(self) -> None:
        self._loaded = False
        self._load_error: str | None = None

    @abstractmethod
    def availability(self) -> tuple[ModelStatus, str]: ...

    def load(self) -> None:
        if self._loaded:
            return

        try:
            self._load()
            self._loaded = True
            self._load_error = None
        except Exception as error:
            self._load_error = f"{type(error).__name__}: {error}"
            raise

    @abstractmethod
    def _load(self) -> None: ...

    def predict(self, text: str) -> ClassificationResult:
        if not self._loaded:
            self.load()

        started = perf_counter()
        raw = self._predict(text)
        elapsed = (perf_counter() - started) * 1000

        return ClassificationResult(
            model_id=self.model_id,
            model_label=self.display_name,
            input_text=text,
            predictions=raw.predictions,
            latency_ms=elapsed,
        )

    @abstractmethod
    def _predict(self, text: str) -> RawOutput: ...

    @property
    def load_error(self) -> str | None:
        return self._load_error
