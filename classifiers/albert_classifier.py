from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from classifiers.base import BaseClassifier, ModelStatus, RawOutput
from config import ALBERT_ADAPTER_DIR, ALBERT_BASE_MODEL, DEFAULT_THRESHOLDS, LABELS, MAX_LENGTH
from core.schemas import LabelPrediction

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLDS_FILE = "thresholds.json"


class AlbertLoraClassifier(BaseClassifier):
    model_id = "albert_lora"
    display_name = "ALBERT + LoRA"
    description = "albert-base-v2 fine-tuned with LoRA (r=8, alpha=16) on the Jigsaw dataset."

    def __init__(self, adapter_dir: Path = ALBERT_ADAPTER_DIR) -> None:
        super().__init__()
        self.adapter_dir = Path(adapter_dir)
        self._model = None
        self._tokenizer = None
        self._thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS

    def availability(self) -> tuple[ModelStatus, str]:
        if not self.adapter_dir.exists() or not self._has_adapter_files():
            return (
                ModelStatus.MISSING_WEIGHTS,
                "Adapter missing. Train it with: python -m training.train_albert_lora",
            )

        missing = self._missing_packages()
        if missing:
            return (
                ModelStatus.UNAVAILABLE,
                f"Missing package(s): {', '.join(missing)}. Run: pip install {' '.join(missing)}",
            )

        return ModelStatus.READY, f"Ready · adapter {self.adapter_dir.name}"

    def _has_adapter_files(self) -> bool:
        return (self.adapter_dir / "adapter_config.json").exists()

    def _needs_sentencepiece(self) -> bool:
        return not (self.adapter_dir / "tokenizer.json").exists()

    def _missing_packages(self) -> list[str]:
        missing = []

        for package in ("transformers", "peft"):
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if self._needs_sentencepiece():
            try:
                __import__("sentencepiece")
            except ImportError:
                missing.append("sentencepiece")

        return missing

    def _load(self) -> None:
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not self.adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {self.adapter_dir}")

        missing = self._missing_packages()
        if missing:
            raise ImportError(
                f"ALBERT needs these packages: {', '.join(missing)}. "
                f"Install them with: pip install {' '.join(missing)}"
            )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir)
        except ValueError as error:
            raise RuntimeError(
                "ALBERT's tokenizer could not be loaded. It is a SentencePiece model, so the "
                "`sentencepiece` package is required: pip install sentencepiece"
            ) from error

        base = AutoModelForSequenceClassification.from_pretrained(
            ALBERT_BASE_MODEL,
            num_labels=len(LABELS),
        )
        base.config.problem_type = "multi_label_classification"

        model = PeftModel.from_pretrained(base, self.adapter_dir)
        model.to(DEVICE)
        model.eval()

        self._model = model
        self._thresholds = self._read_thresholds()

        logger.info("ALBERT adapter loaded from %s", self.adapter_dir)

    def _read_thresholds(self) -> tuple[float, ...]:
        sidecar = self.adapter_dir / THRESHOLDS_FILE
        if not sidecar.exists():
            return DEFAULT_THRESHOLDS

        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            return tuple(float(payload[label]) for label in LABELS)
        except Exception:
            logger.warning("Could not read %s, using 0.5 thresholds.", sidecar)
            return DEFAULT_THRESHOLDS

    def _predict(self, text: str) -> RawOutput:
        assert self._model is not None and self._tokenizer is not None

        encoded = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self._model(**encoded).logits
            probabilities = torch.sigmoid(logits)[0].cpu().tolist()

        predictions = [
            LabelPrediction(
                label=label,
                probability=float(probabilities[index]),
                threshold=float(self._thresholds[index]),
            )
            for index, label in enumerate(LABELS)
        ]
        return RawOutput(predictions=predictions)
