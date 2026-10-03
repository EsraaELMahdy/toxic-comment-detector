from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from classifiers.base import BaseClassifier, ModelStatus, RawOutput
from config import DEFAULT_THRESHOLDS, LABELS, resolve_lstm_checkpoint
from core.preprocessing import encode_for_lstm, ensure_nltk_data
from core.schemas import LabelPrediction

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ToxicLSTM(nn.Module):
    def __init__(
        self,
        vocab_size=30000,
        embedding_dim=300,
        hidden_dim=300,
        output_dim=3,
        dropout=0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lengths = (inputs != 0).sum(dim=1).clamp(min=1)

        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.rnn(packed)
        joined = torch.cat([hidden[-2], hidden[-1]], dim=1)

        return self.fc2(self.dropout(self.relu(self.fc1(self.dropout(joined)))))


class LSTMClassifier(BaseClassifier):
    model_id = "lstm"
    display_name = "LSTM"
    description = "Bidirectional LSTM over a 30k vocabulary with tuned per-label thresholds."

    def __init__(self) -> None:
        super().__init__()
        self._model: ToxicLSTM | None = None
        self._vocab: dict[str, int] = {}
        self._thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
        self._max_len = 200

    def availability(self) -> tuple[ModelStatus, str]:
        checkpoint = resolve_lstm_checkpoint()

        if checkpoint is None:
            return (
                ModelStatus.MISSING_WEIGHTS,
                "Checkpoint not found in models/lstm/.",
            )

        if not ensure_nltk_data():
            return ModelStatus.UNAVAILABLE, "NLTK data unavailable."

        size = checkpoint.stat().st_size / 1e6
        return ModelStatus.READY, f"Ready · {checkpoint.name} ({size:.0f} MB)"

    @staticmethod
    def _read_checkpoint(path):
        try:
            return torch.load(path, map_location=DEVICE, weights_only=True)
        except Exception:
            logger.info("Loading %s with the unrestricted unpickler.", path.name)
            return torch.load(path, map_location=DEVICE, weights_only=False)

    def _load(self) -> None:
        path = resolve_lstm_checkpoint()
        if path is None:
            raise FileNotFoundError("LSTM checkpoint not found.")

        ensure_nltk_data()
        checkpoint = self._read_checkpoint(path)

        self._vocab = checkpoint["vocab"]
        self._thresholds = tuple(float(value) for value in checkpoint["thresholds"])
        self._max_len = int(checkpoint.get("max_len", 200))

        model = ToxicLSTM(
            vocab_size=len(self._vocab),
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=len(checkpoint["target_columns"]),
            dropout=checkpoint["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        self._model = model
        logger.info("LSTM loaded from %s", path)

    def _predict(self, text: str) -> RawOutput:
        assert self._model is not None

        sequence = encode_for_lstm(text, self._vocab, self._max_len)
        inputs = torch.tensor([sequence], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            probabilities = torch.sigmoid(self._model(inputs))[0].cpu().tolist()

        predictions = [
            LabelPrediction(
                label=label,
                probability=float(probabilities[index]),
                threshold=float(self._thresholds[index]),
            )
            for index, label in enumerate(LABELS)
        ]
        return RawOutput(predictions=predictions)
