from __future__ import annotations

import logging

import pandas as pd
import torch
from torch.utils.data import Dataset

from config import LABELS, MAX_LENGTH

logger = logging.getLogger(__name__)


class ToxicCommentDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH) -> None:
        self.texts = dataframe["comment_text"].astype(str).tolist()
        self.labels = dataframe[list(LABELS)].values.astype("float32")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def load_training_frame(csv_path, max_negatives: int, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    logger.info("Loaded %s rows from %s", len(frame), csv_path)

    label_columns = list(LABELS)
    frame = frame[["comment_text", *label_columns]].copy()

    positives = frame[frame[label_columns].sum(axis=1) > 0]
    negatives = frame[frame[label_columns].sum(axis=1) == 0]

    negatives = negatives.sample(n=min(max_negatives, len(negatives)), random_state=seed)

    balanced = pd.concat([positives, negatives], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(
        "Balanced frame: %s rows (%s positive / %s negative)",
        len(balanced),
        len(positives),
        len(negatives),
    )
    return balanced
