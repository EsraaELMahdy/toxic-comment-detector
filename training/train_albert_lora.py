from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import (
    ALBERT_ADAPTER_DIR,
    MAX_LENGTH,
    ALBERT_BASE_MODEL,
    BATCH_SIZE,
    LABELS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_MODULES_TO_SAVE,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_GRAD_NORM,
    MAX_NEGATIVE_SAMPLES,
    NUM_EPOCHS,
    RANDOM_SEED,
    TRAIN_CSV,
    WEIGHT_DECAY,
)
from training.dataset import ToxicCommentDataset, load_training_frame

logger = logging.getLogger("train_albert_lora")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD_GRID = np.arange(0.20, 0.91, 0.05)


def build_model():
    try:
        import torch

        torch.backends.mkldnn.enabled = True
    except Exception:
        pass

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        ALBERT_BASE_MODEL,
        num_labels=len(LABELS),
    )
    model.config.problem_type = "multi_label_classification"

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        modules_to_save=list(LORA_MODULES_TO_SAVE),
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.to(DEVICE)
    return model


def collect_probabilities(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truths, probabilities = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            truths.append(batch["labels"].numpy())

    return np.vstack(truths), np.vstack(probabilities)


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    thresholds = {}

    for index, label in enumerate(LABELS):
        best_f1, best_threshold = -1.0, 0.5
        for candidate in THRESHOLD_GRID:
            score = f1_score(
                y_true[:, index],
                (y_prob[:, index] >= candidate).astype(int),
                zero_division=0,
            )
            if score > best_f1:
                best_f1, best_threshold = score, float(round(candidate, 2))

        thresholds[label] = best_threshold
        logger.info("threshold[%s] = %.2f (F1 %.4f)", label, best_threshold, best_f1)

    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ALBERT + LoRA for toxic comment detection.")
    parser.add_argument("--csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--output", type=Path, default=ALBERT_ADAPTER_DIR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-negatives", type=int, default=MAX_NEGATIVE_SAMPLES)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    torch.manual_seed(RANDOM_SEED)

    if not Path(args.csv).exists():
        raise SystemExit(
            f"Training data not found at {args.csv}.\n"
            "Download train.csv from "
            "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge "
            "and place it in data/."
        )

    from transformers import AutoTokenizer

    frame = load_training_frame(args.csv, args.max_negatives, RANDOM_SEED)

    if args.max_rows and len(frame) > args.max_rows:
        frame = frame.sample(n=args.max_rows, random_state=RANDOM_SEED).reset_index(drop=True)
        logger.info("Capped training frame to %s rows", len(frame))
    train_frame, val_frame = train_test_split(
        frame,
        test_size=0.2,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ALBERT_BASE_MODEL, use_fast=True)
    train_loader = DataLoader(
        ToxicCommentDataset(train_frame, tokenizer, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ToxicCommentDataset(val_frame, tokenizer, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    best_macro_f1 = 0.0
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)

        y_true, y_prob = collect_probabilities(model, val_loader)
        y_pred = (y_prob >= 0.5).astype(int)

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        logger.info(
            "epoch %d/%d | train_loss %.4f | val_macro_f1 %.4f | precision %.4f | recall %.4f",
            epoch,
            args.epochs,
            running_loss / len(train_loader.dataset),
            macro_f1,
            precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_score(y_true, y_pred, average="macro", zero_division=0),
        )

        if macro_f1 >= best_macro_f1:
            best_macro_f1 = macro_f1
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            logger.info("saved adapter (macro F1 %.4f)", macro_f1)

    y_true, y_prob = collect_probabilities(model, val_loader)
    thresholds = tune_thresholds(y_true, y_prob)
    (args.output / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    tuned = np.column_stack(
        [(y_prob[:, index] >= thresholds[label]).astype(int) for index, label in enumerate(LABELS)]
    )
    logger.info(
        "tuned macro F1 %.4f\n%s",
        f1_score(y_true, tuned, average="macro", zero_division=0),
        classification_report(y_true, tuned, target_names=list(LABELS), zero_division=0),
    )


if __name__ == "__main__":
    main()
