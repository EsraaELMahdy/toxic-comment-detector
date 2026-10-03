from __future__ import annotations

import re
from functools import lru_cache

from config import NLTK_PACKAGES

_HTML_TAG = re.compile(r"<.*?>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
_WHITESPACE = re.compile(r"\s+")

_ready = False


def ensure_nltk_data(packages=NLTK_PACKAGES) -> bool:
    global _ready

    if _ready:
        return True

    try:
        import nltk

        for package in packages:
            try:
                nltk.data.find(f"tokenizers/{package}")
            except LookupError:
                try:
                    nltk.data.find(f"corpora/{package}")
                except LookupError:
                    nltk.download(package, quiet=True)

        _ready = True
    except Exception:
        _ready = False

    return _ready


def clean_text(text) -> str:
    text = str(text).lower()
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


@lru_cache(maxsize=1)
def _lemmatizer():
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


def tokenize(text: str) -> list[str]:
    ensure_nltk_data()

    try:
        from nltk.tokenize import word_tokenize

        words = word_tokenize(clean_text(text))
    except Exception:
        words = clean_text(text).split()

    lemmatizer = _lemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def encode_for_lstm(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    unk_id = vocab.get("<UNK>", 1)
    ids = [vocab.get(token, unk_id) for token in tokenize(text)][:max_len]
    return ids + [0] * (max_len - len(ids))
