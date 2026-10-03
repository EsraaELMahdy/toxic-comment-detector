from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "database"
RESEARCH_DIR = PROJECT_ROOT / "research"
FIGURES_DIR = RESEARCH_DIR / "figures"

TRAIN_CSV = DATA_DIR / "train.csv"
RESULTS_CSV = DATABASE_DIR / "results.csv"

LSTM_CHECKPOINT_CANDIDATES = (
    MODELS_DIR / "lstm" / "full_toxic_lstm_checkpoint.pth",
    MODELS_DIR / "full_toxic_lstm_checkpoint.pth",
    PROJECT_ROOT / "full_toxic_lstm_checkpoint.pth",
)

ALBERT_ADAPTER_DIR = MODELS_DIR / "albert_lora"

LABELS = ("toxic", "obscene", "insult")

LABEL_DISPLAY_NAMES = {
    "toxic": "Toxic",
    "obscene": "Obscene",
    "insult": "Insult",
    "non-toxic": "Clean",
}

DEFAULT_THRESHOLDS = (0.5, 0.5, 0.5)

MAX_LENGTH = 200

NLTK_PACKAGES = ("punkt", "punkt_tab", "wordnet", "omw-1.4")

ALBERT_BASE_MODEL = "albert-base-v2"

BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ("query", "value")
LORA_MODULES_TO_SAVE = ("classifier",)

RANDOM_SEED = 42
MAX_NEGATIVE_SAMPLES = 22000

BLIP_MODEL = "Salesforce/blip-image-captioning-base"
BLIP_MAX_NEW_TOKENS = 50

APP_TITLE = "Toxic Content Detector"
APP_ICON = "🛡️"


def resolve_lstm_checkpoint() -> Path | None:
    for candidate in LSTM_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None
