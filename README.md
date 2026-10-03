# 🛡️ Toxic Content Detector

Multi-modal toxicity screening for text comments and images, built with Streamlit.

---

## Flow

The app follows the assignment diagram literally: **input → model → verdict**.

| Step | What happens |
|---|---|
| 1 | The user types a comment, or uploads an image (the captioning model turns it into text) |
| 2 | The user picks a classification model |
| 3 | The model returns a verdict, stored in the CSV database |

---

## Models

| Model | Approach | Weights |
|---|---|---|
| **LSTM** | Bidirectional LSTM, 30k vocabulary, tuned per-label thresholds | 43 MB checkpoint |
| **ALBERT + LoRA** | `albert-base-v2` fine-tuned with rank-8 LoRA, trained adapter included | 0.81 macro-F1 on the held-out split |
| **BLIP-1** | Image captioning (`Salesforce/blip-image-captioning-base`) | downloaded on first use |

The LSTM and ALBERT models share one interface (`BaseClassifier`), so the UI, the database and the
tests never branch on which model produced a result.

---

## Quick start

```bash
git clone <this-repo>
cd toxic-content-detector

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The app is served on **http://localhost:7860**.

Or let `run.sh` handle everything — it checks the folders, installs dependencies, restores the LSTM
checkpoint when it is missing, and starts the server:

```bash
bash run.sh
```

### Training the ALBERT adapter

```bash
python -m training.train_albert_lora         # or: make train-albert
```

> **ALBERT needs `sentencepiece`.** Its tokenizer is a SentencePiece model, and without that package
> loading fails with *"Couldn't instantiate the backend tokenizer"*. It is listed in
> `requirements.txt`, so `pip install -r requirements.txt` covers it. The app also detects the
> missing package and says so in the sidebar instead of failing at prediction time.

The script re-balances the Jigsaw data, fine-tunes the LoRA adapter, then searches the per-label
decision threshold that maximises macro-F1 and writes it to `models/albert_lora/thresholds.json`.

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--max-length`, and `--max-rows` /
`--max-negatives` to train on a smaller slice (the full 159,571-row file needs a GPU).

The shipped adapter was trained on 9,600 balanced samples (64-token windows) with a 1,000× higher
learning rate on the classification head than on the LoRA weights, and reaches **0.81 macro-F1** on
the held-out split, with tuned per-label thresholds in `models/albert_lora/thresholds.json`.

---

## Dataset

**Jigsaw Toxic Comment Classification Challenge**
<https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge>

Place `train.csv` (159,571 rows) in `data/`. The file is intentionally not committed — it is ~68 MB
and openly downloadable from the URL above.

Three of the six labels are used, in this column order:

| Column | Meaning |
|---|---|
| `toxic` | General toxicity |
| `obscene` | Profanity |
| `insult` | Insulting language |

---

## Layout

```
toxic-content-detector/
├── app.py                      Streamlit entry point
├── imagecaption.py             BLIP-1 captioning module
├── config.py                   paths, labels and hyper-parameters
├── run.sh                      install, restore and launch
├── Makefile
│
├── classifiers/
│   ├── base.py                 BaseClassifier contract + ModelStatus
│   ├── lstm_classifier.py      ToxicLSTM + tuned thresholds from the checkpoint
│   ├── albert_classifier.py    PEFT adapter loader
│   └── registry.py             model lookup
│
├── core/
│   ├── schemas.py              ClassificationResult / LabelPrediction / HistoryRecord
│   ├── preprocessing.py        NLTK bootstrap, cleaner, LSTM encoder
│   └── database.py             ResultRepository over the CSV file
│
├── training/
│   ├── dataset.py              ToxicCommentDataset + Jigsaw re-balancing
│   └── train_albert_lora.py    fine-tune, threshold search, adapter export
│
├── ui/
│   ├── theme.py                dark design system
│   ├── components.py           model picker, run + persist, result renderer
│   └── views/                  text · image · database · system
│
├── research/
│   ├── quantization_research.md
│   ├── generate_figures.py
│   └── figures/
│
└── tests/
```

---

## Design notes

**Two text contracts.** The LSTM receives cleaned, lemmatised text — frozen to match the regexes its
checkpoint was trained with. ALBERT receives raw text, because its tokenizer already handles casing,
punctuation and sub-words.

**Status instead of exceptions.** Every model reports a `ModelStatus` (`ready` / `missing_weights` /
`unavailable`) plus a human sentence, so a missing adapter or a missing package degrades the app to
one working model and explains why, instead of crashing at prediction time.

**Models load once.** `functools.lru_cache` keeps the LSTM and BLIP weights resident across Streamlit
re-runs.

**`MAX_LENGTH` lives in one place.** Training and inference both read `config.MAX_LENGTH`, so they
cannot drift apart.

**NLTK bootstraps itself.** `core.preprocessing.ensure_nltk_data()` downloads `punkt`, `punkt_tab`,
`wordnet` and `omw-1.4` on first run and caches them. Without it, `word_tokenize` raises
`LookupError` on any clean machine.

**The database heals itself.** `ResultRepository.load()` re-creates missing columns and recovers from
a corrupted file instead of raising, and writing is append-only.

---

## Research deliverable

`research/quantization_research.md` — 18 sections covering the memory equation, affine
quantization, PTQ vs QAT, NF4, LLM.int8(), QLoRA and a worked NumPy example, with three generated
figures:

<p align="center">
  <img src="research/figures/01_memory_footprint.png" width="46%">
  <img src="research/figures/03_compression_ratio.png" width="46%">
</p>

Rebuild them with `python research/generate_figures.py` (or `make figures`).

---

## Tests

```bash
pytest -q
```

Covers text cleaning, encoding, the CSV repository (including corrupted files), verdict logic, the
availability contract, and real inference through the LSTM when its checkpoint is present.

---

## Requirements

Python 3.10+. See `requirements.txt`. A CUDA GPU is optional — everything runs on CPU.
