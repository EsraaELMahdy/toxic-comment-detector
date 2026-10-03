#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-7860}"
CHECKPOINT="models/lstm/full_toxic_lstm_checkpoint.pth"
REPO="https://github.com/EsraaELMahdy/toxic-comment-detector.git"

mkdir -p models/lstm models/albert_lora data assets database
touch models/albert_lora/.gitkeep data/.gitkeep assets/.gitkeep

if [ -f streamlit_config.toml ]; then
    mkdir -p .streamlit
    cp streamlit_config.toml .streamlit/config.toml
fi

if ! python3 -c "import streamlit, torch, transformers, peft, sentencepiece, nltk, pandas" 2>/dev/null; then
    echo "▸ installing dependencies"
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    pip install --quiet -r requirements.txt
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "▸ fetching the LSTM checkpoint"
    tmp="$(mktemp -d)"
    git -C "$tmp" init -q
    git -C "$tmp" remote add origin "$REPO"
    git -C "$tmp" fetch --depth 1 origin main -q
    git -C "$tmp" checkout FETCH_HEAD -- full_toxic_lstm_checkpoint.pth
    mv "$tmp/full_toxic_lstm_checkpoint.pth" "$CHECKPOINT"
    rm -rf "$tmp"
fi

echo "▸ starting streamlit on port ${PORT}"
exec streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0
