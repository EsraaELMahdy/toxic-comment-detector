from __future__ import annotations

import platform
import sys

import streamlit as st

import imagecaption
from classifiers.registry import all_classifiers
from core.preprocessing import ensure_nltk_data
from ui import theme


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"
    except ImportError:
        return "unknown"


def render() -> None:
    theme.hero("System", mark="⚙️")

    blip_ready, blip_reason = imagecaption.is_available()
    nltk_ready = ensure_nltk_data()

    left, middle, right = st.columns(3)
    with left:
        st.metric("Device", _device())
    with middle:
        st.metric("Python", platform.python_version())
        st.caption(f"{sys.platform} · {platform.machine()}")
    with right:
        st.metric("NLTK data", "ready" if nltk_ready else "missing")

    st.divider()

    for classifier in all_classifiers():
        state, reason = classifier.availability()
        line = classifier.display_name if state.value == "ready" else f"{classifier.display_name} — {reason}"
        st.markdown(f"- {line}")

    st.markdown("- BLIP-1" if blip_ready else f"- BLIP-1 — {blip_reason}")
