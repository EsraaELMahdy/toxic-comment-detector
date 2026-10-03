from __future__ import annotations

import streamlit as st

from ui import theme
from ui.components import pick_classifier, render_last_result, run_classification


def render() -> None:
    theme.step(1, "Enter a comment")

    text = st.text_area(
        "Comment",
        height=160,
        placeholder="Type or paste a comment…",
        key="text_input",
        label_visibility="collapsed",
    )

    st.divider()

    theme.step(2, "Choose the model")
    classifier = pick_classifier(key="text")

    if classifier is None:
        return

    st.write("")
    if st.button("Analyse", type="primary", width="stretch", key="run_text"):
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            run_classification(classifier, text, input_type="text")

    st.divider()
    render_last_result()
