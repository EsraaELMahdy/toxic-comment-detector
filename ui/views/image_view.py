from __future__ import annotations

import streamlit as st
from PIL import Image

import imagecaption
from ui import theme
from ui.components import pick_classifier, render_last_result, run_classification


def render() -> None:
    theme.step(1, "Upload an image")

    uploaded = st.file_uploader(
        "Image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
        label_visibility="collapsed",
    )

    if uploaded is None:
        render_last_result()
        return

    image = Image.open(uploaded).convert("RGB")

    preview, controls = st.columns([1, 1], gap="large")

    with preview:
        st.image(image, width="stretch")

    with controls:
        st.divider()
        theme.step(2, "Choose the model")
        classifier = pick_classifier(key="image")

        if classifier is None:
            return

        st.write("")
        if st.button("Caption & analyse", type="primary", width="stretch", key="run_image"):
            with st.spinner("Describing the image…"):
                caption_result = imagecaption.generate_caption(image)

            st.write("**Caption**")
            st.info(caption_result.caption)

            run_classification(classifier, caption_result.caption, input_type="image_caption")

    st.divider()
    render_last_result()
