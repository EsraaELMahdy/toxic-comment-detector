from __future__ import annotations

import logging

import streamlit as st

import config
from core.database import ResultRepository
from core.preprocessing import ensure_nltk_data
from ui import theme
from ui.views import database_view, image_view, system_view, text_view

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

PAGES = {
    "Analyse": {"icon": "🔍", "render": None},
    "History": {"icon": "🗄️", "render": database_view.render},
    "System": {"icon": "⚙️", "render": system_view.render},
}


def configure_page() -> None:
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon=config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.inject_theme()


def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:10px'>"
            "<div style='width:32px;height:32px;border-radius:10px;display:grid;place-items:center;"
            "font-size:1rem;background:linear-gradient(140deg,#22D3EE,#818CF8)'>🛡️</div>"
            f"<div style='font-weight:700;color:#E8EEF9;font-size:.95rem'>{config.APP_TITLE}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        page = st.radio(
            "Navigation",
            options=list(PAGES),
            format_func=lambda name: f"{PAGES[name]['icon']}  {name}",
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(f"{ResultRepository().count()} record(s) stored")

    return page


def analyse_page() -> None:
    theme.hero(config.APP_TITLE)

    tab_text, tab_image = st.tabs(["✍️   Text comment", "🖼️   Image"])

    with tab_text:
        text_view.render()

    with tab_image:
        image_view.render()


def main() -> None:
    configure_page()
    ensure_nltk_data()

    page = sidebar()

    if page == "Analyse":
        analyse_page()
    else:
        PAGES[page]["render"]()


if __name__ == "__main__":
    main()
