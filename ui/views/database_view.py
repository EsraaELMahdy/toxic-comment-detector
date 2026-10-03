from __future__ import annotations

import streamlit as st

from core.database import ResultRepository
from ui import theme


def render() -> None:
    theme.hero("Stored results", mark="🗄️")

    repository = ResultRepository()
    frame = repository.load()

    if frame.empty:
        st.info("No records yet — run an analysis first.")
        return

    total = len(frame)
    flagged = int((frame["classification"].str.lower() != "non-toxic").sum())
    clean = total - flagged

    columns = st.columns(4)
    for column, (label, value) in zip(
        columns,
        [
            ("Records", total),
            ("Flagged", flagged),
            ("Clean", clean),
            ("Models used", frame["model"].nunique()),
        ],
    ):
        with column:
            st.metric(label, value)

    st.write("")
    filter_col, toggle_col = st.columns([3, 1])
    with filter_col:
        model_filter = st.multiselect(
            "Filter by model",
            options=sorted(frame["model"].unique().tolist()),
            default=[],
        )
    with toggle_col:
        st.write("")
        only_flagged = st.toggle("Only flagged", value=False)

    view = frame.copy()
    if model_filter:
        view = view[view["model"].isin(model_filter)]
    if only_flagged:
        view = view[view["classification"].str.lower() != "non-toxic"]

    st.dataframe(view.iloc[::-1], width="stretch", hide_index=True)
    st.caption(f"Showing {len(view)} of {total} records · newest first")

    download_col, clear_col = st.columns([3, 1])
    with download_col:
        st.download_button(
            "⬇️  Download CSV",
            data=repository.to_csv_bytes(),
            file_name="toxic_detector_history.csv",
            mime="text/csv",
            width="stretch",
        )
    with clear_col:
        if st.button("🗑️  Clear", width="stretch"):
            repository.clear()
            st.success("History cleared.")
            st.rerun()
