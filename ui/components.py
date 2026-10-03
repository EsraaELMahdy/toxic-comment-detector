from __future__ import annotations

import logging

import streamlit as st

from classifiers.base import BaseClassifier, ModelStatus
from classifiers.registry import all_classifiers, ready_classifiers
from core.database import ResultRepository
from core.schemas import ClassificationResult, HistoryRecord
from ui import theme

logger = logging.getLogger(__name__)

_LAST_RESULT_KEY = "last_result"
_LAST_META_KEY = "last_result_meta"


def pick_classifier(key: str) -> BaseClassifier | None:
    available = ready_classifiers()

    if not available:
        problems = [
            f"{classifier.display_name}: {reason}"
            for classifier in all_classifiers()
            for state, reason in [classifier.availability()]
            if state is not ModelStatus.READY
        ]
        st.error("No classifier is available.")
        for problem in problems:
            st.caption(problem)
        return None

    if len(available) == 1:
        st.caption(f"Using **{available[0].display_name}**")
        return available[0]

    names = [classifier.display_name for classifier in available]
    by_name = {classifier.display_name: classifier for classifier in available}

    chosen = st.segmented_control(
        "Model",
        options=names,
        default=names[0],
        key=f"{key}_model",
        label_visibility="collapsed",
    )

    return by_name.get(chosen or names[0], available[0])


def run_classification(classifier: BaseClassifier, text: str, input_type: str):
    with st.spinner("Analysing…"):
        try:
            result = classifier.predict(text)
        except Exception as error:
            logger.exception("Inference failed for %s", classifier.model_id)
            st.error(f"**{classifier.display_name} could not run.** {error}")
            return None

    saved = True
    try:
        ResultRepository().save(
            HistoryRecord(
                created_at=result.created_at,
                input_type=input_type,
                input_text=result.input_text,
                model_label=result.model_label,
                classification=result.verdict,
                probabilities=result.probabilities_json,
                latency_ms=result.latency_ms,
            )
        )
    except Exception as error:
        logger.exception("Could not persist result")
        saved = False

    st.session_state[_LAST_RESULT_KEY] = result
    st.session_state[_LAST_META_KEY] = {"saved": saved}
    return result


def render_last_result() -> None:
    result: ClassificationResult | None = st.session_state.get(_LAST_RESULT_KEY)
    if result is None:
        return

    meta = st.session_state.get(_LAST_META_KEY, {})
    theme.step(3, "Result")
    theme.verdict_banner(result)
    theme.probability_cards(result)

    chips = [f"{result.latency_ms:.0f} ms", result.model_label]
    if meta.get("saved"):
        chips.append("saved")

    theme.chips(chips)

    with st.expander("Details"):
        st.code(result.input_text or "(empty)", language=None)
        st.dataframe(
            [
                {
                    "label": prediction.display_name,
                    "probability": round(prediction.probability, 4),
                    "threshold": prediction.threshold,
                    "flagged": prediction.positive,
                }
                for prediction in result.predictions
            ],
            width="stretch",
            hide_index=True,
        )


def clear_last_result() -> None:
    st.session_state.pop(_LAST_RESULT_KEY, None)
    st.session_state.pop(_LAST_META_KEY, None)
