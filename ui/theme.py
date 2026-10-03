from __future__ import annotations

import html

import streamlit as st

from core.schemas import ClassificationResult, LabelPrediction

_CSS = """
<style>
    :root {
        --bg:       #080B14;
        --panel:    rgba(255,255,255,.035);
        --line:     rgba(255,255,255,.08);
        --line-2:   rgba(255,255,255,.16);
        --ink:      #E8EEF9;
        --muted:    #8A99B3;
        --cyan:     #22D3EE;
        --indigo:   #818CF8;
        --danger:   #FB7185;
        --success:  #34D399;
        --shadow:   0 24px 60px -34px rgba(0,0,0,.95);
    }

    .stApp {
        background:
            radial-gradient(950px 520px at 6% -8%,  rgba(34,211,238,.10), transparent 60%),
            radial-gradient(820px 470px at 96% 2%,  rgba(129,140,248,.13), transparent 58%),
            var(--bg);
    }
    html, body, [class*="css"] {
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1180px; }

    .hero {
        position: relative; overflow: hidden;
        border-radius: 20px; padding: 22px 26px; margin-bottom: 22px;
        background: linear-gradient(135deg, rgba(34,211,238,.11), rgba(129,140,248,.13) 55%);
        border: 1px solid var(--line-2);
        box-shadow: var(--shadow);
    }
    .hero .row { display: flex; align-items: center; gap: 14px; position: relative; z-index: 1; }
    .hero .mark {
        width: 44px; height: 44px; flex: 0 0 44px; border-radius: 13px;
        display: grid; place-items: center; font-size: 1.35rem;
        background: linear-gradient(140deg, var(--cyan), var(--indigo));
        box-shadow: 0 10px 26px -12px rgba(34,211,238,.85);
    }
    .hero h1 { margin: 0; font-size: 1.45rem; font-weight: 700; letter-spacing: -.4px; color: var(--ink); }

    .step { display: flex; align-items: center; gap: 10px; margin: 2px 0 14px; }
    .step .n {
        width: 22px; height: 22px; flex: 0 0 22px; border-radius: 7px;
        display: grid; place-items: center; font-size: .72rem; font-weight: 700;
        background: rgba(34,211,238,.14); color: var(--cyan);
        border: 1px solid rgba(34,211,238,.32);
    }
    .step .t { font-weight: 650; font-size: .99rem; color: var(--ink); }

    .metric-grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); }
    .metric-card {
        position: relative; overflow: hidden;
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 16px; padding: 15px 16px;
    }
    .metric-card::before {
        content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
        background: linear-gradient(90deg, var(--cyan), var(--indigo));
        opacity: .55;
    }
    .metric-card .name {
        font-size: .7rem; letter-spacing: 1px; text-transform: uppercase;
        color: var(--muted); font-weight: 700;
    }
    .metric-card .value { font-size: 1.6rem; font-weight: 750; color: var(--ink); margin: 6px 0 3px; letter-spacing: -.5px; }
    .metric-card .hint { font-size: .72rem; color: var(--muted); }
    .metric-card .bar { height: 6px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden; margin-top: 11px; }
    .metric-card .bar > span { display: block; height: 100%; border-radius: 99px; }
    .metric-card.flagged { border-color: rgba(251,113,133,.34); background: linear-gradient(180deg, rgba(251,113,133,.10), rgba(251,113,133,.03)); }
    .metric-card.flagged .value { color: var(--danger); }
    .metric-card.flagged::before { background: linear-gradient(90deg, var(--danger), #FBBF24); opacity: .8; }
    .metric-card.clean .value { color: var(--success); }
    .metric-card.clean::before { background: linear-gradient(90deg, var(--success), var(--cyan)); opacity: .8; }

    .verdict {
        position: relative; overflow: hidden;
        border-radius: 18px; padding: 19px 23px; margin: 4px 0 18px;
        display: flex; align-items: center; gap: 17px;
    }
    .verdict .icon {
        width: 50px; height: 50px; flex: 0 0 50px; border-radius: 16px;
        display: grid; place-items: center; font-size: 1.5rem;
    }
    .verdict .title { font-weight: 700; font-size: 1.09rem; letter-spacing: -.2px; }
    .verdict.bad  { background: linear-gradient(120deg, rgba(251,113,133,.15), rgba(251,113,133,.04)); border: 1px solid rgba(251,113,133,.34); }
    .verdict.bad .icon  { background: rgba(251,113,133,.16); border: 1px solid rgba(251,113,133,.36); }
    .verdict.bad .title { color: #FFE1E7; }
    .verdict.good { background: linear-gradient(120deg, rgba(52,211,153,.15), rgba(52,211,153,.04)); border: 1px solid rgba(52,211,153,.34); }
    .verdict.good .icon  { background: rgba(52,211,153,.16); border: 1px solid rgba(52,211,153,.36); }
    .verdict.good .title { color: #CFFAEA; }

    .chips-row { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip-meta {
        font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .75rem;
        padding: 4px 11px; border-radius: 9px; color: #9FB0CC;
        background: rgba(255,255,255,.045); border: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0C1220, #0A0F1B);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel); border-color: var(--line); border-radius: 16px;
    }
    .stTextArea textarea, .stTextInput input {
        background: rgba(255,255,255,.04) !important;
        border: 1px solid var(--line) !important; border-radius: 12px !important;
        color: var(--ink) !important; font-size: .92rem !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(34,211,238,.55) !important;
        box-shadow: 0 0 0 3px rgba(34,211,238,.12) !important;
    }
    .stButton > button {
        border-radius: 11px; font-weight: 620; border: 1px solid var(--line-2);
        transition: all .16s ease;
    }
    .stButton > button[kind="secondary"]:hover { border-color: rgba(34,211,238,.45); color: var(--cyan); }
    .stButton > button[kind="primary"] {
        background: linear-gradient(120deg, var(--cyan), var(--indigo));
        border: none; color: #04121A; font-weight: 700;
        box-shadow: 0 14px 32px -18px rgba(34,211,238,.9);
    }
    .stButton > button[kind="primary"]:hover { filter: brightness(1.08); }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] {
        background: rgba(255,255,255,.04) !important; border-color: var(--line) !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0; padding: 9px 20px; color: var(--muted); font-weight: 600;
    }
    .stTabs [aria-selected="true"] { color: var(--ink) !important; background: var(--panel); }
    div[data-testid="stMetricValue"] { color: var(--ink); }
    .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, mark: str = "🛡️") -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="row">
                <div class="mark">{mark}</div>
                <div><h1>{html.escape(title)}</h1></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step(number: int, title: str) -> None:
    st.markdown(
        f'<div class="step"><div class="n">{number}</div><div class="t">{html.escape(title)}</div></div>',
        unsafe_allow_html=True,
    )


def chips(items: list[str]) -> None:
    rendered = "".join(f'<span class="chip-meta">{html.escape(item)}</span>' for item in items)
    st.markdown(f'<div class="chips-row">{rendered}</div>', unsafe_allow_html=True)


def verdict_banner(result: ClassificationResult) -> None:
    if result.is_clean:
        icon, kind = "✅", "good"
        title = "No toxic content detected"
    else:
        icon, kind = "⚠️", "bad"
        title = f"Toxic content detected — {', '.join(p.display_name for p in result.detected)}"

    st.markdown(
        f"""
        <div class="verdict {kind}">
            <div class="icon">{icon}</div>
            <div><div class="title">{html.escape(title)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_cards(result: ClassificationResult) -> None:
    cards = "".join(_metric_card(prediction) for prediction in result.predictions)
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def _metric_card(prediction: LabelPrediction) -> str:
    ratio = max(0.0, min(1.0, prediction.probability))
    state = "flagged" if prediction.positive else "clean"
    colour = "var(--danger)" if prediction.positive else "var(--success)"

    return f"""
    <div class="metric-card {state}">
        <div class="name">{html.escape(prediction.display_name)}</div>
        <div class="value">{prediction.probability:.1%}</div>
        <div class="bar"><span style="width:{ratio * 100:.1f}%;background:{colour}"></span></div>
    </div>
    """
