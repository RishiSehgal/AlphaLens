"""AlphaLens — Streamlit entry point.

Orchestrates the full UI: ticker input, streaming agent progress, report,
sidebar, charts, and follow-up chat. All styling is custom HTML/CSS — no
default Streamlit look.
"""

import time
import logging

import streamlit as st

from src.config import COLORS
from src.graph import get_node_order, stream_pipeline
from src.state import AlphaLensState
from src.ui.chat import render_chat
from src.ui.components import render_agent_progress, render_footer, render_header
from src.ui.report_view import render_full_report
from src.ui.sidebar import render_sidebar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="AlphaLens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — overrides ALL default Streamlit styling ─────────────────────
GLOBAL_CSS = f"""
<style>
    /* Hide Streamlit chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    [data-testid="stToolbar"] {{display: none;}}

    /* App background */
    .stApp {{
        background-color: {COLORS['bg_primary']};
    }}
    .block-container {{
        padding: 1.5rem 2rem 2rem 2rem;
        max-width: 1400px;
    }}

    /* Text inputs */
    .stTextInput input {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['text_primary']};
        padding: 12px 16px;
        font-size: 15px;
        font-family: Inter, sans-serif;
        transition: border-color 0.2s, box-shadow 0.2s;
    }}
    .stTextInput input:focus {{
        border-color: {COLORS['accent']};
        box-shadow: 0 0 0 2px rgba(59,130,246,0.18);
        outline: none;
    }}
    .stTextInput input::placeholder {{
        color: {COLORS['text_muted']};
    }}

    /* Primary button */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {COLORS['accent']}, #8B5CF6);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        font-size: 15px;
        font-family: Inter, sans-serif;
        transition: opacity 0.2s, transform 0.1s;
        width: 100%;
    }}
    .stButton > button[kind="primary"]:hover {{
        opacity: 0.88;
        transform: translateY(-1px);
    }}
    .stButton > button[kind="primary"]:active {{
        transform: translateY(0);
    }}

    /* Secondary button */
    .stButton > button:not([kind="primary"]) {{
        background-color: {COLORS['bg_elevated']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['text_secondary']};
        font-family: Inter, sans-serif;
        font-size: 14px;
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['text_secondary']};
        font-size: 13px;
        font-family: Inter, sans-serif;
    }}
    .streamlit-expanderContent {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-top: none;
        border-radius: 0 0 8px 8px;
    }}

    /* Chat input */
    .stChatInput textarea {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        color: {COLORS['text_primary']};
        font-family: Inter, sans-serif;
        border-radius: 8px;
    }}
    .stChatInput textarea:focus {{
        border-color: {COLORS['accent']};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['bg_primary']};
        border-right: 1px solid {COLORS['border']};
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding: 1.5rem 1rem;
    }}

    /* Plotly containers */
    .stPlotlyChart {{
        border-radius: 8px;
        overflow: hidden;
        background: {COLORS['bg_card']};
    }}

    /* Spinner */
    .stSpinner > div > div {{
        border-top-color: {COLORS['accent']} !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: {COLORS['bg_primary']}; }}
    ::-webkit-scrollbar-thumb {{ background: {COLORS['border_hover']}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {COLORS['text_muted']}; }}

    /* Columns gap */
    [data-testid="column"] {{ padding: 0 6px; }}

    /* All Streamlit text defaults */
    body, p, div, span, li, td, th {{
        font-family: Inter, -apple-system, sans-serif;
    }}

    /* Alert / info boxes */
    .stAlert {{
        background-color: {COLORS['bg_elevated']};
        border-color: {COLORS['border']};
        color: {COLORS['text_secondary']};
        border-radius: 8px;
    }}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
def _init_session() -> None:
    defaults = {
        "result": None,
        "running": False,
        "agents_status": {},
        "last_ticker": "",
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(render_header(), unsafe_allow_html=True)

# ── Ticker input row ──────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    ticker_input = st.text_input(
        label="ticker",
        label_visibility="collapsed",
        placeholder="Enter ticker (e.g. NVDA, AAPL, MSFT)",
        value=st.session_state.get("last_ticker", ""),
        key="ticker_field",
    )
with col_btn:
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

# ── Disclaimer banner ─────────────────────────────────────────────────────────
st.markdown(
    f'<div style="font-size:11px;color:{COLORS["text_muted"]};font-family:Inter,sans-serif;'
    f'padding:4px 0 16px 0;">Not financial advice. For educational purposes only.</div>',
    unsafe_allow_html=True,
)

# ── Layout: main (7) + sidebar already rendered by render_sidebar ─────────────
col_main, col_charts = st.columns([7, 3])


def _run_analysis(ticker: str) -> None:
    """Stream the pipeline and progressively update session state."""
    st.session_state.running = True
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.last_ticker = ticker

    # Initialise all agents as waiting
    node_order = get_node_order()
    st.session_state.agents_status = {n: "waiting" for n in node_order}

    # Track which nodes have started based on pipeline edges
    # data_fusion → (rag_citation, quant_analysis) → risk_scanner → verify → report_synthesis
    _STARTS_AFTER: dict[str, str | None] = {
        "data_fusion": None,
        "rag_citation": "data_fusion",
        "quant_analysis": "data_fusion",
        "risk_scanner": "rag_citation",
        "verify": "risk_scanner",
        "report_synthesis": "verify",
    }

    pipeline_start = time.time()
    accumulated_state: AlphaLensState = {"ticker": ticker}  # type: ignore[assignment]

    progress_placeholder = st.empty()

    def _mark_running(node: str) -> None:
        for n, after in _STARTS_AFTER.items():
            if after == node and st.session_state.agents_status.get(n) == "waiting":
                st.session_state.agents_status[n] = "running"

    # Mark data_fusion as running immediately
    st.session_state.agents_status["data_fusion"] = "running"

    for event in stream_pipeline(ticker):
        node = event.get("node", "")
        partial = event.get("partial_state", {})
        elapsed = event.get("elapsed", 0.0)

        if node == "error":
            st.session_state.agents_status["data_fusion"] = "error"
        else:
            # Mark completed
            t_str = f"done ({elapsed:.1f}s)"
            st.session_state.agents_status[node] = t_str
            # Mark next node(s) as running
            _mark_running(node)
            # Merge partial state
            for k, v in partial.items():
                if k == "error_log" and isinstance(v, list):
                    existing = accumulated_state.get("error_log", [])
                    accumulated_state["error_log"] = existing + v  # type: ignore[assignment]
                elif k == "metadata" and isinstance(v, dict):
                    m = accumulated_state.get("metadata", {})
                    m.update(v)
                    accumulated_state["metadata"] = m  # type: ignore[assignment]
                else:
                    accumulated_state[k] = v  # type: ignore[assignment]

        # Re-render progress
        progress_placeholder.markdown(
            render_agent_progress(st.session_state.agents_status),
            unsafe_allow_html=True,
        )

    # Add pipeline total to metadata
    if "metadata" not in accumulated_state:
        accumulated_state["metadata"] = {}  # type: ignore[assignment]
    accumulated_state["metadata"]["pipeline_total_seconds"] = round(time.time() - pipeline_start, 2)  # type: ignore[index]

    st.session_state.result = accumulated_state
    st.session_state.running = False
    progress_placeholder.empty()
    st.rerun()


# ── Trigger analysis ──────────────────────────────────────────────────────────
if analyze_clicked and ticker_input.strip():
    ticker_clean = ticker_input.strip().upper()
    with col_main:
        _run_analysis(ticker_clean)

elif analyze_clicked and not ticker_input.strip():
    with col_main:
        st.markdown(
            f'<div style="color:{COLORS["amber"]};font-size:13px;font-family:Inter,sans-serif;'
            f'padding:8px 0;">Please enter a ticker symbol.</div>',
            unsafe_allow_html=True,
        )

# ── Render completed result ───────────────────────────────────────────────────
if st.session_state.result:
    state: AlphaLensState = st.session_state.result

    # Sidebar
    render_sidebar(state)

    with col_main:
        # Report confidence badge
        report = state.get("report", {})
        overall_conf = report.get("overall_confidence", "")
        if overall_conf:
            conf_color = {
                "HIGH": COLORS["green"],
                "MEDIUM": COLORS["amber"],
                "LOW": COLORS["red"],
            }.get(overall_conf.upper(), COLORS["text_muted"])
            st.markdown(
                f'<div style="display:inline-block;background:{conf_color}22;'
                f'border:1px solid {conf_color}55;color:{conf_color};'
                f'font-size:12px;font-weight:700;letter-spacing:0.06em;'
                f'padding:4px 14px;border-radius:14px;font-family:Inter,sans-serif;'
                f'margin-bottom:16px;">OVERALL CONFIDENCE · {overall_conf.upper()}</div>',
                unsafe_allow_html=True,
            )

        render_full_report(state)

        # Chat widget
        render_chat(state)

    st.markdown(render_footer(), unsafe_allow_html=True)

# ── Empty state (no result yet) ───────────────────────────────────────────────
elif not st.session_state.running:
    with col_main:
        _agent_tags = "".join(
            f'<div style="background:{COLORS["bg_elevated"]};border:1px solid {COLORS["border"]};'
            f'border-radius:8px;padding:10px 18px;'
            f'font-size:12px;color:{COLORS["text_muted"]};font-family:Inter,sans-serif;">'
            f'{agent}</div>'
            for agent in [
                "Data Fusion", "RAG Citation", "Quant Analysis",
                "Risk Scanner", "Verification", "Report Synthesis",
            ]
        )
        st.markdown(
            f'<div style="background:{COLORS["bg_card"]};border:1px solid {COLORS["border"]};'
            f'border-radius:12px;padding:3rem 2rem;text-align:center;margin-top:1rem;">'
            f'<div style="font-size:36px;margin-bottom:16px;">📊</div>'
            f'<div style="font-size:18px;font-weight:600;color:{COLORS["text_primary"]};'
            f'font-family:Inter,sans-serif;margin-bottom:10px;">AI Equity Research in 90 Seconds</div>'
            f'<div style="font-size:14px;color:{COLORS["text_secondary"]};'
            f'font-family:Inter,sans-serif;line-height:1.7;max-width:500px;margin:0 auto;">'
            f'Enter a ticker above to generate a full equity research report powered by '
            f'6 AI agents — covering financials, valuation, risk flags, and verified data.</div>'
            f'<div style="display:flex;justify-content:center;gap:24px;margin-top:24px;flex-wrap:wrap;">'
            f'{_agent_tags}</div></div>',
            unsafe_allow_html=True,
        )

    # Sidebar placeholder
    with st.sidebar:
        st.markdown(
            f'<div style="font-size:13px;color:{COLORS["text_muted"]};'
            f'font-family:Inter,sans-serif;padding:1rem 0;text-align:center;">'
            f'Run an analysis to see metrics here.</div>',
            unsafe_allow_html=True,
        )
