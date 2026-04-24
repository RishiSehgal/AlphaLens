"""Sidebar renderer: metrics, sources, confidence, latency, cache status."""

import streamlit as st

from src.config import COLORS
from src.state import AlphaLensState
from src.ui.components import render_confidence_bar, render_metric_card


def _fmt(val, fmt=".2f", prefix="", suffix="", fallback="N/A") -> str:
    if val is None:
        return fallback
    try:
        return f"{prefix}{float(val):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def render_sidebar(state: AlphaLensState) -> None:
    """Render the complete sidebar content for a completed pipeline run.

    Args:
        state: Completed AlphaLensState.
    """
    with st.sidebar:
        meta = state.get("metadata", {})
        report = state.get("report", {})
        sections = report.get("sections", {})
        verification = state.get("verification", {})
        fd = state.get("financial_data", {})

        # ── Section heading helper ────────────────────────────────────────────
        def _heading(text: str) -> None:
            st.markdown(
                f'<div style="font-size:11px;font-weight:700;letter-spacing:0.08em;'
                f'text-transform:uppercase;color:{COLORS["text_muted"]};'
                f'font-family:Inter,sans-serif;margin:16px 0 8px 0;">{text}</div>',
                unsafe_allow_html=True,
            )

        # ── Generation metrics ────────────────────────────────────────────────
        _heading("Pipeline Stats")

        total_s = meta.get("pipeline_total_seconds")
        st.markdown(
            render_metric_card("Generation Time", _fmt(total_s, ".1f", suffix="s"), color=COLORS["accent"]),
            unsafe_allow_html=True,
        )

        n_errors = len(state.get("error_log", []))
        err_color = COLORS["red"] if n_errors else COLORS["green"]
        st.markdown(
            render_metric_card(
                "Errors",
                str(n_errors),
                subtitle="pipeline errors" if n_errors else "clean run",
                color=err_color,
            ),
            unsafe_allow_html=True,
        )

        n_chunks = len(state.get("rag_chunks", []))
        st.markdown(
            render_metric_card("RAG Chunks", str(n_chunks), color=COLORS["text_muted"]),
            unsafe_allow_html=True,
        )

        # ── Sources breakdown ─────────────────────────────────────────────────
        sources_status = fd.get("sources_status", meta.get("sources_status", {}))
        if sources_status:
            _heading("Data Sources")
            for source, status in sources_status.items():
                is_ok = str(status).lower() in ("ok", "success", "true", "1")
                dot_color = COLORS["green"] if is_ok else COLORS["amber"]
                label = source.replace("_", " ").title()
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;'
                    f'padding:5px 0;font-size:13px;font-family:Inter,sans-serif;">'
                    f'<span style="color:{dot_color};font-size:10px;">●</span>'
                    f'<span style="color:{COLORS["text_secondary"]};">{label}</span>'
                    f'<span style="margin-left:auto;font-size:11px;color:{COLORS["text_muted"]};">'
                    f'{"OK" if is_ok else "degraded"}</span></div>',
                    unsafe_allow_html=True,
                )

        # ── Confidence bars ───────────────────────────────────────────────────
        conf_scores = verification.get("confidence_scores", {})
        section_confs = {k: v.get("confidence") for k, v in sections.items() if isinstance(v, dict) and v.get("confidence")}
        all_conf = {**section_confs, **conf_scores}

        if all_conf:
            _heading("Confidence Scores")
            for k, v in all_conf.items():
                label = k.replace("_", " ").title()
                st.markdown(render_confidence_bar(label, str(v)), unsafe_allow_html=True)

        # ── Agent latency breakdown ───────────────────────────────────────────
        latencies = meta.get("agent_latencies", {})
        if latencies:
            _heading("Agent Latencies")
            _DISPLAY = {
                "data_fusion": "Data Fusion",
                "rag_citation": "RAG Citation",
                "quant_analysis": "Quant Analysis",
                "risk_scanner": "Risk Scanner",
                "verification": "Verification",
                "report_synthesis": "Report Synthesis",
            }
            for node, t in latencies.items():
                label = _DISPLAY.get(node, node.replace("_", " ").title())
                t_str = f"{t:.1f}s" if isinstance(t, (int, float)) else str(t)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:4px 0;font-size:12px;font-family:Inter,sans-serif;">'
                    f'<span style="color:{COLORS["text_secondary"]};">{label}</span>'
                    f'<span style="color:{COLORS["text_muted"]};">{t_str}</span></div>',
                    unsafe_allow_html=True,
                )

        # ── Cache status ──────────────────────────────────────────────────────
        rag_available = meta.get("rag_available", True)
        rag_filing_date = meta.get("rag_filing_date", "")
        _heading("Cache")
        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["text_secondary"]};'
            f'font-family:Inter,sans-serif;line-height:1.6;">'
            f'RAG: <span style="color:{COLORS["green"] if rag_available else COLORS["amber"]};">'
            f'{"Available" if rag_available else "Unavailable"}</span>'
            + (f"<br>Filing: {rag_filing_date}" if rag_filing_date else "")
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── Error log (collapsed) ─────────────────────────────────────────────
        error_log = state.get("error_log", [])
        if error_log:
            with st.expander(f"⚠ {len(error_log)} Error(s)", expanded=False):
                for err in error_log:
                    st.markdown(
                        f'<div style="font-size:11px;color:{COLORS["red"]};'
                        f'font-family:monospace;padding:2px 0;">{err}</div>',
                        unsafe_allow_html=True,
                    )
