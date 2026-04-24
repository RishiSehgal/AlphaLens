"""Render the full AlphaLens report using custom HTML/CSS components."""

import streamlit as st

from src.config import COLORS
from src.state import AlphaLensState
from src.ui.charts import (
    create_dcf_heatmap,
    create_earnings_chart,
    create_price_macd_chart,
    create_rsi_gauge,
)
from src.ui.components import (
    render_confidence_bar,
    render_divergence,
    render_report_section,
    render_risk_flag,
)


def _fmt(val, fmt=".2f", prefix="", suffix="", fallback="N/A") -> str:
    """Safe number formatter."""
    if val is None:
        return fallback
    try:
        return f"{prefix}{float(val):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def _billions(val) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if abs(v) >= 1e9:
            return f"${v/1e9:.1f}B"
        if abs(v) >= 1e6:
            return f"${v/1e6:.1f}M"
        return f"${v:.0f}"
    except (TypeError, ValueError):
        return "N/A"


def render_full_report(state: AlphaLensState) -> None:
    """Render all report sections, charts, and risk flags into the Streamlit main area.

    Args:
        state: Completed AlphaLensState with all agent outputs populated.
    """
    report = state.get("report", {})
    sections = report.get("sections", {})
    ticker = state.get("ticker", "")
    fd = state.get("financial_data", {})
    quant = state.get("quant_results", {})
    risk_flags = state.get("risk_flags", [])
    verification = state.get("verification", {})

    if not sections:
        st.markdown(
            f'<div style="color:{COLORS["text_muted"]};font-size:14px;text-align:center;'
            f'padding:3rem;">No report sections generated. Check error log.</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Executive Summary ─────────────────────────────────────────────────────
    _render_section(sections, "executive_summary", "Executive Summary")

    # ── Key Metrics Row ───────────────────────────────────────────────────────
    _render_key_metrics(fd)

    # ── Price & MACD Chart ────────────────────────────────────────────────────
    price_fig = create_price_macd_chart(ticker)
    if price_fig:
        st.plotly_chart(price_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Financial Health ──────────────────────────────────────────────────────
    _render_section(sections, "financial_health", "Financial Health")

    # ── Valuation + Charts ────────────────────────────────────────────────────
    _render_section(sections, "valuation", "Valuation")

    dcf = quant.get("dcf", {})
    technicals = quant.get("technicals", {})
    if dcf or technicals:
        chart_cols = st.columns([3, 2])
        with chart_cols[0]:
            heatmap = create_dcf_heatmap(quant)
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True, config={"displayModeBar": False})
        with chart_cols[1]:
            rsi = technicals.get("rsi")
            if rsi is not None:
                gauge = create_rsi_gauge(rsi)
                if gauge:
                    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

    earnings_fig = create_earnings_chart(quant)
    if earnings_fig:
        st.plotly_chart(earnings_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Risk Flags ────────────────────────────────────────────────────────────
    _render_risk_section(sections, risk_flags)

    # ── Verification ─────────────────────────────────────────────────────────
    _render_verification_section(sections, verification)

    # ── Source Chunks (expandable) ────────────────────────────────────────────
    _render_source_chunks(state.get("rag_chunks", []))


def _render_section(sections: dict, key: str, fallback_title: str) -> None:
    sec = sections.get(key)
    if not sec:
        return
    title = sec.get("title") or fallback_title
    confidence = sec.get("confidence", "MEDIUM")
    content = sec.get("content", "")
    citations = sec.get("citations", [])

    # Preserve newlines as paragraph breaks
    body_html = "".join(
        f'<p style="margin:0 0 10px 0;">{para.strip()}</p>'
        for para in content.split("\n\n")
        if para.strip()
    ) or content

    st.markdown(
        render_report_section(title, confidence, body_html, citations),
        unsafe_allow_html=True,
    )


def _render_key_metrics(fd: dict) -> None:
    """Render a tight row of key financial metrics."""
    cols = st.columns(4)
    metrics = [
        ("Price", _fmt(fd.get("price"), ".2f", "$"), fd.get("sector", "")),
        ("Market Cap", _billions(fd.get("market_cap")), ""),
        ("P/E Ratio", _fmt(fd.get("pe_ratio"), ".1f", ""), ""),
        ("EPS", _fmt(fd.get("eps"), ".2f", "$"), ""),
    ]
    metric_colors = [COLORS["accent"], COLORS["green"], COLORS["amber"], COLORS["text_secondary"]]

    for col, (label, value, sub), color in zip(cols, metrics, metric_colors):
        with col:
            st.markdown(
                f"""
<div style="
    background:{COLORS['bg_card']};border:1px solid {COLORS['border']};
    border-top:2px solid {color};border-radius:8px;
    padding:12px 14px;margin-bottom:14px;text-align:center;
">
    <div style="font-size:11px;letter-spacing:0.07em;text-transform:uppercase;
                color:{COLORS['text_muted']};font-family:Inter,sans-serif;">{label}</div>
    <div style="font-size:22px;font-weight:700;color:{COLORS['text_primary']};
                font-family:Inter,sans-serif;margin:4px 0;">{value}</div>
    <div style="font-size:11px;color:{COLORS['text_muted']};font-family:Inter,sans-serif;">{sub}</div>
</div>""",
                unsafe_allow_html=True,
            )


def _render_risk_section(sections: dict, risk_flags: list[dict]) -> None:
    sec = sections.get("risk_flags")
    if sec:
        title = sec.get("title", "Risk Flags")
        confidence = sec.get("confidence", "MEDIUM")
        content = sec.get("content", "")
        body_html = f'<p style="margin:0 0 12px 0;">{content}</p>' if content else ""

        if risk_flags:
            for flag in risk_flags:
                body_html += render_risk_flag(
                    flag_type=flag.get("type", "unknown"),
                    severity=flag.get("severity", "MEDIUM"),
                    description=flag.get("description", ""),
                    citation=flag.get("citation", ""),
                )

        st.markdown(
            render_report_section(title, confidence, body_html, sec.get("citations", [])),
            unsafe_allow_html=True,
        )
    elif risk_flags:
        # Render flags even if section synthesis is missing
        flags_html = "".join(
            render_risk_flag(
                flag.get("type", "unknown"),
                flag.get("severity", "MEDIUM"),
                flag.get("description", ""),
                flag.get("citation", ""),
            )
            for flag in risk_flags
        )
        st.markdown(
            render_report_section("Risk Flags", "MEDIUM", flags_html),
            unsafe_allow_html=True,
        )


def _render_verification_section(sections: dict, verification: dict) -> None:
    sec = sections.get("verification_verdict")
    divergences = verification.get("divergences", [])
    confidence_scores = verification.get("confidence_scores", {})

    intro_html = ""
    if sec:
        intro_html = f'<p style="margin:0 0 14px 0;">{sec.get("content", "")}</p>'

    divs_html = ""
    for div in divergences:
        divs_html += render_divergence(
            claim=div.get("management_claim", ""),
            actual_data=div.get("data_evidence", ""),
            severity=div.get("severity", "MEDIUM"),
        )

    if confidence_scores:
        bars_html = "".join(
            render_confidence_bar(k.replace("_", " ").title(), v)
            for k, v in confidence_scores.items()
        )
        divs_html += f'<div style="margin-top:14px;">{bars_html}</div>'

    body_html = intro_html + divs_html

    title = sec.get("title", "Verification Verdict") if sec else "Verification"
    conf = sec.get("confidence", "MEDIUM") if sec else "MEDIUM"

    if body_html.strip():
        st.markdown(
            render_report_section(title, conf, body_html, (sec or {}).get("citations", [])),
            unsafe_allow_html=True,
        )


def _render_source_chunks(rag_chunks: list[dict]) -> None:
    if not rag_chunks:
        return
    with st.expander(f"Show {len(rag_chunks)} source chunks", expanded=False):
        for i, chunk in enumerate(rag_chunks[:20], 1):
            meta = chunk.get("metadata", {})
            section = meta.get("section_name", "?")
            filing = meta.get("filing_type", "?")
            date = meta.get("filing_date", "?")
            text = chunk.get("text", "")[:400]

            st.markdown(
                f"""
<div style="
    background:{COLORS['bg_elevated']};
    border:1px solid {COLORS['border']};
    border-radius:6px;
    padding:10px 12px;
    margin-bottom:8px;
">
    <div style="font-size:11px;font-weight:600;color:{COLORS['accent']};
                margin-bottom:4px;font-family:Inter,sans-serif;">
        [{i}] {section} · {filing} · {date}
    </div>
    <div style="font-size:12px;color:{COLORS['text_muted']};
                line-height:1.5;font-family:Inter,sans-serif;">
        {text}{'…' if len(chunk.get('text','')) > 400 else ''}
    </div>
</div>""",
                unsafe_allow_html=True,
            )
