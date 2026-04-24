"""Follow-up Q&A chat widget powered by Gemini, grounded in pipeline state."""

import logging
import time

import streamlit as st

from src.config import COLORS, GEMINI_MODEL, GOOGLE_API_KEY
from src.state import AlphaLensState
from src.ui.components import render_chat_message

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are AlphaLens, an AI equity research assistant. You have access to a
freshly-run analysis for a specific stock, including financial metrics, DCF valuation, risk flags,
full report sections, verification results, and excerpts from the company's SEC filings (10-K/10-Q).

Answer the user's follow-up questions using ALL available data from this analysis — metrics,
report sections, AND filing excerpts. Be concise and cite specific figures when available.
Draw on filing excerpts to answer qualitative questions (e.g. revenue drivers, risks, strategy).
Never say data is unavailable if it appears anywhere in the context.
Never provide investment advice."""


def _build_context_snippet(state: AlphaLensState) -> str:
    """Summarise the pipeline state into a rich context string for Gemini."""
    ticker = state.get("ticker", "")
    fd = state.get("financial_data", {})
    quant = state.get("quant_results", {})
    risk_flags = state.get("risk_flags", [])
    verification = state.get("verification", {})
    report = state.get("report", {})
    sections = report.get("sections", {})
    rag_chunks = state.get("rag_chunks", [])

    lines = [f"STOCK: {ticker}"]

    # ── Financial metrics ─────────────────────────────────────────────────────
    if fd:
        lines.append(
            f"Price: ${fd.get('price', 'N/A')} | P/E: {fd.get('pe_ratio', 'N/A')} | "
            f"Fwd P/E: {fd.get('forward_pe', 'N/A')} | Market Cap: {fd.get('market_cap', 'N/A')} | "
            f"Beta: {fd.get('beta', 'N/A')}"
        )
        lines.append(
            f"Revenue: {fd.get('revenue', 'N/A')} | Net Income: {fd.get('net_income', 'N/A')} | "
            f"FCF: {fd.get('free_cash_flow', 'N/A')} | Gross Margin: {fd.get('gross_margin', 'N/A')} | "
            f"Op Margin: {fd.get('operating_margin', 'N/A')}"
        )
        lines.append(
            f"EPS: {fd.get('eps', 'N/A')} | Total Debt: {fd.get('total_debt', 'N/A')} | "
            f"Cash: {fd.get('total_cash', 'N/A')} | 52W High: {fd.get('week_52_high', 'N/A')} | "
            f"52W Low: {fd.get('week_52_low', 'N/A')}"
        )

    # ── Quant ─────────────────────────────────────────────────────────────────
    dcf = quant.get("dcf", {})
    if dcf:
        wacc = dcf.get('wacc')
        wacc_str = f" at {wacc:.1%} WACC" if isinstance(wacc, float) else ""
        lines.append(
            f"DCF: Bear ${dcf.get('bear', 'N/A')} / Base ${dcf.get('base', 'N/A')} / "
            f"Bull ${dcf.get('bull', 'N/A')}{wacc_str}"
        )
    tech = quant.get("technicals", {})
    if tech:
        lines.append(
            f"RSI(14): {tech.get('rsi', 'N/A')} ({tech.get('rsi_signal', '')}) | "
            f"MACD: {tech.get('macd_signal', 'N/A')}"
        )
    es = quant.get("earnings_surprise", {})
    if es and isinstance(es.get('surprise_pct'), (int, float)):
        lines.append(
            f"Earnings: Actual ${es.get('actual', 'N/A')} vs Est ${es.get('estimate', 'N/A')} "
            f"({es.get('surprise_pct', 0):+.1f}%)"
        )

    # ── Risk flags ────────────────────────────────────────────────────────────
    if risk_flags:
        lines.append("\nRISK FLAGS:")
        for f in risk_flags[:6]:
            lines.append(
                f"  [{f.get('severity', '?')}] {f.get('type', '?')}: {f.get('description', '')[:150]}"
            )

    # ── Full report sections ──────────────────────────────────────────────────
    section_order = ["executive_summary", "financial_health", "risk_flags", "valuation", "verification_verdict"]
    for key in section_order:
        sec = sections.get(key, {})
        if sec and sec.get("content"):
            lines.append(f"\n{key.upper().replace('_', ' ')}:\n{sec['content'][:800]}")

    # ── Verification divergences ──────────────────────────────────────────────
    divs = verification.get("divergences", [])
    if divs:
        lines.append("\nVERIFICATION DIVERGENCES:")
        for d in divs[:4]:
            lines.append(
                f"  [{d.get('severity', '?')}] {d.get('field', '?')}: "
                f"Management says '{d.get('management_claim', '')}' | "
                f"Data shows: {d.get('data_evidence', '')}"
            )
    verdict = verification.get("verification_verdict", "")
    if verdict:
        lines.append(f"Verdict: {verdict}")

    # ── RAG chunks (filing excerpts) ──────────────────────────────────────────
    if rag_chunks:
        lines.append("\nFILING EXCERPTS (from 10-K/10-Q):")
        seen_sections: set[str] = set()
        count = 0
        for chunk in rag_chunks:
            if count >= 8:
                break
            text = chunk.get("text", "").strip()
            section = chunk.get("metadata", {}).get("section_name", "")
            if not text:
                continue
            label = f"[{section}]" if section else ""
            lines.append(f"  {label} {text[:400]}")
            seen_sections.add(section)
            count += 1

    return "\n".join(lines)


def _call_gemini(messages: list[dict]) -> str:
    """Send conversation history to Gemini and return text response.

    Args:
        messages: List of {"role": "user"|"model", "parts": [{"text": str}]} dicts.

    Returns:
        Response text string.
    """
    try:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=GOOGLE_API_KEY)
        contents = []
        for m in messages:
            role = m["role"]
            text = m["parts"][0]["text"]
            contents.append(gtypes.Content(role=role, parts=[gtypes.Part(text=text)]))

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=gtypes.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        return response.text or "I couldn't generate a response. Please try again."

    except Exception as exc:
        # Retry once on 429
        if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
            time.sleep(65)
            return _call_gemini(messages)
        logger.error("Gemini chat failed: %s", exc)
        return f"Error communicating with Gemini: {exc}"


def render_chat(state: AlphaLensState) -> None:
    """Render the follow-up Q&A chat widget.

    Uses st.session_state["chat_history"] to persist conversation turns.
    Each turn is stored as a dict with "role" and "content" keys for display,
    and the Gemini API format is reconstructed on each call.

    Args:
        state: Completed AlphaLensState providing grounding context.
    """
    ticker = state.get("ticker", "stock")

    # Section heading
    st.markdown(
        f'<div style="font-size:16px;font-weight:600;color:{COLORS["text_primary"]};'
        f'font-family:Inter,sans-serif;margin:28px 0 14px 0;">'
        f'Ask a follow-up about {ticker}</div>',
        unsafe_allow_html=True,
    )

    # Initialise history if needed
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render existing messages
    for msg in st.session_state.chat_history:
        st.markdown(render_chat_message(msg["role"], msg["content"]), unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input(
        placeholder=f"e.g. What's the biggest risk for {ticker}?",
        key="alphalens_chat_input",
    )

    if user_input:
        # Append user turn
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.markdown(render_chat_message("user", user_input), unsafe_allow_html=True)

        # Build Gemini conversation (inject context as first user turn)
        context = _build_context_snippet(state)
        gemini_messages = [
            {
                "role": "user",
                "parts": [{"text": f"Here is the analysis context:\n\n{context}"}],
            },
            {
                "role": "model",
                "parts": [{"text": "Understood. I'll answer your follow-up questions based on this analysis."}],
            },
        ]

        for turn in st.session_state.chat_history:
            gemini_role = "user" if turn["role"] == "user" else "model"
            gemini_messages.append(
                {"role": gemini_role, "parts": [{"text": turn["content"]}]}
            )

        with st.spinner(""):
            answer = _call_gemini(gemini_messages)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.markdown(render_chat_message("assistant", answer), unsafe_allow_html=True)
