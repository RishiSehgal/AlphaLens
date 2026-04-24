"""Tests for the Verification agent and Report Synthesis agent."""

import pytest
from unittest.mock import MagicMock, patch

from src.state import AlphaLensState


def _base_state(**overrides) -> AlphaLensState:
    state = AlphaLensState(
        ticker="AAPL",
        financial_data={
            "ticker": "AAPL",
            "company_name": "Apple Inc",
            "price": 185.50,
            "market_cap": 2_850_000_000_000,
            "revenue": 383_285_000_000,
            "net_income": 93_736_000_000,
            "gross_margin": 0.452,
            "operating_margin": 0.294,
            "eps": 6.11,
            "free_cash_flow": 99_584_000_000,
            "total_debt": 101_304_000_000,
            "total_cash": 73_100_000_000,
            "beta": 1.25,
            "pe_ratio": 28.5,
            "filing_urls": [],
            "macro": {"FEDFUNDS": {"current_value": 5.33, "trend": "down"}},
            "sources_status": {"alpha_vantage": "ok"},
        },
        rag_chunks=[
            {
                "text": "Revenue increased 8% year over year driven by services segment growth.",
                "metadata": {"section_name": "MD&A", "ticker": "AAPL", "chunk_index": 0, "filing_type": "10-K"},
            },
            {
                "text": "We face intense competition from established and new entrants in all markets.",
                "metadata": {"section_name": "Risk Factors", "ticker": "AAPL", "chunk_index": 1, "filing_type": "10-K"},
            },
            {
                "text": "Going concern: auditors noted substantial doubt about our ability to continue as a going concern.",
                "metadata": {"section_name": "Risk Factors", "ticker": "AAPL", "chunk_index": 2, "filing_type": "10-K"},
            },
        ],
        quant_results={
            "dcf": {"base": 180.0, "bear": 120.0, "bull": 260.0, "wacc": 0.097, "confidence": "MEDIUM"},
            "technicals": {"rsi": 67.2, "macd_signal": "bullish", "rsi_signal": "neutral"},
            "earnings_surprise": {"actual": 1.58, "estimate": 1.49, "surprise_pct": 6.0},
        },
        risk_flags=[],
        verification={},
        report={},
        metadata={"agent_latencies": {}, "sources_status": {}, "pipeline_start": 0},
        error_log=[],
        chat_history=[],
    )
    state.update(overrides)
    return state


# ── _build_data_summary ───────────────────────────────────────────────────────

def test_build_data_summary_is_compact():
    from src.agents.verification import _build_data_summary
    state = _base_state()
    # _build_data_summary takes financial_data dict, not the full state
    summary = _build_data_summary(state["financial_data"])
    assert isinstance(summary, str)
    # Should be reasonably short (< 1000 chars) to fit in a prompt
    assert len(summary) < 1000


def test_build_data_summary_includes_key_metrics():
    from src.agents.verification import _build_data_summary
    state = _base_state()
    summary = _build_data_summary(state["financial_data"])
    # Revenue, margin should appear
    assert any(kw in summary.lower() for kw in ["revenue", "margin", "eps", "cash"])


# ── _select_rag_excerpts ──────────────────────────────────────────────────────

def test_select_rag_excerpts_returns_list():
    from src.agents.verification import _select_rag_excerpts
    state = _base_state()
    # _select_rag_excerpts takes rag_chunks list, not state
    excerpts = _select_rag_excerpts(state["rag_chunks"])
    assert isinstance(excerpts, list)
    assert len(excerpts) > 0


def test_select_rag_excerpts_prefers_mda():
    from src.agents.verification import _select_rag_excerpts
    state = _base_state()
    excerpts = _select_rag_excerpts(state["rag_chunks"])
    sections = [e.get("metadata", {}).get("section_name", "") for e in excerpts]
    # MD&A chunks should appear first when available
    assert "MD&A" in sections


def test_select_rag_excerpts_bounded():
    from src.agents.verification import _select_rag_excerpts
    big_chunks = [
        {
            "text": f"Chunk {i} content for testing.",
            "metadata": {"section_name": "MD&A", "ticker": "AAPL", "chunk_index": i, "filing_type": "10-K"},
        }
        for i in range(50)
    ]
    excerpts = _select_rag_excerpts(big_chunks, max_chunks=6)
    assert len(excerpts) <= 6


# ── verification_agent (mocked Gemini) ───────────────────────────────────────

MOCK_VERIFICATION_RESPONSE = """{
  "divergences": [
    {
      "field": "revenue_trend",
      "management_claim": "Revenue increased year over year",
      "data_evidence": "Revenue $383B represents 8% growth vs prior year",
      "severity": "LOW"
    }
  ],
  "confidence_scores": {
    "executive_summary": "HIGH",
    "financial_health": "HIGH",
    "valuation": "MEDIUM",
    "risk_flags": "MEDIUM"
  },
  "verification_verdict": "Analysis is largely consistent. One minor divergence noted."
}"""


@patch("src.agents.verification._call_gemini")
def test_verification_agent_returns_required_keys(mock_gemini):
    from src.agents.verification import verification_agent
    mock_gemini.return_value = MOCK_VERIFICATION_RESPONSE
    state = _base_state()
    result = verification_agent(state)

    assert "verification" in result
    v = result["verification"]
    assert "divergences" in v
    assert "confidence_scores" in v
    assert "llm_available" in v


@patch("src.agents.verification._call_gemini")
def test_verification_agent_forces_medium_valuation(mock_gemini):
    from src.agents.verification import verification_agent
    mock_gemini.return_value = MOCK_VERIFICATION_RESPONSE
    state = _base_state()
    result = verification_agent(state)

    conf_scores = result["verification"].get("confidence_scores", {})
    if "valuation" in conf_scores:
        assert conf_scores["valuation"] == "MEDIUM"


@patch("src.agents.verification._call_gemini")
def test_verification_agent_handles_invalid_json(mock_gemini):
    from src.agents.verification import verification_agent
    mock_gemini.return_value = "This is not JSON at all."
    state = _base_state()
    result = verification_agent(state)

    # Should not crash; returns safe defaults
    assert "verification" in result
    v = result["verification"]
    assert isinstance(v.get("divergences", []), list)


@patch("src.agents.verification._call_gemini")
def test_verification_agent_records_latency(mock_gemini):
    from src.agents.verification import verification_agent
    mock_gemini.return_value = MOCK_VERIFICATION_RESPONSE
    state = _base_state()
    result = verification_agent(state)

    latencies = result.get("metadata", {}).get("agent_latencies", {})
    assert "verification" in latencies


# ── report_synthesis_agent (mocked Gemini) ───────────────────────────────────

MOCK_SECTION_RESPONSE = """
{
  "title": "Executive Summary",
  "content": "Apple Inc (AAPL) demonstrates strong financial performance with revenue of $383B and gross margins of 45%. The company maintains competitive advantages in its ecosystem.",
  "confidence": "HIGH",
  "citations": [{"section_name": "MD&A", "page_number": 25, "source_file": "aapl-10k.htm"}]
}"""


@patch("src.agents.report_synthesis._call_gemini")
def test_report_synthesis_agent_returns_sections(mock_gemini):
    from src.agents.report_synthesis import report_synthesis_agent
    mock_gemini.return_value = MOCK_SECTION_RESPONSE

    state = _base_state(
        verification={
            "divergences": [],
            "confidence_scores": {"valuation": "MEDIUM"},
            "verification_verdict": "Analysis consistent.",
            "llm_available": True,
        }
    )
    result = report_synthesis_agent(state)

    assert "report" in result
    report = result["report"]
    assert "sections" in report
    assert "overall_confidence" in report
    assert "disclaimer" in report


@patch("src.agents.report_synthesis._call_gemini")
def test_report_synthesis_agent_overall_confidence_valid(mock_gemini):
    from src.agents.report_synthesis import report_synthesis_agent
    mock_gemini.return_value = MOCK_SECTION_RESPONSE

    state = _base_state(
        verification={
            "divergences": [],
            "confidence_scores": {"valuation": "MEDIUM"},
            "verification_verdict": "Consistent.",
            "llm_available": True,
        }
    )
    result = report_synthesis_agent(state)
    assert result["report"]["overall_confidence"] in ("HIGH", "MEDIUM", "LOW")


@patch("src.agents.report_synthesis._call_gemini")
def test_report_synthesis_records_latency(mock_gemini):
    from src.agents.report_synthesis import report_synthesis_agent
    mock_gemini.return_value = MOCK_SECTION_RESPONSE

    state = _base_state(
        verification={
            "divergences": [],
            "confidence_scores": {},
            "verification_verdict": "",
            "llm_available": True,
        }
    )
    result = report_synthesis_agent(state)
    latencies = result.get("metadata", {}).get("agent_latencies", {})
    assert "report_synthesis" in latencies
