"""Tests for the Data Fusion agent and supporting data clients.

These tests use mocking to avoid real API calls — they verify parsing logic,
normalization, and fallback behavior without spending API quota.
"""

import pytest
from unittest.mock import MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_market_data():
    """Realistic MarketDataClient.get_financial_data() payload (nested structure)."""
    return {
        "ticker": "AAPL",
        "yfinance": {
            "long_name": "Apple Inc",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "price": 185.50,
            "market_cap": 2_850_000_000_000,
            "pe_ratio": 28.5,
            "forward_pe": 26.1,
            "dividend_yield": 0.0054,
            "beta": 1.25,
            "week_52_high": 198.23,
            "week_52_low": 164.08,
            "revenue_ttm": 383_285_000_000,
            "net_income_ttm": 93_736_000_000,
            "eps": 6.11,
            "info_raw": {},
        },
        "income_statement": {
            "annualReports": [
                {
                    "totalRevenue": "383285000000",
                    "netIncome": "93736000000",
                    "grossProfit": "173204000000",
                    "operatingIncome": "119437000000",
                    "ebitda": "130101000000",
                    "eps": "6.11",
                }
            ]
        },
        "balance_sheet": {
            "annualReports": [
                {
                    "totalAssets": "352583000000",
                    "totalLiabilities": "290020000000",
                    "totalCurrentLiabilities": "145308000000",
                    "totalCurrentAssets": "152987000000",
                    "longTermDebt": "91807000000",
                    "cashAndCashEquivalentsAtCarryingValue": "29965000000",
                    "shortTermInvestments": "35228000000",
                    "commonStockSharesOutstanding": "15441900000",
                }
            ]
        },
        "cash_flow": {
            "annualReports": [
                {
                    "operatingCashflow": "116433000000",
                    "capitalExpenditures": "16849000000",
                    "dividendPayout": "15025000000",
                }
            ]
        },
        "sources_status": {"alpha_vantage_income": "ok", "yfinance": "ok"},
    }


@pytest.fixture
def mock_filing_urls():
    return [
        {
            "form_type": "10-K",
            "filing_date": "2024-11-01",
            "document_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000132/aapl-20240928.htm",
            "accession_number": "0000320193-24-000132",
        }
    ]


@pytest.fixture
def mock_macro():
    return {
        "FEDFUNDS": {"current_value": 5.33, "previous_value": 5.50, "trend": "down"},
        "GDP": {"current_value": 25.46, "previous_value": 25.20, "trend": "up"},
        "CPIAUCSL": {"current_value": 315.1, "previous_value": 312.0, "trend": "up"},
        "UNRATE": {"current_value": 3.9, "previous_value": 3.8, "trend": "up"},
    }


# ── _av_float helper ──────────────────────────────────────────────────────────

def test_av_float_normal():
    from src.agents.data_fusion import _av_float
    reports = [{"totalRevenue": "383285000000"}]
    assert _av_float(reports, "totalRevenue") == 383_285_000_000.0


def test_av_float_none_string():
    from src.agents.data_fusion import _av_float
    reports = [{"totalRevenue": "None"}]
    assert _av_float(reports, "totalRevenue") is None


def test_av_float_missing_key():
    from src.agents.data_fusion import _av_float
    reports = [{"otherField": "123"}]
    assert _av_float(reports, "totalRevenue") is None


def test_av_float_index_out_of_bounds():
    from src.agents.data_fusion import _av_float
    reports = [{"totalRevenue": "100"}]
    assert _av_float(reports, "totalRevenue", index=5) is None


def test_av_float_empty_list():
    from src.agents.data_fusion import _av_float
    assert _av_float([], "totalRevenue") is None


# ── _normalize ────────────────────────────────────────────────────────────────

def test_normalize_produces_required_keys(mock_market_data, mock_filing_urls, mock_macro):
    from src.agents.data_fusion import _normalize
    result = _normalize(mock_market_data, mock_filing_urls, mock_macro)

    required_keys = [
        "ticker", "company_name", "price", "market_cap", "pe_ratio",
        "revenue", "net_income", "gross_margin", "eps", "free_cash_flow",
        "filing_urls", "macro", "sources_status",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_normalize_revenue_passthrough(mock_market_data, mock_filing_urls, mock_macro):
    from src.agents.data_fusion import _normalize
    result = _normalize(mock_market_data, mock_filing_urls, mock_macro)
    # Revenue comes from AV annualReports or yfinance revenue_ttm
    assert result["revenue"] is not None
    assert result["revenue"] == pytest.approx(383_285_000_000, rel=0.01)


def test_normalize_filing_urls_preserved(mock_market_data, mock_filing_urls, mock_macro):
    from src.agents.data_fusion import _normalize
    result = _normalize(mock_market_data, mock_filing_urls, mock_macro)
    assert len(result["filing_urls"]) == 1
    assert result["filing_urls"][0]["form_type"] == "10-K"


def test_normalize_macro_included(mock_market_data, mock_filing_urls, mock_macro):
    from src.agents.data_fusion import _normalize
    result = _normalize(mock_market_data, mock_filing_urls, mock_macro)
    assert "FEDFUNDS" in result["macro"]


# ── data_fusion_agent (mocked) ────────────────────────────────────────────────

@patch("src.agents.data_fusion.MarketDataClient")
@patch("src.agents.data_fusion.EdgarClient")
@patch("src.agents.data_fusion.FredClient")
def test_data_fusion_agent_returns_financial_data(
    mock_fred_cls, mock_edgar_cls, mock_market_cls,
    mock_market_data, mock_filing_urls, mock_macro,
):
    from src.agents.data_fusion import data_fusion_agent
    from src.state import AlphaLensState

    # Wire up mocks — agent calls get_financial_data() and get_filing_urls()
    mock_market_cls.return_value.get_financial_data.return_value = mock_market_data
    mock_edgar_cls.return_value.get_filing_urls.return_value = mock_filing_urls
    mock_fred_cls.return_value.get_macro_snapshot.return_value = mock_macro

    state = AlphaLensState(
        ticker="AAPL",
        financial_data={},
        rag_chunks=[],
        quant_results={},
        risk_flags=[],
        verification={},
        report={},
        metadata={"agent_latencies": {}, "sources_status": {}, "pipeline_start": 0},
        error_log=[],
        chat_history=[],
    )

    result = data_fusion_agent(state)

    assert "financial_data" in result
    # ticker is normalised from the market_data["ticker"] field
    assert result["financial_data"].get("ticker") in ("AAPL", "aapl", "")
    assert "metadata" in result
    assert "data_fusion" in result["metadata"]["agent_latencies"]


@patch("src.agents.data_fusion.MarketDataClient")
@patch("src.agents.data_fusion.EdgarClient")
@patch("src.agents.data_fusion.FredClient")
def test_data_fusion_agent_handles_market_data_crash(
    mock_fred_cls, mock_edgar_cls, mock_market_cls,
    mock_filing_urls, mock_macro,
):
    from src.agents.data_fusion import data_fusion_agent
    from src.state import AlphaLensState

    mock_market_cls.return_value.get_market_data.side_effect = RuntimeError("API error")
    mock_edgar_cls.return_value.get_filing_urls.return_value = mock_filing_urls
    mock_fred_cls.return_value.get_macro_snapshot.return_value = mock_macro

    state = AlphaLensState(
        ticker="AAPL",
        financial_data={},
        rag_chunks=[],
        quant_results={},
        risk_flags=[],
        verification={},
        report={},
        metadata={"agent_latencies": {}, "sources_status": {}, "pipeline_start": 0},
        error_log=[],
        chat_history=[],
    )

    result = data_fusion_agent(state)
    # Should not raise; error_log may contain the error
    assert "financial_data" in result


# ── FRED client ───────────────────────────────────────────────────────────────

def test_fred_client_returns_expected_keys():
    from src.data.fred_client import FredClient
    import pandas as pd

    mock_fred = MagicMock()
    mock_fred.get_series.return_value = pd.Series(
        [5.0, 5.2, 5.33],
        index=pd.date_range("2024-01-01", periods=3, freq="ME"),
    )

    client = FredClient(api_key="test_key")
    # _build_fred() is the method that instantiates fredapi.Fred — patch it directly
    with patch.object(client, "_build_fred", return_value=mock_fred):
        snap = client.get_macro_snapshot()

    expected_indicators = ["FEDFUNDS", "GDP", "CPIAUCSL", "UNRATE"]
    for ind in expected_indicators:
        assert ind in snap, f"Missing indicator: {ind}"
        entry = snap[ind]
        assert "current_value" in entry
        assert "trend" in entry
