"""Golden test cases for the AlphaLens evaluation suite.

Values sourced from public filings and consensus data (FY2024 / trailing twelve months).
Tolerances are intentionally generous to account for data-source variation between
Alpha Vantage, yfinance, and EDGAR extraction.
"""

from typing import TypedDict


class GoldenCase(TypedDict):
    ticker: str
    company_name: str
    # Revenue / income (USD)
    revenue_min: float
    revenue_max: float
    net_income_min: float
    net_income_max: float
    # Margins (0–1 scale)
    gross_margin_min: float
    gross_margin_max: float
    # EPS
    eps_min: float
    eps_max: float
    # Debt-to-equity (ratio)
    dte_min: float
    dte_max: float
    # Required retrieval keywords — at least one must appear in retrieved chunks
    required_rag_keywords: list[str]
    # Risks that the risk_scanner MUST flag (at least one)
    expected_risk_types: list[str]
    # Sections that must appear in the final report
    required_report_sections: list[str]


GOLDEN_CASES: list[GoldenCase] = [
    GoldenCase(
        ticker="AAPL",
        company_name="Apple Inc",
        revenue_min=370_000_000_000,      # ~$380B FY2024
        revenue_max=420_000_000_000,
        net_income_min=90_000_000_000,     # ~$94B FY2024
        net_income_max=110_000_000_000,
        gross_margin_min=0.43,             # ~46%
        gross_margin_max=0.52,
        eps_min=5.5,
        eps_max=8.5,
        dte_min=0.0,
        dte_max=4.0,
        required_rag_keywords=[
            "services",
            "mac",
            "iphone",
            "risk factors",
            "liquidity",
        ],
        expected_risk_types=[
            "customer_concentration",
            "litigation",
            "regulatory",
            "competition",
        ],
        required_report_sections=[
            "executive_summary",
            "financial_health",
            "valuation",
            "risk_flags",
        ],
    ),
    GoldenCase(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        revenue_min=55_000_000_000,        # FY2024 ~$61B, FY2025 ramping
        revenue_max=140_000_000_000,
        net_income_min=15_000_000_000,
        net_income_max=75_000_000_000,
        gross_margin_min=0.60,             # data center mix → ~73%
        gross_margin_max=0.80,
        eps_min=1.5,
        eps_max=30.0,                      # wide range: post-split adjusted
        dte_min=0.0,
        dte_max=1.0,
        required_rag_keywords=[
            "data center",
            "gpu",
            "cuda",
            "artificial intelligence",
            "export control",
        ],
        expected_risk_types=[
            "customer_concentration",
            "regulatory",
            "competition",
            "litigation",
        ],
        required_report_sections=[
            "executive_summary",
            "financial_health",
            "valuation",
            "risk_flags",
        ],
    ),
    GoldenCase(
        ticker="MSFT",
        company_name="Microsoft Corporation",
        revenue_min=200_000_000_000,       # FY2024 ~$245B
        revenue_max=280_000_000_000,
        net_income_min=70_000_000_000,
        net_income_max=110_000_000_000,
        gross_margin_min=0.68,             # ~70%
        gross_margin_max=0.76,
        eps_min=9.0,
        eps_max=15.0,
        dte_min=0.0,
        dte_max=1.0,
        required_rag_keywords=[
            "azure",
            "cloud",
            "artificial intelligence",
            "gaming",
            "linkedin",
        ],
        expected_risk_types=[
            "regulatory",
            "competition",
            "litigation",
            "customer_concentration",
        ],
        required_report_sections=[
            "executive_summary",
            "financial_health",
            "valuation",
            "risk_flags",
        ],
    ),
]

# ── Retrieval evaluation queries (used to build the golden query→section map) ──
RETRIEVAL_QUERIES: dict[str, list[str]] = {
    "risk_factors": [
        "What are the main risk factors?",
        "What could cause revenue to decline?",
        "What regulatory risks does the company face?",
    ],
    "financials": [
        "What is the revenue growth trend?",
        "What is the gross margin?",
        "How does cash flow compare to net income?",
    ],
    "mda": [
        "What does management say about the business outlook?",
        "What segments drove revenue growth?",
        "What are the key operational challenges?",
    ],
}

# ── Section-name → query mapping for recall evaluation ─────────────────────────
SECTION_QUERY_MAP: dict[str, str] = {
    "Risk Factors": "What are the main risk factors?",
    "MD&A": "What does management say about the business outlook?",
    "Financial Statements": "What is the revenue growth trend?",
}
