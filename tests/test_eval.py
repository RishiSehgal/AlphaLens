"""Tests for the evaluation metrics and test_cases module."""

import pytest
from src.eval.metrics import (
    retrieval_precision_at_k,
    retrieval_recall_at_k,
    faithfulness_score,
    numerical_accuracy,
    earnings_surprise_accuracy,
    compute_all_metrics,
)
from src.eval.test_cases import GOLDEN_CASES, RETRIEVAL_QUERIES


# ── Test cases sanity checks ──────────────────────────────────────────────────

def test_golden_cases_not_empty():
    assert len(GOLDEN_CASES) >= 3


def test_golden_cases_have_required_tickers():
    tickers = {g["ticker"] for g in GOLDEN_CASES}
    assert "AAPL" in tickers
    assert "NVDA" in tickers
    assert "MSFT" in tickers


def test_golden_cases_revenue_ranges_sensible():
    for g in GOLDEN_CASES:
        assert g["revenue_min"] < g["revenue_max"]
        assert g["revenue_min"] > 0
        assert g["revenue_max"] > 1_000_000_000  # at least $1B


def test_golden_cases_margin_ranges_in_0_1():
    for g in GOLDEN_CASES:
        assert 0 <= g["gross_margin_min"] <= 1
        assert 0 <= g["gross_margin_max"] <= 1
        assert g["gross_margin_min"] < g["gross_margin_max"]


def test_golden_cases_have_rag_keywords():
    for g in GOLDEN_CASES:
        assert len(g["required_rag_keywords"]) >= 3


def test_retrieval_queries_not_empty():
    assert len(RETRIEVAL_QUERIES) > 0
    for category, queries in RETRIEVAL_QUERIES.items():
        assert len(queries) >= 1


# ── retrieval_precision_at_k ──────────────────────────────────────────────────

def _make_chunks(texts: list[str]) -> list[dict]:
    return [{"text": t, "metadata": {"section_name": "MD&A"}} for t in texts]


def test_precision_all_relevant():
    chunks = _make_chunks([
        "revenue growth services", "risk factors competition", "gross margin improved",
        "iphone sales increased", "mac revenue strong",
    ])
    score = retrieval_precision_at_k(chunks, ["revenue", "risk", "margin", "iphone", "mac"], k=5)
    assert score == pytest.approx(1.0)


def test_precision_none_relevant():
    chunks = _make_chunks(["The quick brown fox", "lorem ipsum dolor", "sat on a mat"])
    score = retrieval_precision_at_k(chunks, ["revenue", "risk", "iphone"], k=3)
    assert score == pytest.approx(0.0)


def test_precision_partial():
    chunks = _make_chunks(["revenue grew 8%", "unrelated text here", "risk factors noted"])
    score = retrieval_precision_at_k(chunks, ["revenue", "risk"], k=3)
    assert pytest.approx(2 / 3, abs=0.01) == score


def test_precision_empty_chunks():
    score = retrieval_precision_at_k([], ["revenue"], k=5)
    assert score == 0.0


def test_precision_empty_keywords():
    chunks = _make_chunks(["revenue grew"])
    score = retrieval_precision_at_k(chunks, [], k=5)
    assert score == 0.0


def test_precision_k_larger_than_chunks():
    chunks = _make_chunks(["revenue text"])
    score = retrieval_precision_at_k(chunks, ["revenue"], k=10)
    assert score == 1.0


# ── retrieval_recall_at_k ────────────────────────────────────────────────────

def _make_sectioned_chunks(section_names: list[str]) -> list[dict]:
    return [
        {"text": f"Content for {s}", "metadata": {"section_name": s}}
        for s in section_names
    ]


def test_recall_all_sections_covered():
    chunks = _make_sectioned_chunks(["Risk Factors", "MD&A", "Financial Statements"])
    score = retrieval_recall_at_k(chunks, ["Risk Factors", "MD&A"], k=3)
    assert score == pytest.approx(1.0)


def test_recall_no_sections_covered():
    chunks = _make_sectioned_chunks(["Notes to Financial Statements", "Exhibits"])
    score = retrieval_recall_at_k(chunks, ["Risk Factors", "MD&A"], k=2)
    assert score == pytest.approx(0.0)


def test_recall_partial_coverage():
    chunks = _make_sectioned_chunks(["Risk Factors", "Exhibits"])
    score = retrieval_recall_at_k(chunks, ["Risk Factors", "MD&A"], k=2)
    assert score == pytest.approx(0.5)


def test_recall_empty_required_sections():
    chunks = _make_sectioned_chunks(["Risk Factors"])
    score = retrieval_recall_at_k(chunks, [], k=5)
    assert score == pytest.approx(1.0)


def test_recall_empty_chunks():
    score = retrieval_recall_at_k([], ["Risk Factors"], k=5)
    assert score == pytest.approx(0.0)


# ── faithfulness_score ────────────────────────────────────────────────────────

def test_faithfulness_all_grounded():
    report = "Revenue was $383 billion for the fiscal year."
    chunks = [{"text": "Total net revenues were 383 billion dollars for fiscal 2024."}]
    score = faithfulness_score(report, chunks)
    assert score == pytest.approx(1.0)


def test_faithfulness_no_monetary_claims():
    report = "The company has strong management and good strategy."
    chunks = [{"text": "Management team has strong experience."}]
    score = faithfulness_score(report, chunks)
    assert score == pytest.approx(1.0)  # vacuously faithful


def test_faithfulness_empty_report():
    score = faithfulness_score("", [{"text": "some text"}])
    assert score == 0.0


def test_faithfulness_ungrounded_claim():
    report = "Revenue was $999 billion."
    chunks = [{"text": "Revenue was actually much lower at 200 million."}]
    score = faithfulness_score(report, chunks)
    # 999 billion not in chunks → low score
    assert score < 1.0


# ── numerical_accuracy ────────────────────────────────────────────────────────

AAPL_GOLDEN = next(g for g in GOLDEN_CASES if g["ticker"] == "AAPL")


def test_numerical_accuracy_all_within_range():
    fd = {
        "revenue": 390_000_000_000,       # within AAPL range
        "net_income": 95_000_000_000,
        "gross_margin": 0.46,
        "eps": 6.5,
        "debt_to_equity": 1.5,
    }
    score = numerical_accuracy(fd, AAPL_GOLDEN)
    assert score == pytest.approx(1.0)


def test_numerical_accuracy_one_wrong():
    fd = {
        "revenue": 10_000_000,            # way too low
        "net_income": 95_000_000_000,
        "gross_margin": 0.46,
        "eps": 6.5,
        "debt_to_equity": 1.5,
    }
    score = numerical_accuracy(fd, AAPL_GOLDEN)
    assert score < 1.0


def test_numerical_accuracy_all_none():
    fd = {"revenue": None, "net_income": None, "gross_margin": None, "eps": None, "debt_to_equity": None}
    score = numerical_accuracy(fd, AAPL_GOLDEN)
    assert score == 0.0


def test_numerical_accuracy_empty_fd():
    score = numerical_accuracy({}, AAPL_GOLDEN)
    assert score == 0.0


# ── earnings_surprise_accuracy ───────────────────────────────────────────────

def test_earnings_surprise_accuracy_beat():
    quant = {"earnings_surprise": {"actual": 1.58, "estimate": 1.49, "surprise_pct": 6.0}}
    score = earnings_surprise_accuracy(quant, AAPL_GOLDEN)
    assert score == 1.0


def test_earnings_surprise_accuracy_miss():
    quant = {"earnings_surprise": {"actual": 1.30, "estimate": 1.49, "surprise_pct": -12.8}}
    score = earnings_surprise_accuracy(quant, AAPL_GOLDEN)
    assert score == 1.0  # direction is internally consistent (miss + negative %)


def test_earnings_surprise_accuracy_missing_data():
    quant = {"earnings_surprise": {}}
    score = earnings_surprise_accuracy(quant, AAPL_GOLDEN)
    assert score == 0.0


def test_earnings_surprise_accuracy_no_key():
    quant = {}
    score = earnings_surprise_accuracy(quant, AAPL_GOLDEN)
    assert score == 0.0


# ── compute_all_metrics ───────────────────────────────────────────────────────

def test_compute_all_metrics_returns_all_keys():
    state = {
        "rag_chunks": _make_sectioned_chunks(["MD&A", "Risk Factors"]),
        "financial_data": {
            "revenue": 390_000_000_000,
            "net_income": 95_000_000_000,
            "gross_margin": 0.46,
            "eps": 6.5,
            "debt_to_equity": 1.5,
        },
        "quant_results": {
            "earnings_surprise": {"actual": 1.58, "estimate": 1.49, "surprise_pct": 6.0}
        },
        "report": {"sections": {}},
    }
    result = compute_all_metrics(state, AAPL_GOLDEN, k=2)
    expected_keys = [
        "retrieval_precision_at_k",
        "retrieval_recall_at_k",
        "faithfulness_score",
        "numerical_accuracy",
        "earnings_surprise_accuracy",
    ]
    for key in expected_keys:
        assert key in result
        assert 0.0 <= result[key] <= 1.0
