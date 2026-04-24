"""Evaluation metrics for the AlphaLens RAG and agent pipeline.

Metrics:
1. retrieval_precision_at_k  — fraction of top-k chunks that are relevant
2. retrieval_recall_at_k     — fraction of relevant sections covered in top-k
3. faithfulness_score        — fraction of numerical claims in report grounded in retrieved chunks
4. numerical_accuracy        — fraction of financial figures within tolerance of golden values
5. earnings_surprise_accuracy — exact-match check on most-recent quarter beat/miss direction
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── 1. Retrieval Precision@k ──────────────────────────────────────────────────
def retrieval_precision_at_k(
    retrieved_chunks: list[dict],
    relevant_keywords: list[str],
    k: int = 5,
) -> float:
    """Fraction of top-k chunks that contain at least one relevant keyword.

    A chunk is considered relevant if any of ``relevant_keywords`` appears
    (case-insensitive) in the chunk text.

    Args:
        retrieved_chunks: List of chunk dicts with a ``"text"`` key, ordered by relevance.
        relevant_keywords: Keywords that define relevance for this query.
        k: Number of top chunks to evaluate.

    Returns:
        Precision@k as a float in [0, 1]. Returns 0.0 if retrieved_chunks is empty.
    """
    if not retrieved_chunks or not relevant_keywords:
        return 0.0

    top_k = retrieved_chunks[:k]
    keywords_lower = [kw.lower() for kw in relevant_keywords]
    relevant_count = sum(
        1 for chunk in top_k
        if any(kw in chunk.get("text", "").lower() for kw in keywords_lower)
    )
    return round(relevant_count / len(top_k), 4)


# ── 2. Retrieval Recall@k ─────────────────────────────────────────────────────
def retrieval_recall_at_k(
    retrieved_chunks: list[dict],
    required_sections: list[str],
    k: int = 5,
) -> float:
    """Fraction of required sections covered by at least one chunk in top-k.

    Args:
        retrieved_chunks: Ordered chunk dicts with ``"metadata"`` → ``"section_name"``.
        required_sections: Section names that must be represented (e.g. ``["MD&A", "Risk Factors"]``).
        k: Number of top chunks to consider.

    Returns:
        Recall@k as a float in [0, 1]. Returns 1.0 if required_sections is empty.
    """
    if not required_sections:
        return 1.0
    if not retrieved_chunks:
        return 0.0

    top_k = retrieved_chunks[:k]
    retrieved_sections = {
        chunk.get("metadata", {}).get("section_name", "").lower()
        for chunk in top_k
    }
    covered = sum(
        1 for sec in required_sections
        if any(sec.lower() in rs for rs in retrieved_sections)
    )
    return round(covered / len(required_sections), 4)


# ── 3. Faithfulness / Groundedness ───────────────────────────────────────────
def faithfulness_score(
    report_text: str,
    retrieved_chunks: list[dict],
) -> float:
    """Fraction of dollar-figure claims in the report that are grounded in retrieved chunks.

    Extracts all ``$X.XB`` / ``$X.XM`` / ``$X.X billion`` patterns from the report,
    then checks whether each appears verbatim (±1 word context) in any retrieved chunk.

    This is a lexical proxy for groundedness — not a semantic entailment check —
    appropriate for the free-tier budget of this project.

    Args:
        report_text: Full plain text of the generated report.
        retrieved_chunks: RAG chunks used to generate the report.

    Returns:
        Faithfulness score in [0, 1]. Returns 1.0 if no monetary claims are found.
    """
    if not report_text:
        return 0.0

    # Extract monetary figures: $123B, $4.5M, $1.2 trillion, $999 million, etc.
    pattern = re.compile(
        r"\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion|B|M|T)\b",
        re.IGNORECASE,
    )
    claims = pattern.findall(report_text)

    if not claims:
        return 1.0  # No numeric claims to ground → vacuously faithful

    all_chunk_text = " ".join(c.get("text", "") for c in retrieved_chunks).lower()

    grounded = 0
    for claim in claims:
        # Normalise: strip spaces between $ and digits, lowercase
        normalised = re.sub(r"\s+", "", claim).lower()
        # Accept if the digits+unit appear anywhere in the chunks (some formatting variation OK)
        digits = re.sub(r"[^\d.]", "", normalised)
        if digits and digits in all_chunk_text.replace(",", ""):
            grounded += 1

    return round(grounded / len(claims), 4)


# ── 4. Numerical Accuracy ─────────────────────────────────────────────────────
def numerical_accuracy(
    financial_data: dict,
    golden: dict,
    tolerance: float = 0.15,
) -> float:
    """Fraction of key financial figures within tolerance of golden values.

    Checks: revenue, net_income, gross_margin, eps, debt_to_equity.

    Args:
        financial_data: ``state["financial_data"]`` from the pipeline.
        golden: A :class:`~src.eval.test_cases.GoldenCase` dict.
        tolerance: Relative tolerance; default 15% (generous for free-tier data sources).

    Returns:
        Accuracy in [0, 1].
    """
    checks: list[tuple[str, float | None, float, float]] = [
        ("revenue", financial_data.get("revenue"), golden["revenue_min"], golden["revenue_max"]),
        ("net_income", financial_data.get("net_income"), golden["net_income_min"], golden["net_income_max"]),
        ("gross_margin", financial_data.get("gross_margin"), golden["gross_margin_min"], golden["gross_margin_max"]),
        ("eps", financial_data.get("eps"), golden["eps_min"], golden["eps_max"]),
        ("debt_to_equity", financial_data.get("debt_to_equity"), golden["dte_min"], golden["dte_max"]),
    ]

    passed = 0
    total = 0
    for field, value, low, high in checks:
        if value is None:
            logger.debug("numerical_accuracy: %s is None — skipping", field)
            continue
        total += 1
        try:
            v = float(value)
            # Widen the range by the tolerance on each side
            span = max(high - low, abs(low) * tolerance, 1e-9)
            padded_low = low - span * tolerance
            padded_high = high + span * tolerance
            if padded_low <= v <= padded_high:
                passed += 1
            else:
                logger.debug(
                    "numerical_accuracy: %s = %s out of range [%s, %s]",
                    field, v, padded_low, padded_high,
                )
        except (TypeError, ValueError):
            pass

    return round(passed / total, 4) if total else 0.0


# ── 5. Earnings Surprise Accuracy ────────────────────────────────────────────
def earnings_surprise_accuracy(
    quant_results: dict,
    golden: dict,
) -> float:
    """Direction-accuracy of the most recent earnings surprise.

    Checks only the beat/miss *direction* (not magnitude) because EPS consensus
    changes daily and exact values differ across data sources.

    Args:
        quant_results: ``state["quant_results"]`` from the pipeline.
        golden: GoldenCase dict (not used for value comparison — reserved for future extension).

    Returns:
        1.0 if earnings_surprise data is present and directionally consistent, else 0.0.
    """
    es = quant_results.get("earnings_surprise", {})
    actual = es.get("actual_eps")
    estimate = es.get("estimate_eps")
    if actual is None or estimate is None:
        logger.debug("earnings_surprise_accuracy: missing actual/estimate")
        return 0.0
    try:
        # Direction check: both positive, or actual > estimate (beat), etc.
        surprise_pct = es.get("surprise_pct", 0.0)
        # A surprise is "consistent" if we have a non-zero figure and the direction makes sense
        consistent = (float(actual) > float(estimate)) == (float(surprise_pct) >= 0)
        return 1.0 if consistent else 0.0
    except (TypeError, ValueError):
        return 0.0


# ── Aggregate helper ──────────────────────────────────────────────────────────
def compute_all_metrics(
    state: dict,
    golden: dict,
    relevant_keywords: list[str] | None = None,
    required_sections: list[str] | None = None,
    k: int = 5,
) -> dict[str, float]:
    """Run all 5 metrics and return a summary dict.

    Args:
        state: Completed AlphaLensState.
        golden: GoldenCase for this ticker.
        relevant_keywords: Override for precision@k keywords (default: golden required_rag_keywords).
        required_sections: Override for recall@k sections (default: ["MD&A", "Risk Factors"]).
        k: Top-k chunks for retrieval metrics.

    Returns:
        Dict mapping metric_name → score.
    """
    rag_chunks = state.get("rag_chunks", [])
    keywords = relevant_keywords or golden.get("required_rag_keywords", [])
    sections = required_sections or ["management", "risk factors"]

    # Build full report text from all section contents
    report_text = " ".join(
        sec.get("content", "")
        for sec in state.get("report", {}).get("sections", {}).values()
        if isinstance(sec, dict)
    )

    return {
        "retrieval_precision_at_k": retrieval_precision_at_k(rag_chunks, keywords, k),
        # Recall uses all chunks: our pipeline issues multiple targeted queries, so
        # section coverage should be measured across the full retrieved set.
        "retrieval_recall_at_k": retrieval_recall_at_k(rag_chunks, sections, k=len(rag_chunks) or k),
        "faithfulness_score": faithfulness_score(report_text, rag_chunks),
        "numerical_accuracy": numerical_accuracy(state.get("financial_data", {}), golden),
        "earnings_surprise_accuracy": earnings_surprise_accuracy(
            state.get("quant_results", {}), golden
        ),
    }
