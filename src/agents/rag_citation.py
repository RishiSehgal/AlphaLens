"""RAG Citation Agent — Agent 2 of the AlphaLens pipeline.

Replaces the Research Associate role on a traditional equity research team.
Downloads the latest 10-K filing from SEC EDGAR, parses it into named sections,
chunks and embeds the text, persists embeddings in ChromaDB, then retrieves the
most relevant chunks across three query dimensions for downstream agents.

Idempotent: if chunks for the ticker already exist in ChromaDB, the
embed-and-store step is skipped entirely.
"""

import logging
import time
from typing import Any

from src.state import AlphaLensState
from src.config import TOP_K
from src.data.edgar_client import EdgarClient
from src.data.filing_parser import parse_filing
from src.rag.chunker import chunk_sections
from src.rag.embeddings import embed_chunks
from src.rag.retriever import FilingRetriever

logger = logging.getLogger(__name__)

# Query dimensions used for retrieval — label: (query_text, section_filter)
_RETRIEVAL_QUERIES: list[tuple[str, str, str | None]] = [
    (
        "financial_performance",
        "financial performance and revenue growth",
        None,
    ),
    (
        "risk_factors",
        "risk factors and material uncertainties",
        "Risk Factors",
    ),
    (
        "management_strategy",
        "management strategy and business outlook",
        "Management's Discussion and Analysis",
    ),
]


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Deduplicate retrieved chunks by chunk_index, keeping the highest-ranked copy.

    When the same physical chunk appears in results from multiple queries,
    only the copy with the lowest rank number (i.e., highest relevance for its
    query) is retained. The ``query_label`` of the winning copy is preserved.

    Args:
        chunks: Flat list of result dicts, each with a ``metadata`` dict
            containing ``chunk_index`` and a top-level ``rank`` and
            ``query_label`` key.

    Returns:
        Deduplicated list sorted by chunk_index ascending.
    """
    seen: dict[int, dict] = {}  # chunk_index → best result dict

    for chunk in chunks:
        idx: int = chunk.get("metadata", {}).get("chunk_index", -1)
        if idx == -1:
            # No chunk_index in metadata — keep it unconditionally
            seen[id(chunk)] = chunk
            continue

        if idx not in seen or chunk["rank"] < seen[idx]["rank"]:
            seen[idx] = chunk

    return sorted(seen.values(), key=lambda c: c.get("metadata", {}).get("chunk_index", 0))


def rag_citation_agent(state: AlphaLensState) -> dict:
    """Fetch, embed, store, and retrieve SEC filing chunks for a ticker.

    Reads ``state["ticker"]`` and (optionally) ``state["financial_data"]["filing_urls"]``
    to determine which filing to process. The latest 10-K is always fetched
    directly from EDGAR via ``EdgarClient``; the ``filing_urls`` field is used
    only to obtain the filing date for metadata when the filing is already
    cached in ChromaDB.

    Steps:
        1. Download latest 10-K HTML from EDGAR.
        2. Parse into named sections (MD&A, Risk Factors, etc.).
        3. Chunk sections into 500-token overlapping pieces.
        4. Check ChromaDB — skip embed+store if data already exists (idempotent).
        5. Embed chunks with Gemini and persist in ChromaDB.
        6. Retrieve top-5 chunks for each of three query dimensions.
        7. Deduplicate results across queries by chunk_index.

    Args:
        state: LangGraph pipeline state. Must contain ``ticker`` (str).
            ``financial_data`` is read if available but not required.

    Returns:
        Partial state dict with keys:
            ``rag_chunks``: List of retrieved chunk dicts with ``query_label``.
            ``metadata``: Merged dict with ``agent_latencies.rag_citation``,
                ``rag_filing_date``, and ``rag_available``.
            ``error_log``: List of error strings (appended, not replaced).
    """
    t_start = time.monotonic()
    ticker: str = state.get("ticker", "").upper().strip()
    error_log: list[str] = []

    logger.info("[rag_citation] Starting for ticker=%s", ticker)

    if not ticker:
        logger.error("[rag_citation] No ticker in state — aborting")
        return _failure_result(
            t_start=t_start,
            error_log=["rag_citation: ticker missing from state"],
        )

    # ── Step 1: Fetch latest 10-K from EDGAR ─────────────────────────────────
    filing_date: str = ""
    filing_url: str = ""
    html_content: str = ""

    try:
        client = EdgarClient(max_filings_per_type=1)
        filings = client.get_filings(ticker, form_types=["10-K"])
    except Exception as exc:
        msg = f"rag_citation: EDGAR fetch exception for {ticker}: {exc}"
        logger.error("[rag_citation] %s", msg)
        return _failure_result(t_start=t_start, error_log=[msg])

    if not filings:
        msg = f"rag_citation: No 10-K filings returned from EDGAR for {ticker}"
        logger.warning("[rag_citation] %s", msg)
        return _failure_result(t_start=t_start, error_log=[msg])

    filing_result = filings[0]

    if not filing_result.fetch_success:
        msg = (
            f"rag_citation: EDGAR HTML fetch failed for {ticker}: "
            f"{filing_result.error}"
        )
        logger.error("[rag_citation] %s", msg)
        return _failure_result(t_start=t_start, error_log=[msg])

    html_content = filing_result.html_content
    filing_date = filing_result.metadata.filing_date
    filing_url = filing_result.metadata.document_url
    filing_type = filing_result.metadata.form_type

    logger.info(
        "[rag_citation] Fetched %s filed %s (%d chars)",
        filing_type, filing_date, len(html_content),
    )

    # ── Step 2: Parse into sections ──────────────────────────────────────────
    try:
        sections = parse_filing(html_content, source_file=filing_url)
    except Exception as exc:
        msg = f"rag_citation: parse_filing raised for {ticker}: {exc}"
        logger.error("[rag_citation] %s", msg)
        return _failure_result(t_start=t_start, error_log=[msg])

    if not sections:
        msg = f"rag_citation: No sections extracted from {ticker} {filing_type} {filing_date}"
        logger.warning("[rag_citation] %s", msg)
        error_log.append(msg)
        # Still attempt retrieval from a previously cached collection

    # ── Step 3: Chunk sections ────────────────────────────────────────────────
    chunks: list[dict] = []
    if sections:
        try:
            chunks = chunk_sections(
                sections,
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                source_file=filing_url,
            )
        except Exception as exc:
            msg = f"rag_citation: chunk_sections raised for {ticker}: {exc}"
            logger.error("[rag_citation] %s", msg)
            error_log.append(msg)
            chunks = []

    # ── Step 4 & 5: Embed and store (idempotent) ─────────────────────────────
    retriever = FilingRetriever()

    try:
        existing_count = retriever.collection_count(ticker)
    except Exception as exc:
        logger.warning("[rag_citation] collection_count raised: %s", exc)
        existing_count = 0

    if existing_count > 0:
        logger.info(
            "[rag_citation] ChromaDB already has %d chunks for %s — skipping embed+store",
            existing_count, ticker,
        )
    elif chunks:
        logger.info(
            "[rag_citation] Embedding %d chunks for %s…", len(chunks), ticker
        )
        try:
            embedded = embed_chunks(chunks)
        except Exception as exc:
            msg = f"rag_citation: embed_chunks raised for {ticker}: {exc}"
            logger.error("[rag_citation] %s", msg)
            error_log.append(msg)
            embedded = []

        if embedded:
            try:
                retriever.add_filing_chunks(ticker, embedded)
                logger.info(
                    "[rag_citation] Stored %d embedded chunks for %s",
                    len(embedded), ticker,
                )
            except Exception as exc:
                msg = f"rag_citation: add_filing_chunks raised for {ticker}: {exc}"
                logger.error("[rag_citation] %s", msg)
                error_log.append(msg)
    else:
        logger.warning(
            "[rag_citation] No chunks produced and no cached data for %s — "
            "retrieval will return empty results",
            ticker,
        )

    # ── Steps 6 & 7: Retrieve and deduplicate ────────────────────────────────
    all_results: list[dict] = []

    for query_label, query_text, section_filter in _RETRIEVAL_QUERIES:
        try:
            results = retriever.retrieve_relevant(
                ticker=ticker,
                query=query_text,
                top_k=TOP_K,
                section_filter=section_filter,
            )
        except Exception as exc:
            msg = (
                f"rag_citation: retrieve_relevant raised for {ticker} "
                f"query='{query_label}': {exc}"
            )
            logger.error("[rag_citation] %s", msg)
            error_log.append(msg)
            results = []

        for chunk in results:
            chunk["query_label"] = query_label

        logger.info(
            "[rag_citation] Query '%s' → %d result(s) (section_filter=%s)",
            query_label, len(results), section_filter or "none",
        )
        all_results.extend(results)

    rag_chunks = _deduplicate_chunks(all_results)

    logger.info(
        "[rag_citation] Final deduplicated chunk count: %d for %s",
        len(rag_chunks), ticker,
    )

    t_elapsed = time.monotonic() - t_start
    rag_available = len(rag_chunks) > 0

    return {
        "rag_chunks": rag_chunks,
        "metadata": {
            "agent_latencies": {"rag_citation": round(t_elapsed, 3)},
            "rag_filing_date": filing_date,
            "rag_available": rag_available,
        },
        "error_log": error_log,
    }


def _failure_result(
    t_start: float,
    error_log: list[str],
) -> dict:
    """Return a well-formed failure partial-state so the pipeline never crashes.

    Args:
        t_start: Monotonic timestamp from when the agent started.
        error_log: List of error messages to surface in the pipeline state.

    Returns:
        Partial state dict with empty ``rag_chunks`` and ``rag_available=False``.
    """
    return {
        "rag_chunks": [],
        "metadata": {
            "agent_latencies": {"rag_citation": round(time.monotonic() - t_start, 3)},
            "rag_filing_date": "",
            "rag_available": False,
        },
        "error_log": error_log,
    }
