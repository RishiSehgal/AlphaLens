"""Chunk parsed filing sections into token-sized pieces with citation metadata."""

import logging
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

# Approximate chars-per-token for cl100k_base; used only as a display label
_APPROX_CHARS_PER_TOKEN = 4


def chunk_sections(
    sections: list[dict],
    ticker: str,
    filing_type: str,
    filing_date: str,
    source_file: str = "",
) -> list[dict]:
    """Split parsed filing sections into overlapping token-sized chunks.

    Each output chunk carries full citation metadata so downstream agents can
    trace any claim back to a specific section and page in the source filing.

    Args:
        sections: Output of ``parse_filing()`` — list of dicts with keys
            ``section_name``, ``text_content``, ``estimated_page``.
        ticker: Ticker symbol (e.g. "AAPL"), stored verbatim in metadata.
        filing_type: Form type (e.g. "10-K" or "10-Q").
        filing_date: Filing date string (e.g. "2025-10-31").
        source_file: Optional URL or filename label for logging.

    Returns:
        List of dicts, each with keys:
            ``text`` (str): The chunk text,
            ``metadata`` (dict): Citation-ready metadata with keys
                ticker, section_name, filing_type, filing_date,
                chunk_index, estimated_page, source_file.
    """
    if not sections:
        logger.warning("chunk_sections received empty sections list for %s %s", ticker, filing_type)
        return []

    # Use tiktoken-aware splitter so chunk_size is in tokens, not chars
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    except Exception as exc:
        logger.warning("tiktoken splitter unavailable (%s) — falling back to char-based", exc)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE * _APPROX_CHARS_PER_TOKEN,
            chunk_overlap=CHUNK_OVERLAP * _APPROX_CHARS_PER_TOKEN,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    all_chunks: list[dict] = []
    global_index = 0

    for section in sections:
        section_name: str = section.get("section_name", "Unknown")
        text: str = section.get("text_content", "")
        estimated_page: int = section.get("estimated_page", 1)

        if not text.strip():
            logger.debug("Skipping empty section '%s' for %s", section_name, ticker)
            continue

        try:
            raw_chunks = splitter.split_text(text)
        except Exception as exc:
            logger.error("Splitter failed on section '%s' for %s: %s", section_name, ticker, exc)
            continue

        for local_idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 50:
                continue  # skip degenerate tiny fragments

            all_chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "ticker": ticker.upper(),
                        "section_name": section_name,
                        "filing_type": filing_type,
                        "filing_date": filing_date,
                        "chunk_index": global_index,
                        "local_chunk_index": local_idx,
                        "estimated_page": estimated_page,
                        "source_file": source_file,
                    },
                }
            )
            global_index += 1

        logger.debug(
            "Section '%s' → %d chunk(s) for %s %s",
            section_name, len(raw_chunks), ticker, filing_type,
        )

    logger.info(
        "Chunked %d section(s) → %d chunk(s) for %s %s (%s)",
        len(sections), len(all_chunks), ticker, filing_type, filing_date,
    )
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.data.edgar_client import EdgarClient
    from src.data.filing_parser import parse_filing

    edgar = EdgarClient(max_filings_per_type=1)
    filings = edgar.get_filings("AAPL", form_types=["10-K"])

    if not filings or not filings[0].fetch_success:
        print("Could not fetch AAPL 10-K")
    else:
        f = filings[0]
        sections = parse_filing(f.html_content, source_file=f.metadata.document_url)
        chunks = chunk_sections(
            sections,
            ticker="AAPL",
            filing_type=f.metadata.form_type,
            filing_date=f.metadata.filing_date,
            source_file=f.metadata.document_url,
        )

        print(f"\nTotal chunks: {len(chunks)}")
        for i, c in enumerate(chunks[:3]):
            meta = c["metadata"]
            print(f"\n--- Chunk {i} ---")
            print(f"  section:  {meta['section_name']}")
            print(f"  page:     {meta['estimated_page']}")
            print(f"  filing:   {meta['filing_type']} {meta['filing_date']}")
            print(f"  text[:120]: {c['text'][:120]}…")
