"""Embed text chunks using Google Gemini embedding models via the google-genai SDK."""

import logging
import time
from typing import Any

from google import genai
from google.genai import types

from src.config import EMBEDDING_MODEL, GOOGLE_API_KEY

logger = logging.getLogger(__name__)

# Gemini free tier: 15 RPM. Each batch call counts as 1 request.
_BATCH_SIZE = 100
_INTER_BATCH_SLEEP = 5.0  # seconds between batches to stay under 15 RPM
_MAX_RETRIES = 4
_RETRY_BASE = 2.0          # exponential backoff base (seconds)
_EMBEDDING_DIM = 3072      # gemini-embedding-001 output dimension

_client = genai.Client(api_key=GOOGLE_API_KEY)


def _embed_batch(
    texts: list[str],
    task_type: str,
    attempt: int = 0,
) -> list[list[float]]:
    """Call Gemini embed_content for a batch of texts, with exponential backoff.

    Args:
        texts: List of strings to embed (≤ _BATCH_SIZE).
        task_type: "RETRIEVAL_DOCUMENT" for stored chunks, "RETRIEVAL_QUERY" for queries.
        attempt: Current retry attempt number.

    Returns:
        List of embedding vectors, one per input text.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    try:
        response = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [list(e.values) for e in response.embeddings]

    except Exception as exc:
        if attempt >= _MAX_RETRIES:
            raise RuntimeError(
                f"Gemini embed_content failed after {_MAX_RETRIES} retries: {exc}"
            ) from exc

        # 429 quota exhausted needs a much longer cooldown than other errors
        is_quota = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
        wait = 65.0 if is_quota else _RETRY_BASE ** attempt
        logger.warning(
            "Gemini embed error (attempt %d/%d), retrying in %.0fs: %s",
            attempt + 1, _MAX_RETRIES, wait, exc,
        )
        time.sleep(wait)
        return _embed_batch(texts, task_type, attempt + 1)


def embed_chunks(
    chunks: list[dict],
    batch_size: int = _BATCH_SIZE,
) -> list[tuple[str, list[float], dict]]:
    """Generate RETRIEVAL_DOCUMENT embeddings for a list of text chunks.

    Args:
        chunks: Output of ``chunk_sections()`` — list of dicts with ``text``
            and ``metadata`` keys.
        batch_size: Number of texts per Gemini API call (max 100).

    Returns:
        List of ``(chunk_text, embedding_vector, metadata)`` tuples in the
        same order as the input chunks. On permanent API failure, a zero
        vector of the correct dimension is substituted so the pipeline continues.
    """
    if not chunks:
        logger.warning("embed_chunks received empty chunk list")
        return []

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set — cannot generate embeddings")

    batch_size = min(batch_size, _BATCH_SIZE)
    results: list[tuple[str, list[float], dict]] = []
    total = len(chunks)

    for start in range(0, total, batch_size):
        batch = chunks[start: start + batch_size]
        texts = [c["text"] for c in batch]

        logger.info(
            "Embedding batch %d–%d of %d chunks…",
            start + 1, min(start + batch_size, total), total,
        )

        try:
            vectors = _embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")
        except RuntimeError as exc:
            logger.error(
                "Batch %d–%d failed permanently: %s — substituting zero vectors",
                start + 1, min(start + batch_size, total), exc,
            )
            vectors = [[0.0] * _EMBEDDING_DIM] * len(batch)

        for chunk, vector in zip(batch, vectors):
            results.append((chunk["text"], vector, chunk["metadata"]))

        if start + batch_size < total:
            time.sleep(_INTER_BATCH_SLEEP)

    logger.info(
        "Embedded %d chunk(s) for ticker %s",
        len(results), chunks[0]["metadata"].get("ticker", "?"),
    )
    return results


def embed_query(query: str) -> list[float]:
    """Embed a single query string for nearest-neighbour retrieval.

    Uses ``RETRIEVAL_QUERY`` task type so Gemini optimises the vector for
    cosine similarity against ``RETRIEVAL_DOCUMENT`` vectors.

    Args:
        query: Natural language query string.

    Returns:
        Embedding vector as a list of floats.

    Raises:
        ValueError: If GOOGLE_API_KEY is not set.
        RuntimeError: If the API call fails after retries.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set — cannot embed query")

    return _embed_batch([query], task_type="RETRIEVAL_QUERY")[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.data.edgar_client import EdgarClient
    from src.data.filing_parser import parse_filing
    from src.rag.chunker import chunk_sections

    edgar = EdgarClient(max_filings_per_type=1)
    filings = edgar.get_filings("AAPL", form_types=["10-K"])

    if not filings or not filings[0].fetch_success:
        print("Could not fetch AAPL 10-K")
    else:
        f = filings[0]
        sections = parse_filing(f.html_content, source_file=f.metadata.document_url)
        chunks = chunk_sections(
            sections[:1],  # first section only to keep test fast
            ticker="AAPL",
            filing_type=f.metadata.form_type,
            filing_date=f.metadata.filing_date,
        )
        embedded = embed_chunks(chunks[:5])

        print(f"\nEmbedded {len(embedded)} chunk(s)")
        for text, vec, meta in embedded:
            print(f"  [{meta['section_name']}] dim={len(vec)}  text[:60]: {text[:60]}…")

        print("\nQuery embedding:")
        q_vec = embed_query("What are the main risk factors?")
        print(f"  dim={len(q_vec)}  first 5 values: {q_vec[:5]}")
