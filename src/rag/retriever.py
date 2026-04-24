"""ChromaDB collection manager for AlphaLens filing chunks."""

import logging
import os
from typing import Any, Optional

import chromadb
from chromadb import Collection

from src.config import TOP_K
from src.rag.embeddings import embed_query

logger = logging.getLogger(__name__)

_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".chroma")
_COLLECTION_PREFIX = "alphalens_"


def _collection_name(ticker: str) -> str:
    """Normalise a ticker to a valid ChromaDB collection name."""
    return f"{_COLLECTION_PREFIX}{ticker.lower()}"


class FilingRetriever:
    """Manages ChromaDB collections for per-ticker filing chunks.

    Collections are persistent (stored under ``.chroma/`` at the project root)
    so embeddings survive across runs and are only recomputed when new filings
    are ingested.

    Args:
        chroma_path: Directory for ChromaDB persistent storage.
    """

    def __init__(self, chroma_path: str = _CHROMA_PATH) -> None:
        os.makedirs(chroma_path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=chroma_path)
        logger.info("ChromaDB initialised at %s", chroma_path)

    # ── Collection management ──────────────────────────────────────────────────

    def create_collection(self, ticker: str) -> Collection:
        """Get or create a ChromaDB collection for a ticker.

        Creating is idempotent — calling this when data already exists returns
        the existing collection without clearing it.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            ChromaDB Collection object.
        """
        name = _collection_name(ticker)
        collection = self._client.get_or_create_collection(
            name=name,
            metadata={"ticker": ticker.upper(), "hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' ready (%d items)", name, collection.count())
        return collection

    def delete_collection(self, ticker: str) -> None:
        """Delete the ChromaDB collection for a ticker (used to force re-ingest).

        Args:
            ticker: Stock ticker symbol.
        """
        name = _collection_name(ticker)
        try:
            self._client.delete_collection(name)
            logger.info("Deleted collection '%s'", name)
        except Exception as exc:
            logger.warning("Could not delete collection '%s': %s", name, exc)

    def collection_count(self, ticker: str) -> int:
        """Return the number of chunks stored for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Chunk count, or 0 if the collection does not exist.
        """
        try:
            col = self._client.get_collection(_collection_name(ticker))
            return col.count()
        except Exception:
            return 0

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_filing_chunks(
        self,
        ticker: str,
        chunks_with_embeddings: list[tuple[str, list[float], dict]],
    ) -> None:
        """Store pre-embedded chunks in the ticker's ChromaDB collection.

        Existing chunks with the same ID are silently overwritten so ingestion
        is idempotent (safe to re-run on the same filing).

        Args:
            ticker: Stock ticker symbol.
            chunks_with_embeddings: Output of ``embed_chunks()`` — list of
                ``(chunk_text, embedding_vector, metadata)`` tuples.
        """
        if not chunks_with_embeddings:
            logger.warning("add_filing_chunks: empty input for %s", ticker)
            return

        collection = self.create_collection(ticker)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for text, vector, meta in chunks_with_embeddings:
            # Build a deterministic ID from filing identity + chunk index
            chunk_id = (
                f"{meta.get('ticker','?')}_"
                f"{meta.get('filing_type','?')}_"
                f"{meta.get('filing_date','?')}_"
                f"{meta.get('chunk_index', 0)}"
            ).replace(" ", "_")

            # ChromaDB metadata values must be str/int/float/bool
            safe_meta: dict[str, Any] = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in meta.items()
            }

            ids.append(chunk_id)
            embeddings.append(vector)
            documents.append(text)
            metadatas.append(safe_meta)

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                "Upserted %d chunk(s) into collection '%s' (total: %d)",
                len(ids), _collection_name(ticker), collection.count(),
            )
        except Exception as exc:
            logger.error("ChromaDB upsert failed for %s: %s", ticker, exc)
            raise

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve_relevant(
        self,
        ticker: str,
        query: str,
        top_k: int = TOP_K,
        section_filter: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve the most relevant chunks for a query.

        Args:
            ticker: Stock ticker symbol.
            query: Natural language query (e.g. "revenue growth drivers").
            top_k: Number of results to return.
            section_filter: If provided, restrict results to this section name
                (e.g. "Risk Factors"). Must match exactly as stored in metadata.

        Returns:
            List of result dicts, each with keys:
                ``text`` (str): Chunk text,
                ``distance`` (float): Cosine distance (lower = more similar),
                ``metadata`` (dict): Full citation metadata,
                ``rank`` (int): 1-based rank.
            Returns empty list if the collection does not exist or query fails.
        """
        try:
            collection = self._client.get_collection(_collection_name(ticker))
        except Exception:
            logger.warning("No collection found for %s — run ingestion first", ticker)
            return []

        if collection.count() == 0:
            logger.warning("Collection for %s is empty", ticker)
            return []

        try:
            query_vector = embed_query(query)
        except Exception as exc:
            logger.error("Failed to embed query for %s: %s", ticker, exc)
            return []

        # Build optional metadata filter
        where_clause: Optional[dict] = None
        if section_filter:
            where_clause = {"section_name": {"$eq": section_filter}}

        try:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_vector],
                "n_results": min(top_k, collection.count()),
                "include": ["documents", "metadatas", "distances"],
            }
            if where_clause is not None:
                query_kwargs["where"] = where_clause

            raw = collection.query(**query_kwargs)
        except Exception as exc:
            logger.error("ChromaDB query failed for %s: %s", ticker, exc)
            return []

        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: list[dict] = []
        for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
            results.append(
                {
                    "text": doc,
                    "distance": round(float(dist), 6),
                    "metadata": meta,
                    "rank": rank,
                }
            )

        logger.info(
            "Retrieved %d chunk(s) for query '%s' (ticker=%s, section=%s)",
            len(results), query[:50], ticker, section_filter or "any",
        )
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.data.edgar_client import EdgarClient
    from src.data.filing_parser import parse_filing
    from src.rag.chunker import chunk_sections
    from src.rag.embeddings import embed_chunks

    # ── Step 1: Fetch latest AAPL 10-K ────────────────────────────────────────
    print("\n[1/5] Fetching AAPL 10-K from EDGAR…")
    edgar = EdgarClient(max_filings_per_type=1)
    filings = edgar.get_filings("AAPL", form_types=["10-K"])
    assert filings and filings[0].fetch_success, "Could not fetch AAPL 10-K"
    filing = filings[0]
    print(f"  Got {filing.metadata.form_type} filed {filing.metadata.filing_date} "
          f"({len(filing.html_content):,} chars)")

    # ── Step 2: Parse into sections ────────────────────────────────────────────
    print("\n[2/5] Parsing sections…")
    sections = parse_filing(filing.html_content, source_file=filing.metadata.document_url)
    print(f"  Found {len(sections)} section(s): {[s['section_name'] for s in sections]}")

    # ── Step 3: Chunk ──────────────────────────────────────────────────────────
    print("\n[3/5] Chunking sections…")
    chunks = chunk_sections(
        sections,
        ticker="AAPL",
        filing_type=filing.metadata.form_type,
        filing_date=filing.metadata.filing_date,
        source_file=filing.metadata.document_url,
    )
    print(f"  {len(chunks)} chunk(s) produced")

    # ── Step 4: Embed ──────────────────────────────────────────────────────────
    print("\n[4/5] Embedding chunks (this takes ~30s)…")
    embedded = embed_chunks(chunks)
    print(f"  {len(embedded)} embedding(s) generated, dim={len(embedded[0][1])}")

    # ── Step 5: Store and retrieve ─────────────────────────────────────────────
    print("\n[5/5] Storing in ChromaDB and querying…")
    retriever = FilingRetriever()
    retriever.delete_collection("AAPL")   # fresh start for the test
    retriever.add_filing_chunks("AAPL", embedded)

    QUERY = "What are the main risk factors?"
    print(f"\n  Query: '{QUERY}'")
    print(f"  Section filter: Risk Factors\n")
    results = retriever.retrieve_relevant("AAPL", QUERY, top_k=3, section_filter="Risk Factors")

    for r in results:
        m = r["metadata"]
        print(f"  Rank {r['rank']}  dist={r['distance']:.4f}")
        print(f"    citation: {m['filing_type']} {m['filing_date']} · {m['section_name']} · p{m['estimated_page']}")
        print(f"    text[:150]: {r['text'][:150]}…")
        print()
