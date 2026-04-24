"""Tests for the RAG pipeline: chunker, embeddings, retriever."""

import pytest
from unittest.mock import MagicMock, patch


# ── Chunker ───────────────────────────────────────────────────────────────────

SAMPLE_SECTIONS = [
    {
        "section_name": "Risk Factors",
        "text_content": (
            "The following risk factors could materially affect our business. "
            "We face intense competition from established players in the market. "
            "Our revenue depends significantly on a single product line. " * 30
        ),
        "estimated_page": 12,
        "char_offset": 0,
    },
    {
        "section_name": "MD&A",
        "text_content": (
            "Management's Discussion and Analysis of Financial Condition. "
            "Revenue increased 8% year over year driven by services growth. "
            "Gross margin improved to 46% from 44% in the prior year. " * 30
        ),
        "estimated_page": 25,
        "char_offset": 5000,
    },
]


def test_chunk_sections_returns_list():
    from src.rag.chunker import chunk_sections
    chunks = chunk_sections(SAMPLE_SECTIONS, ticker="AAPL", filing_type="10-K", filing_date="2024-11-01")
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_sections_metadata_fields():
    from src.rag.chunker import chunk_sections
    chunks = chunk_sections(SAMPLE_SECTIONS, ticker="AAPL", filing_type="10-K", filing_date="2024-11-01")
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        meta = chunk["metadata"]
        assert meta["ticker"] == "AAPL"
        assert meta["filing_type"] == "10-K"
        assert meta["filing_date"] == "2024-11-01"
        assert "section_name" in meta
        assert "chunk_index" in meta


def test_chunk_sections_text_not_empty():
    from src.rag.chunker import chunk_sections
    chunks = chunk_sections(SAMPLE_SECTIONS, ticker="AAPL", filing_type="10-K", filing_date="2024-11-01")
    for chunk in chunks:
        assert len(chunk["text"].strip()) > 0


def test_chunk_sections_empty_input():
    from src.rag.chunker import chunk_sections
    result = chunk_sections([], ticker="AAPL", filing_type="10-K", filing_date="2024-11-01")
    assert result == []


def test_chunk_sections_preserves_section_name():
    from src.rag.chunker import chunk_sections
    chunks = chunk_sections(SAMPLE_SECTIONS, ticker="AAPL", filing_type="10-K", filing_date="2024-11-01")
    section_names = {c["metadata"]["section_name"] for c in chunks}
    assert "Risk Factors" in section_names
    assert "MD&A" in section_names


def test_chunk_sections_chunk_index_is_sequential():
    from src.rag.chunker import chunk_sections
    chunks = chunk_sections([SAMPLE_SECTIONS[0]], ticker="AAPL", filing_type="10-K", filing_date="2024-11-01")
    indices = [c["metadata"]["chunk_index"] for c in chunks]
    assert indices == list(range(len(indices)))


# ── Embeddings ────────────────────────────────────────────────────────────────

@patch("src.rag.embeddings._client")
def test_embed_chunks_returns_tuples(mock_client):
    from src.rag.embeddings import embed_chunks

    fake_dim = 3072
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * fake_dim
    mock_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding, mock_embedding])

    chunks = [
        {"text": "Apple reported strong revenue growth.", "metadata": {"ticker": "AAPL", "section_name": "MD&A"}},
        {"text": "Risk factors include intense competition.", "metadata": {"ticker": "AAPL", "section_name": "Risk Factors"}},
    ]

    result = embed_chunks(chunks)

    assert len(result) == 2
    for text, vector, meta in result:
        assert isinstance(text, str)
        assert isinstance(vector, list)
        assert len(vector) == fake_dim
        assert isinstance(meta, dict)


@patch("src.rag.embeddings._client")
def test_embed_chunks_empty_input(mock_client):
    from src.rag.embeddings import embed_chunks
    result = embed_chunks([])
    assert result == []
    mock_client.models.embed_content.assert_not_called()


@patch("src.rag.embeddings._client")
def test_embed_query_returns_vector(mock_client):
    from src.rag.embeddings import embed_query

    mock_embedding = MagicMock()
    mock_embedding.values = [0.05] * 3072
    mock_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

    result = embed_query("What are the main risk factors?")
    assert isinstance(result, list)
    assert len(result) == 3072


@patch("src.rag.embeddings._client")
def test_embed_batch_retries_on_error(mock_client):
    from src.rag.embeddings import _embed_batch

    call_count = 0
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 3072

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("Transient error")
        return MagicMock(embeddings=[mock_embedding])

    mock_client.models.embed_content.side_effect = side_effect

    with patch("src.rag.embeddings.time.sleep"):
        result = _embed_batch(["test text"], task_type="RETRIEVAL_DOCUMENT")

    assert call_count == 2
    assert len(result) == 1


# ── Filing Parser ─────────────────────────────────────────────────────────────

SAMPLE_HTML = """
<html><body>
<h2>Item 1A. Risk Factors</h2>
<p>We face significant competition from larger companies with greater resources.
Our business may be materially adversely affected by changes in macroeconomic conditions.</p>

<h2>Item 7. Management's Discussion and Analysis</h2>
<p>Total net revenues increased 8% year over year. Services revenue grew 14%.
Gross margin improved 200 basis points driven by favorable product mix.</p>

<h2>Item 8. Financial Statements</h2>
<p>Consolidated Statements of Operations for fiscal year 2024.</p>
</body></html>
"""


def test_parse_filing_returns_sections():
    from src.data.filing_parser import parse_filing
    sections = parse_filing(SAMPLE_HTML, source_file="test.htm")
    assert isinstance(sections, list)
    assert len(sections) > 0


def test_parse_filing_section_has_required_keys():
    from src.data.filing_parser import parse_filing
    sections = parse_filing(SAMPLE_HTML, source_file="test.htm")
    for sec in sections:
        assert "section_name" in sec
        assert "text_content" in sec
        assert "estimated_page" in sec


def test_parse_filing_captures_text():
    from src.data.filing_parser import parse_filing
    sections = parse_filing(SAMPLE_HTML, source_file="test.htm")
    all_text = " ".join(s["text_content"] for s in sections).lower()
    # At least one of the key sections should appear in the text
    assert any(kw in all_text for kw in ["competition", "revenue", "margin", "risk"])


def test_parse_filing_empty_html():
    from src.data.filing_parser import parse_filing
    sections = parse_filing("", source_file="empty.htm")
    assert isinstance(sections, list)


# ── Retriever ─────────────────────────────────────────────────────────────────

def test_filing_retriever_initializes():
    from src.rag.retriever import FilingRetriever
    with patch("src.rag.retriever.chromadb.PersistentClient") as mock_chroma:
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        retriever = FilingRetriever()
        assert retriever is not None


def test_filing_retriever_add_and_query():
    from src.rag.retriever import FilingRetriever

    with patch("src.rag.retriever.chromadb.PersistentClient") as mock_chroma:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2  # must be int for min() call

        mock_collection.query.return_value = {
            "documents": [["Risk of competition.", "Revenue grew 8%."]],
            "metadatas": [[
                {"ticker": "AAPL", "section_name": "Risk Factors", "chunk_index": 0},
                {"ticker": "AAPL", "section_name": "MD&A", "chunk_index": 1},
            ]],
            "distances": [[0.12, 0.25]],  # real floats required
        }
        client_mock = mock_chroma.return_value
        # FilingRetriever.__init__ calls get_or_create_collection for the default collection
        client_mock.get_or_create_collection.return_value = mock_collection
        # retrieve_relevant calls get_collection (not get_or_create)
        client_mock.get_collection.return_value = mock_collection

        retriever = FilingRetriever()

        with patch("src.rag.retriever.embed_query", return_value=[0.1] * 3072):
            results = retriever.retrieve_relevant("AAPL", "What are the main risk factors?", top_k=2)

        assert isinstance(results, list)
        assert len(results) >= 1
        for chunk in results:
            assert "text" in chunk
            assert "metadata" in chunk
