"""LangGraph shared state definition for the AlphaLens pipeline.

Reducers on `metadata` and `error_log` allow parallel branches (rag_citation +
quant_analysis) to update those fields without clobbering each other.
"""

import operator
from typing import Annotated, TypedDict


def _merge_metadata(a: dict, b: dict) -> dict:
    """Deep-merge two metadata dicts, combining agent_latencies and sources_status sub-dicts."""
    merged = {**a}
    for k, v in b.items():
        if k in ("agent_latencies", "sources_status") and isinstance(v, dict):
            merged.setdefault(k, {}).update(v)
        else:
            merged[k] = v
    return merged


class AlphaLensState(TypedDict):
    ticker: str
    financial_data: dict
    rag_chunks: list[dict]
    quant_results: dict
    risk_flags: list[dict]
    verification: dict
    report: dict
    metadata: Annotated[dict, _merge_metadata]
    error_log: Annotated[list[str], operator.add]
    chat_history: list[dict]
