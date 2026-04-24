"""Parse raw SEC EDGAR HTML filings into named sections.

EDGAR formatting varies wildly across companies and years, so multiple
strategies are applied in order and the one that finds the most sections wins.
"""

import logging
import re
import warnings
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

# ── Section definitions ────────────────────────────────────────────────────────

# Each tuple: (canonical_name, list_of_regex_patterns matched case-insensitively)
_SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "Management's Discussion and Analysis",
        [
            r"management.s discussion and analysis",
            r"\bmd&a\b",
            r"item\s*7\.?\s*management.s discussion",
            r"item\s*2\.?\s*management.s discussion",
        ],
    ),
    (
        "Risk Factors",
        [
            r"risk factors",
            r"item\s*1a\.?\s*risk factors",
        ],
    ),
    (
        "Financial Statements",
        [
            r"financial statements",
            r"consolidated (balance sheet|statements? of (income|operations|earnings|comprehensive))",
            r"item\s*8\.?\s*financial statements",
            r"item\s*1\.?\s*financial statements",
        ],
    ),
    (
        "Notes to Financial Statements",
        [
            r"notes to (consolidated )?financial statements",
        ],
    ),
]

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "b", "strong", "p"}
_MAX_SECTION_CHARS = 50_000
_CHARS_PER_PAGE = 3_500


def _matches(text: str, patterns: list[str]) -> bool:
    """Return True if any pattern matches text (case-insensitive)."""
    t = text.lower().strip()
    return any(re.search(p, t) for p in patterns)


def _section_for_text(text: str) -> Optional[str]:
    """Return the canonical section name if text matches any pattern, else None."""
    for name, patterns in _SECTION_PATTERNS:
        if _matches(text, patterns):
            return name
    return None


def _estimate_page(char_offset: int) -> int:
    return max(1, char_offset // _CHARS_PER_PAGE + 1)


def _clean(text: str) -> str:
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


@dataclass
class _Section:
    section_name: str
    text_content: str
    char_offset: int

    @property
    def estimated_page(self) -> int:
        return _estimate_page(self.char_offset)


# ── Strategy 1: semantic heading tags ─────────────────────────────────────────

def _strategy_heading_tags(soup: BeautifulSoup) -> list[_Section]:
    """Walk heading/bold tags and slice content between matched headings."""
    full_text = soup.get_text(" ")
    hits: list[tuple[int, str, Tag]] = []

    for tag in soup.find_all(_HEADING_TAGS):
        heading = tag.get_text(" ", strip=True)
        if not 5 <= len(heading) <= 200:
            continue
        name = _section_for_text(heading)
        if name is None:
            continue
        offset = full_text.find(heading[:50])
        hits.append((max(0, offset), name, tag))

    # deduplicate — keep first occurrence of each section name
    seen: set[str] = set()
    unique: list[tuple[int, str, Tag]] = []
    for offset, name, tag in sorted(hits, key=lambda x: x[0]):
        if name not in seen:
            seen.add(name)
            unique.append((offset, name, tag))

    sections: list[_Section] = []
    for offset, name, tag in unique:
        parts: list[str] = []
        collected = 0
        node = tag.next_sibling
        while node is not None and collected < _MAX_SECTION_CHARS:
            if isinstance(node, Tag) and node.name in _HEADING_TAGS:
                node_text = node.get_text(" ", strip=True)
                if _section_for_text(node_text) is not None:
                    break
            chunk = node.get_text(" ") if isinstance(node, Tag) else str(node)
            parts.append(chunk)
            collected += len(chunk)
            node = node.next_sibling

        cleaned = _clean(" ".join(parts))
        if len(cleaned) >= 100:
            sections.append(_Section(name, cleaned[:_MAX_SECTION_CHARS], offset))

    return sections


# ── Strategy 2: TOC anchor links ──────────────────────────────────────────────

def _strategy_toc_anchors(soup: BeautifulSoup) -> list[_Section]:
    """Resolve <a href="#id"> table-of-contents links to their targets."""
    toc: list[tuple[str, str]] = []  # (anchor_id, section_name)
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if not href.startswith("#"):
            continue
        name = _section_for_text(a.get_text(" ", strip=True))
        if name:
            toc.append((href[1:], name))

    if not toc:
        return []

    full_text = soup.get_text()
    seen: set[str] = set()
    sections: list[_Section] = []

    for anchor_id, name in toc:
        if name in seen:
            continue
        seen.add(name)

        target = soup.find(id=anchor_id) or soup.find(attrs={"name": anchor_id})
        if target is None:
            continue

        parts: list[str] = []
        collected = 0
        node = target.next_sibling
        other_ids = {aid for aid, _ in toc if aid != anchor_id}

        while node is not None and collected < _MAX_SECTION_CHARS:
            if isinstance(node, Tag):
                if any(
                    node.find("a", id=aid) or node.find("a", attrs={"name": aid})
                    for aid in other_ids
                ):
                    break
                chunk = node.get_text(" ")
            else:
                chunk = str(node)
            parts.append(chunk)
            collected += len(chunk)
            node = node.next_sibling

        cleaned = _clean(" ".join(parts))
        if len(cleaned) < 100:
            continue

        snippet = cleaned[:40]
        pos = full_text.find(snippet)
        offset = max(0, pos) if pos != -1 else 0
        sections.append(_Section(name, cleaned[:_MAX_SECTION_CHARS], offset))

    return sections


# ── Strategy 3: plain-text line scan ──────────────────────────────────────────

def _strategy_text_scan(soup: BeautifulSoup) -> list[_Section]:
    """Scan plain text line-by-line for section headers."""
    lines = soup.get_text("\n").splitlines()

    hits: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not 4 <= len(stripped) <= 300:
            continue
        name = _section_for_text(stripped)
        if name:
            hits.append((i, name))

    seen: set[str] = set()
    unique: list[tuple[int, str]] = []
    for idx, name in hits:
        if name not in seen:
            seen.add(name)
            unique.append((idx, name))

    sections: list[_Section] = []
    for j, (line_idx, name) in enumerate(unique):
        end_idx = unique[j + 1][0] if j + 1 < len(unique) else len(lines)
        raw = "\n".join(lines[line_idx + 1: end_idx])
        cleaned = _clean(raw)
        if len(cleaned) < 100:
            continue
        char_offset = sum(len(l) + 1 for l in lines[:line_idx])
        sections.append(_Section(name, cleaned[:_MAX_SECTION_CHARS], char_offset))

    return sections


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_filing(html_content: str, source_file: str = "") -> list[dict]:
    """Parse a raw EDGAR filing HTML string into named sections.

    Applies three strategies in order and returns the result from whichever
    finds the most sections. Sections missing from a filing are simply omitted.

    Args:
        html_content: Raw HTML string from an EDGAR filing document.
        source_file: Optional label used in log messages only.

    Returns:
        List of dicts with keys:
            section_name (str): e.g. "Risk Factors",
            text_content (str): Plain text of the section (≤50,000 chars),
            estimated_page (int): Approximate page number,
            char_offset (int): Character position in the raw document.
        Returns empty list if no sections were found.
    """
    if not html_content or len(html_content) < 500:
        logger.warning(
            "Filing content too short to parse: %s (%d chars)",
            source_file, len(html_content),
        )
        return []

    try:
        soup = BeautifulSoup(html_content, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
        except Exception as exc:
            logger.error("BeautifulSoup parse failed for %s: %s", source_file, exc)
            return []

    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    best: list[_Section] = []
    for strategy in [_strategy_heading_tags, _strategy_toc_anchors, _strategy_text_scan]:
        try:
            found = strategy(soup)
            if len(found) > len(best):
                best = found
                logger.info(
                    "Strategy %s: %d section(s) for %s",
                    strategy.__name__, len(found), source_file or "filing",
                )
        except Exception as exc:
            logger.warning("Strategy %s raised for %s: %s", strategy.__name__, source_file, exc)

    if not best:
        logger.warning("No sections extracted from %s", source_file)
        return []

    return [
        {
            "section_name": s.section_name,
            "text_content": s.text_content,
            "estimated_page": s.estimated_page,
            "char_offset": s.char_offset,
        }
        for s in sorted(best, key=lambda s: s.char_offset)
    ]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from src.data.edgar_client import EdgarClient

    edgar = EdgarClient(max_filings_per_type=1)
    filings = edgar.get_filings("AAPL", form_types=["10-K"])

    if not filings:
        print("No filings fetched — cannot test parser")
        sys.exit(1)

    filing = filings[0]
    print(f"\nParsing {filing.metadata.form_type} filed {filing.metadata.filing_date}")
    print(f"HTML length: {len(filing.html_content):,} chars")

    sections = parse_filing(filing.html_content, source_file=filing.metadata.document_url)

    if not sections:
        print("No sections found.")
    else:
        print(f"\nFound {len(sections)} section(s):\n")
        for s in sections:
            preview = s["text_content"][:200].replace("\n", " ")
            print(f"  [p{s['estimated_page']:>3}] {s['section_name']}")
            print(f"         {len(s['text_content']):,} chars | {preview}…")
            print()
