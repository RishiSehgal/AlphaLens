"""SEC EDGAR client — fetches 10-K/10-Q filing metadata and HTML content."""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "AlphaLens research@alphalens.dev",
    "Accept-Encoding": "gzip, deflate",
}
_EDGAR_BASE = "https://data.sec.gov"
_MIN_INTERVAL = 1.0 / 10  # 10 req/sec max


@dataclass
class FilingMetadata:
    """Metadata for a single SEC filing."""

    ticker: str
    cik: str
    form_type: str
    filing_date: str
    accession_number: str
    primary_document: str
    filing_url: str
    document_url: str
    period_of_report: str = ""


@dataclass
class FilingResult:
    """Full result from EDGAR: metadata plus raw HTML."""

    metadata: FilingMetadata
    html_content: str
    fetch_success: bool
    error: str = ""


class EdgarClient:
    """Client for SEC EDGAR — company search, filing index, and HTML download.

    Args:
        max_filings_per_type: Maximum number of filings to return per form type.
    """

    def __init__(self, max_filings_per_type: int = 3) -> None:
        self.max_filings_per_type = max_filings_per_type
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Sleep if needed to respect 10 req/sec limit."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, url: str, params: Optional[dict] = None) -> requests.Response:
        """Rate-limited GET with retries on transient errors.

        Args:
            url: Target URL.
            params: Optional query parameters.

        Returns:
            Response object.

        Raises:
            requests.RequestException: On failure after retries.
        """
        self._throttle()
        for attempt in range(3):
            try:
                resp = self._session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp
            except requests.exceptions.Timeout:
                logger.warning("EDGAR timeout on attempt %d: %s", attempt + 1, url)
                time.sleep(2 ** attempt)
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    logger.warning("EDGAR rate limited — sleeping 60s")
                    time.sleep(60)
                else:
                    raise
        raise requests.exceptions.ConnectionError(f"EDGAR request failed after retries: {url}")

    def _get_cik(self, ticker: str) -> str:
        """Resolve a ticker symbol to a zero-padded CIK.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            Zero-padded 10-digit CIK string.

        Raises:
            ValueError: If ticker is not found in EDGAR.
        """
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp = self._get(tickers_url)
        data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
        raise ValueError(f"Ticker '{ticker}' not found in EDGAR company list")

    def _get_filings_from_submissions(
        self, cik: str, form_types: list[str]
    ) -> list[FilingMetadata]:
        """Fetch filing list from the EDGAR submissions endpoint.

        Args:
            cik: Zero-padded CIK.
            form_types: List of form types to retrieve, e.g. ["10-K", "10-Q"].

        Returns:
            List of FilingMetadata sorted newest-first.
        """
        url = f"{_EDGAR_BASE}/submissions/CIK{cik}.json"
        resp = self._get(url)
        data = resp.json()

        filings_data = data.get("filings", {}).get("recent", {})
        forms = filings_data.get("form", [])
        dates = filings_data.get("filingDate", [])
        accessions = filings_data.get("accessionNumber", [])
        primary_docs = filings_data.get("primaryDocument", [])
        periods = filings_data.get("reportDate", [])

        results: list[FilingMetadata] = []
        counts: dict[str, int] = {ft: 0 for ft in form_types}

        for form, date, acc, doc, period in zip(forms, dates, accessions, primary_docs, periods):
            if form not in form_types:
                continue
            if counts[form] >= self.max_filings_per_type:
                continue

            acc_clean = acc.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/"
            document_url = f"{filing_url}{doc}"

            results.append(
                FilingMetadata(
                    ticker="",
                    cik=cik,
                    form_type=form,
                    filing_date=date,
                    accession_number=acc,
                    primary_document=doc,
                    filing_url=filing_url,
                    document_url=document_url,
                    period_of_report=period,
                )
            )
            counts[form] += 1

            if all(v >= self.max_filings_per_type for v in counts.values()):
                break

        return results

    def _fetch_html(self, url: str) -> tuple[str, bool, str]:
        """Download the raw HTML of a filing document.

        Args:
            url: Direct URL to the filing document.

        Returns:
            Tuple of (html_content, success_flag, error_message).
        """
        try:
            resp = self._get(url)
            return resp.text, True, ""
        except requests.exceptions.HTTPError as exc:
            msg = f"HTTP {exc.response.status_code if exc.response else '?'}: {url}"
            logger.error("Failed to fetch filing HTML: %s", msg)
            return "", False, msg
        except requests.exceptions.RequestException as exc:
            logger.error("Network error fetching filing HTML: %s", exc)
            return "", False, str(exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_filings(
        self,
        ticker: str,
        form_types: Optional[list[str]] = None,
    ) -> list[FilingResult]:
        """Fetch the most recent 10-K and 10-Q filings for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").
            form_types: Form types to retrieve. Defaults to ["10-K", "10-Q"].

        Returns:
            List of FilingResult objects (metadata + HTML), newest first.
            Returns empty list if ticker not found or no filings available.
        """
        if form_types is None:
            form_types = ["10-K", "10-Q"]

        try:
            cik = self._get_cik(ticker)
            logger.info("Resolved %s → CIK %s", ticker, cik)
        except ValueError as exc:
            logger.error("CIK lookup failed for %s: %s", ticker, exc)
            return []
        except requests.exceptions.RequestException as exc:
            logger.error("Network error during CIK lookup for %s: %s", ticker, exc)
            return []

        try:
            filings = self._get_filings_from_submissions(cik, form_types)
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch submissions for CIK %s: %s", cik, exc)
            return []

        if not filings:
            logger.warning("No %s filings found for %s (CIK %s)", form_types, ticker, cik)
            return []

        results: list[FilingResult] = []
        for meta in filings:
            meta.ticker = ticker
            logger.info(
                "Fetching %s filed %s → %s", meta.form_type, meta.filing_date, meta.document_url
            )
            html, ok, err = self._fetch_html(meta.document_url)
            results.append(FilingResult(metadata=meta, html_content=html, fetch_success=ok, error=err))

        logger.info("Retrieved %d filing(s) for %s", len(results), ticker)
        return results

    def get_filing_urls(
        self, ticker: str, form_types: Optional[list[str]] = None
    ) -> list[dict]:
        """Return filing metadata only (no HTML download).

        Args:
            ticker: Stock ticker symbol.
            form_types: Form types to include.

        Returns:
            List of dicts with keys: ticker, cik, form_type, filing_date,
            document_url, period_of_report.
        """
        if form_types is None:
            form_types = ["10-K", "10-Q"]

        try:
            cik = self._get_cik(ticker)
            filings = self._get_filings_from_submissions(cik, form_types)
        except (ValueError, requests.exceptions.RequestException) as exc:
            logger.error("get_filing_urls failed for %s: %s", ticker, exc)
            return []

        return [
            {
                "ticker": ticker,
                "cik": meta.cik,
                "form_type": meta.form_type,
                "filing_date": meta.filing_date,
                "document_url": meta.document_url,
                "period_of_report": meta.period_of_report,
            }
            for meta in filings
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    client = EdgarClient(max_filings_per_type=2)

    print("\n=== Filing URLs (no HTML) ===")
    urls = client.get_filing_urls("AAPL")
    for u in urls:
        print(f"  {u['form_type']} {u['filing_date']} → {u['document_url']}")

    print("\n=== Full Filings (with HTML) ===")
    filings = client.get_filings("AAPL", form_types=["10-K"])
    for f in filings:
        status = "OK" if f.fetch_success else f"FAIL: {f.error}"
        print(
            f"  {f.metadata.form_type} {f.metadata.filing_date}"
            f" — {status} — {len(f.html_content):,} chars"
        )
