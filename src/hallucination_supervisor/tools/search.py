import logging
from typing import List

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.hallucination_supervisor.config import settings

logger = logging.getLogger(__name__)


class SearchProviderError(Exception):
    """Raised when DuckDuckGo search fails after all retries, or ddgs isn't installed."""


class EvidenceSearcher:
    """
    Thin wrapper around DuckDuckGo search (via the `ddgs` package). This is
    the only search backend we support.
    """

    def __init__(self) -> None:
        self.request_timeout = settings.request_timeout_seconds
        self.max_retries = settings.max_retries

        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise SearchProviderError("ddgs is required. pip install ddgs") from exc
        self._ddgs_cls = DDGS

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.backoff_multiplier_seconds, min=1, max=20),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, RuntimeError)),
    )
    def _search(self, query: str, max_results: int) -> List[str]:
        try:
            with self._ddgs_cls(timeout=self.request_timeout) as ddgs:
                raw = list(ddgs.text(query, max_results=max_results, safesearch="moderate"))
        except Exception as exc:
            logger.warning("Search failed for %r: %s", query, exc)
            raise RuntimeError(f"DuckDuckGo search failed: {exc}") from exc

        snippets = []
        for item in raw:
            body = item.get("body")
            if body:
                snippets.append(f"[Source: {item.get('title', '')} ({item.get('href', '')})]\n{body.strip()}")
        return snippets

    def search_claims(self, query: str, max_results: int = 3) -> str:
        """Runs a search and returns the results joined into one string."""
        query = query.strip()
        if not query:
            return ""

        try:
            snippets = self._search(query, max(1, max_results))
        except Exception as exc:
            raise SearchProviderError(
                f"Search failed after {self.max_retries} attempts for {query!r}: {exc}"
            ) from exc

        return "\n\n---\n\n".join(snippets) if snippets else ""