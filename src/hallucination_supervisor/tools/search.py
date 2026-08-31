from __future__ import annotations

import logging
from typing import List

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import settings

logger = logging.getLogger(__name__)


class SearchProviderError(Exception):
    """
    Raised when DuckDuckGo search fails after exhausting all retry
    attempts, or when the `duckduckgo-search` package is not installed.
    """


class EvidenceSearcher:
    """
    Resilient wrapper around DuckDuckGo web search, used to gather
    grounding evidence for atomic factual claims.

    DuckDuckGo is the only supported backend: it requires no API key and
    no billing account, keeping the pipeline fully free to run. Requests
    are protected with retry-and-backoff against transient network errors
    and DuckDuckGo's own rate limiting.

    Attributes:
        request_timeout: Per-request timeout, in seconds, sourced from
            `settings.request_timeout_seconds`.
        max_retries: Maximum retry attempts, sourced from
            `settings.max_retries`.
    """

    def __init__(self) -> None:
        """
        Initialize the searcher and lazily construct the DuckDuckGo
        client.

        Raises:
            SearchProviderError: If the `duckduckgo-search` package is
                not installed.
        """
        self.request_timeout: float = settings.request_timeout_seconds
        self.max_retries: int = settings.max_retries

        try:
            from ddgs import DDGS  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SearchProviderError(
                "The 'duckduckgo-search' package is required. "
                "Install it with `pip install duckduckgo-search`."
            ) from exc
        self._ddgs_cls = DDGS

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(
            multiplier=settings.backoff_multiplier_seconds, min=1, max=20
        ),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, RuntimeError)),
    )
    def _search_duckduckgo(self, query: str, max_results: int) -> List[str]:
        """
        Execute a single search query against DuckDuckGo.

        Args:
            query: The search query string.
            max_results: Maximum number of results to request.

        Returns:
            A list of raw text snippets ('body' fields) extracted from
            the DuckDuckGo response, each prefixed with source
            attribution.

        Raises:
            RuntimeError: Wraps any exception raised by the DDGS client
                (including DuckDuckGo's own rate-limit errors) so
                tenacity's retry predicate can catch it uniformly.
        """
        try:
            with self._ddgs_cls(timeout=self.request_timeout) as ddgs:
                raw_results = list(
                    ddgs.text(query, max_results=max_results, safesearch="moderate")
                )
        except Exception as exc:  # noqa: BLE001 - normalize all provider errors
            logger.warning("DuckDuckGo search failed for query %r: %s", query, exc)
            raise RuntimeError(f"DuckDuckGo search failed: {exc}") from exc

        snippets: List[str] = []
        for item in raw_results:
            body = item.get("body")
            title = item.get("title", "")
            href = item.get("href", "")
            if body:
                snippets.append(f"[Source: {title} ({href})]\n{body.strip()}")
        return snippets

    def search_claims(self, query: str, max_results: int = 3) -> str:
        """
        Search DuckDuckGo for evidence relevant to a factual claim and
        return it as a single, unified, human-readable context string.

        Args:
            query: The search query (typically the atomic claim text, or
                a reformulated question derived from it).
            max_results: Maximum number of results to aggregate. Must be
                a positive integer.

        Returns:
            A newline-delimited string concatenating all retrieved
            snippets, each prefixed with its source attribution. Returns
            an empty string if no query is provided or no results are
            found.

        Raises:
            SearchProviderError: If DuckDuckGo fails on every retry
                attempt (commonly due to rate limiting under heavy
                concurrent use).
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            logger.debug("search_claims called with an empty query; returning ''.")
            return ""

        safe_max_results = max(1, max_results)

        try:
            snippets = self._search_duckduckgo(cleaned_query, safe_max_results)
        except Exception as exc:  # noqa: BLE001 - final catch after retries exhausted
            raise SearchProviderError(
                f"DuckDuckGo search failed after {self.max_retries} attempts "
                f"for query {cleaned_query!r}: {exc}"
            ) from exc

        if not snippets:
            logger.info("No search results found for query: %r", cleaned_query)
            return ""

        separator = "\n\n---\n\n"
        return separator.join(snippets)