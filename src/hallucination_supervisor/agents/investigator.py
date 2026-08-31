"""
Evidence Investigation agent for the Multi-Agent LLM Hallucination
Supervisor.

Responsible for generating optimized search queries for each atomic
claim and concurrently retrieving supporting/contradicting evidence via
`tools.search.EvidenceSearcher`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import instructor
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import settings
from ..schemas import AtomicClaim, ClaimEvidence
from ..tools.search import EvidenceSearcher, SearchProviderError

logger = logging.getLogger(__name__)

_QUERY_GEN_SYSTEM_PROMPT = (
    "You are a search-query optimization engine for a fact-checking system. "
    "Given a single factual claim, produce 1 to 2 concise, high-precision "
    "web search queries that would surface evidence to confirm or refute "
    "it. Queries must contain concrete entities, names, dates, or numbers "
    "from the claim — avoid vague or overly broad phrasing. Do not include "
    "quotation marks or search operators unless essential."
)


class InvestigationError(Exception):
    """
    Raised when the Evidence Investigation agent fails to generate search
    queries or retrieve evidence for a claim after exhausting all
    configured retry attempts.
    """


class _SearchQueries(BaseModel):
    """
    Internal, LLM-facing schema for optimized search queries generated
    from a single atomic claim.

    Attributes:
        queries: A list of 1-2 concise, high-precision search queries.
    """

    model_config = ConfigDict(extra="forbid")

    queries: List[str] = Field(
        ...,
        min_length=1,
        max_length=2,
        description="1-2 optimized search queries derived from the claim.",
    )

    @field_validator("queries")
    @classmethod
    def _reject_empty_queries(cls, v: List[str]) -> List[str]:
        """Ensure no query in the list is empty or whitespace-only.

        Args:
            v: The candidate list of query strings.

        Returns:
            The validated list of queries.

        Raises:
            ValueError: If any query is empty/whitespace-only.
        """
        for q in v:
            if not q.strip():
                raise ValueError("Search queries must not be empty or whitespace-only.")
        return v


class EvidenceInvestigator:
    """
    Generates optimized search queries for atomic claims and concurrently
    retrieves supporting/contradicting evidence for each.

    Query generation uses a lightweight, low-temperature LLM call
    (structured via `instructor`). Evidence retrieval delegates to
    `EvidenceSearcher`, which is synchronous/blocking, so each claim's
    retrieval is dispatched to a worker thread via `asyncio.to_thread` and
    all claims are processed concurrently with `asyncio.gather`.

    Attributes:
        provider: LLM provider identifier used for query generation.
        model_name: Model identifier passed to `litellm`.
        max_retries: Maximum retry attempts for both query generation and
            evidence retrieval.
        max_results_per_query: Maximum search results requested per query.
        searcher: The `EvidenceSearcher` instance used for retrieval.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        max_results_per_query: int = 3,
        searcher: Optional[EvidenceSearcher] = None,
    ) -> None:
        """
        Initialize the evidence investigator.

        Args:
            provider: Optional override for the LLM provider used for
                query generation. Defaults to `settings.default_llm_provider`.
            model_name: Optional override for the model identifier.
                Defaults to `settings.default_model_name`.
            max_results_per_query: Maximum number of search results to
                request per generated query.
            searcher: Optional pre-constructed `EvidenceSearcher` instance
                (useful for dependency injection/testing). If omitted, a
                new instance is constructed from `settings`.

        Raises:
            InvestigationError: If the `litellm` package is not installed.
        """
        try:
            import litellm  # type: ignore[import-untyped]
        except ImportError as exc:
            raise InvestigationError(
                "The 'litellm' package is required for EvidenceInvestigator. "
                "Install it with `pip install litellm`."
            ) from exc

        self.provider: str = provider or settings.default_llm_provider
        self.model_name: str = model_name or settings.default_model_name
        self.max_retries: int = settings.max_retries
        self.request_timeout: float = settings.request_timeout_seconds
        self.max_results_per_query: int = max(1, max_results_per_query)

        self._resolved_model = self._resolve_model_string(self.provider, self.model_name)

        try:
            self._async_client = instructor.from_litellm(litellm.acompletion)
        except Exception as exc:  # noqa: BLE001
            raise InvestigationError(
                f"Failed to initialize instructor-patched async litellm client: {exc}"
            ) from exc

        try:
            self.searcher: EvidenceSearcher = searcher or EvidenceSearcher()
        except SearchProviderError as exc:
            raise InvestigationError(
                f"Failed to initialize EvidenceSearcher: {exc}"
            ) from exc

    @staticmethod
    def _resolve_model_string(provider: str, model_name: str) -> str:
        """
        Construct the `litellm`-compatible model identifier from a
        provider/model pair.

        Args:
            provider: LLM provider identifier.
            model_name: Bare model name.

        Returns:
            A model string formatted as `litellm` expects.
        """
        if provider == "openai":
            return model_name
        if provider == "groq":
            return f"groq/{model_name}"
        if provider == "ollama":
            return f"ollama/{model_name}"
        return model_name

    def _resolve_api_key(self) -> Optional[str]:
        """
        Select the appropriate secret API key for the active provider.

        Returns:
            The plaintext API key string for `self.provider`, or `None`
            when the provider requires no key.
        """
        if self.provider == "openai" and settings.openai_api_key is not None:
            return settings.openai_api_key.get_secret_value()
        if self.provider == "groq" and settings.groq_api_key is not None:
            return settings.groq_api_key.get_secret_value()
        return None

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(
            multiplier=settings.backoff_multiplier_seconds, min=1, max=20
        ),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, RuntimeError)),
    )
    async def _generate_queries(self, claim_text: str) -> List[str]:
        """
        Generate 1-2 optimized search queries for a single claim.

        Args:
            claim_text: The atomic claim text to generate queries for.

        Returns:
            A list of 1-2 search query strings.

        Raises:
            RuntimeError: Wraps any exception raised during the structured
                completion call so tenacity's retry predicate can catch it
                uniformly.
        """
        try:
            result = await self._async_client.chat.completions.create(
                model=self._resolved_model,
                response_model=_SearchQueries,
                max_retries=self.max_retries,
                temperature=0.0,
                timeout=self.request_timeout,
                api_key=self._resolve_api_key(),
                api_base=settings.ollama_base_url if self.provider == "ollama" else None,
                messages=[
                    {"role": "system", "content": _QUERY_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": claim_text},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query generation failed for claim %r: %s", claim_text, exc)
            raise RuntimeError(f"Query generation failed: {exc}") from exc

        if not isinstance(result, _SearchQueries):
            raise RuntimeError(f"Unexpected query generation result type: {type(result)!r}")

        return result.queries

    async def _retrieve_evidence(self, queries: List[str]) -> List[str]:
        """
        Execute all queries for a claim against the search backend and
        aggregate the results.

        Since `EvidenceSearcher.search_claims` is a blocking/synchronous
        call, each query is dispatched to a worker thread to avoid
        blocking the event loop, and all queries for the claim are run
        concurrently.

        Args:
            queries: The list of search queries to execute.

        Returns:
            A list of raw context strings, one per query, with empty
            results filtered out.

        Raises:
            InvestigationError: If evidence retrieval fails for every
                query.
        """
        async def _run_single_query(query: str) -> Optional[str]:
            try:
                return await asyncio.to_thread(
                    self.searcher.search_claims, query, self.max_results_per_query
                )
            except SearchProviderError as exc:
                logger.warning("Evidence retrieval failed for query %r: %s", query, exc)
                return None

        results = await asyncio.gather(*(_run_single_query(q) for q in queries))
        non_empty_results = [r for r in results if r]

        if not non_empty_results and queries:
            logger.info(
                "All %d queries returned no evidence: %s", len(queries), queries
            )

        return non_empty_results

    async def _investigate_single(self, claim: AtomicClaim) -> ClaimEvidence:
        """
        Generate queries and retrieve evidence for a single atomic claim.

        Args:
            claim: The `AtomicClaim` to investigate.

        Returns:
            A `ClaimEvidence` object populated with the generated queries
            and retrieved context snippets. If query generation fails
            after all retries, falls back to using the raw claim text as
            the sole query rather than dropping the claim entirely.

        Raises:
            InvestigationError: If evidence retrieval fails catastrophically
                (e.g., the search provider itself is unreachable for every
                query, raising rather than degrading).
        """
        try:
            queries = await self._generate_queries(claim.text)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, don't drop the claim
            logger.warning(
                "Falling back to raw claim text as query for claim_id=%s due to: %s",
                claim.claim_id,
                exc,
            )
            queries = [claim.text]

        try:
            context_snippets = await self._retrieve_evidence(queries)
        except Exception as exc:  # noqa: BLE001
            raise InvestigationError(
                f"Evidence retrieval catastrophically failed for claim_id={claim.claim_id}: {exc}"
            ) from exc

        return ClaimEvidence(
            claim_id=claim.claim_id,
            search_queries=queries,
            retrieved_context=context_snippets,
        )

    async def investigate(self, claims: List[AtomicClaim]) -> List[ClaimEvidence]:
        """
        Concurrently generate queries and retrieve evidence for a batch
        of atomic claims.

        Args:
            claims: The list of `AtomicClaim` objects to investigate.

        Returns:
            A list of `ClaimEvidence` objects, one per input claim,
            preserving input order. Returns an empty list if `claims` is
            empty.
        """
        if not claims:
            return []

        results = await asyncio.gather(
            *(self._investigate_single(claim) for claim in claims),
            return_exceptions=True,
        )

        evidence_list: List[ClaimEvidence] = []
        for claim, result in zip(claims, results):
            if isinstance(result, Exception):
                logger.error(
                    "Investigation failed for claim_id=%s; emitting empty evidence: %s",
                    claim.claim_id,
                    result,
                )
                evidence_list.append(
                    ClaimEvidence(
                        claim_id=claim.claim_id,
                        search_queries=[],
                        retrieved_context=[],
                    )
                )
            else:
                evidence_list.append(result)

        return evidence_list