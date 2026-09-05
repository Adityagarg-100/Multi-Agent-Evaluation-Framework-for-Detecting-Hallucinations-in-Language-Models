from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import instructor
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.hallucination_supervisor.config import settings
from src.hallucination_supervisor.schemas import AtomicClaim, ClaimEvidence
from src.hallucination_supervisor.tools.search import EvidenceSearcher, SearchProviderError

logger = logging.getLogger(__name__)

QUERY_PROMPT = (
    "You are a search-query optimizer for a fact-checking system. Given a factual claim, "
    "produce 1-2 concise, high-precision search queries that would surface evidence to "
    "confirm or refute it. Use concrete entities, names, dates, or numbers from the claim. "
    "No quotes, no search operators."
)


class InvestigationError(Exception):
    """Raised when evidence retrieval fails for a claim after retries."""


class _SearchQueries(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queries: List[str] = Field(..., min_length=1, max_length=2)

    @field_validator("queries")
    @classmethod
    def _not_blank(cls, v: List[str]) -> List[str]:
        for q in v:
            if not q.strip():
                raise ValueError("query must not be blank")
        return v


class EvidenceInvestigator:
    """
    Generates search queries per claim and fetches evidence for all of them
    concurrently. Query generation uses a cheap async LLM call; the actual
    search hits EvidenceSearcher (which is sync, so it's dispatched to a
    thread to avoid blocking the loop).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        max_results_per_query: int = 3,
        searcher: Optional[EvidenceSearcher] = None,
    ) -> None:
        try:
            import litellm
        except ImportError as exc:
            raise InvestigationError("litellm is required. pip install litellm") from exc

        self.provider = provider or settings.default_llm_provider
        self.model_name = model_name or settings.default_model_name
        self.max_retries = settings.max_retries
        self.request_timeout = settings.request_timeout_seconds
        self.max_results_per_query = max(1, max_results_per_query)

        self._resolved_model = self._resolve_model_string(self.provider, self.model_name)

        try:
            self._client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)
        except Exception as exc:
            raise InvestigationError(f"Failed to init instructor client: {exc}") from exc

        try:
            self.searcher = searcher or EvidenceSearcher()
        except SearchProviderError as exc:
            raise InvestigationError(f"Failed to init EvidenceSearcher: {exc}") from exc

    @staticmethod
    def _resolve_model_string(provider: str, model_name: str) -> str:
        if provider == "openai":
            return model_name
        if provider == "anthropic":
            return f"anthropic/{model_name}"
        if provider == "groq":
            return f"groq/{model_name}"
        if provider == "ollama":
            return f"ollama/{model_name}"
        return model_name

    def _resolve_api_key(self) -> Optional[str]:
        if self.provider == "openai" and settings.openai_api_key is not None:
            return settings.openai_api_key.get_secret_value()
        if self.provider == "anthropic" and settings.anthropic_api_key is not None:
            return settings.anthropic_api_key.get_secret_value()
        if self.provider == "groq" and settings.groq_api_key is not None:
            return settings.groq_api_key.get_secret_value()
        return None

    async def _generate_queries(self, claim_text: str) -> List[str]:
        try:
            result = await self._client.chat.completions.create(
                model=self._resolved_model,
                response_model=_SearchQueries,
                max_retries=self.max_retries,
                temperature=0.0,
                timeout=self.request_timeout,
                api_key=self._resolve_api_key(),
                api_base=settings.ollama_base_url if self.provider == "ollama" else None,
                messages=[
                    {"role": "system", "content": QUERY_PROMPT},
                    {"role": "user", "content": claim_text},
                ],
            )
        except Exception as exc:
            logger.warning("Query generation failed for claim: %s", exc)
            raise RuntimeError(f"Query generation failed: {exc}") from exc

        return result.queries

    async def _retrieve_evidence(self, queries: List[str]) -> List[str]:
        async def run_one(q: str) -> Optional[str]:
            try:
                return await asyncio.to_thread(self.searcher.search_claims, q, self.max_results_per_query)
            except SearchProviderError as exc:
                logger.warning("Search failed for query %r: %s", q, exc)
                return None

        results = await asyncio.gather(*(run_one(q) for q in queries))
        return [r for r in results if r]

    async def _investigate_one(self, claim: AtomicClaim) -> ClaimEvidence:
        try:
            queries = await self._generate_queries(claim.text)
        except Exception:
            # Query gen failed -- fall back to the raw claim text rather than
            # dropping the claim entirely.
            queries = [claim.text]

        context = await self._retrieve_evidence(queries)

        return ClaimEvidence(claim_id=claim.claim_id, search_queries=queries, retrieved_context=context)

    async def investigate(self, claims: List[AtomicClaim]) -> List[ClaimEvidence]:
        """Fetches evidence for a batch of claims concurrently."""
        if not claims:
            return []

        results = await asyncio.gather(
            *(self._investigate_one(c) for c in claims),
            return_exceptions=True,
        )

        evidence: List[ClaimEvidence] = []
        for claim, result in zip(claims, results):
            if isinstance(result, Exception):
                logger.error("Investigation failed for claim %s: %s", claim.claim_id, result)
                evidence.append(ClaimEvidence(claim_id=claim.claim_id, search_queries=[], retrieved_context=[]))
            else:
                evidence.append(result)

        return evidence