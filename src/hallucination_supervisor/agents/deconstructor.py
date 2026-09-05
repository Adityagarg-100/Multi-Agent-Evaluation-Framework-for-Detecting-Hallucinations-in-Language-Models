from __future__ import annotations

import logging
import uuid
from typing import List, Optional

import instructor
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.hallucination_supervisor.config import settings
from src.hallucination_supervisor.schemas import AtomicClaim

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise claim-decomposition engine. Decompose the provided text into "
    "independently verifiable, atomic factual claims. Strip conversational filler, "
    "hedging, and opinion. Each claim should express exactly one verifiable fact and "
    "split compound sentences apart. Resolve pronouns and vague references to their "
    "actual referent (e.g. 'he was born in 1879' -> 'Einstein was born in 1879'). "
    "For every claim, return the verbatim source_sentence it came from and a confidence "
    "score in [0.0, 1.0] reflecting how clearly it's a factual claim rather than "
    "opinion or filler. If there are no verifiable claims, return an empty list."
)


class ExtractionError(Exception):
    """Raised when claim extraction fails after all retries are exhausted."""


class _ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    source_sentence: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("text", "source_sentence")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class _ExtractedClaimList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: List[_ExtractedClaim] = Field(default_factory=list)


class ClaimExtractor:
    """Breaks a draft down into atomic, independently verifiable claims."""

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None) -> None:
        try:
            import litellm
        except ImportError as exc:
            raise ExtractionError("litellm is required. pip install litellm") from exc

        self.provider = provider or settings.default_llm_provider
        self.model_name = model_name or settings.default_model_name
        self.temperature = settings.claim_extraction_temperature
        self.confidence_threshold = settings.claim_extraction_confidence
        self.max_retries = settings.max_retries
        self.request_timeout = settings.request_timeout_seconds

        self._resolved_model = self._resolve_model_string(self.provider, self.model_name)

        # JSON mode instead of tool-calling
        # reliable at emitting raw JSON than following a nested function-call schema.
        try:
            self._client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)
        except Exception as exc:
            raise ExtractionError(f"Failed to init instructor client: {exc}") from exc

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

    def _call_llm(self, draft: str) -> _ExtractedClaimList:
        try:
            result = self._client.chat.completions.create(
                model=self._resolved_model,
                response_model=_ExtractedClaimList,
                max_retries=self.max_retries,
                temperature=self.temperature,
                timeout=self.request_timeout,
                api_key=self._resolve_api_key(),
                api_base=settings.ollama_base_url if self.provider == "ollama" else None,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": draft},
                ],
            )
        except Exception as exc:
            logger.warning("Claim extraction call failed: %s", exc)
            raise RuntimeError(f"Extraction call failed: {exc}") from exc

        return result

    def extract_claims(self, draft: str) -> List[AtomicClaim]:
        """Pulls atomic claims out of a draft, filtering out low-confidence noise."""
        cleaned = draft.strip()
        if not cleaned:
            raise ExtractionError("extract_claims got an empty draft.")

        try:
            extracted = self._call_llm(cleaned)
        except Exception as exc:
            raise ExtractionError(
                f"Claim extraction failed after {self.max_retries} attempts: {exc}"
            ) from exc

        claims: List[AtomicClaim] = []
        for item in extracted.claims:
            if item.confidence < self.confidence_threshold:
                continue
            claims.append(
                AtomicClaim(
                    claim_id=str(uuid.uuid4()),
                    text=item.text,
                    source_sentence=item.source_sentence,
                )
            )

        return claims