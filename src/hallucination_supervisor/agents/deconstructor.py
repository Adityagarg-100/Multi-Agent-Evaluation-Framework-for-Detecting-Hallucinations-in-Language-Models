"""
Claim Deconstruction agent for the Multi-Agent LLM Hallucination
Supervisor.

Responsible for decomposing a raw LLM-generated draft into a list of
independently verifiable `AtomicClaim` objects, using structured-output
enforcement (via `instructor` patched onto `litellm`) so that malformed or
non-JSON model responses are automatically retried/repaired rather than
crashing the pipeline.
"""

from __future__ import annotations

import logging
import uuid
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
from ..schemas import AtomicClaim

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise claim-decomposition engine. Decompose the provided "
    "text into independently verifiable, atomic factual claims. Strip all "
    "conversational filler, hedging, greetings, and opinion statements. "
    "Each claim must express exactly one verifiable fact — split compound "
    "sentences into separate claims. Maintain context: resolve all pronouns "
    "and ambiguous references (e.g., 'he', 'it', 'the company') to their "
    "actual referent based on the surrounding text (e.g., rewrite 'He was "
    "born in 1879' as 'Einstein was born in 1879'). For every claim, also "
    "return the verbatim source_sentence from the original text it was "
    "derived from, and a confidence score in [0.0, 1.0] reflecting how "
    "clearly the text asserts this as a factual claim (as opposed to "
    "opinion, speculation, or rhetorical filler). If the text contains no "
    "verifiable factual claims, return an empty list."
)


class ExtractionError(Exception):
    """
    Raised when the Claim Extraction agent fails to produce a
    structurally valid, schema-conformant claim list after exhausting all
    configured retry attempts.
    """


class _ExtractedClaim(BaseModel):
    """
    Internal, LLM-facing schema for a single extracted claim.

    This is distinct from `schemas.AtomicClaim` because the LLM must not
    be responsible for generating `claim_id` (which is assigned
    deterministically by the pipeline) and because extraction requires an
    additional `confidence` field used purely for filtering, which is not
    part of the downstream `AtomicClaim` contract.

    Attributes:
        text: Self-contained, atomic factual statement with all pronouns
            and references resolved.
        source_sentence: Verbatim sentence/span from the original text
            this claim was derived from.
        confidence: Model's confidence, in [0.0, 1.0], that this is a
            genuine, independently verifiable factual claim.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        description="Self-contained atomic factual statement, references resolved.",
    )
    source_sentence: str = Field(
        ...,
        min_length=1,
        description="Verbatim source sentence/span from the original text.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence that this is a genuine, verifiable factual claim.",
    )

    @field_validator("text", "source_sentence")
    @classmethod
    def _reject_empty(cls, v: str) -> str:
        """Ensure text fields are non-empty after stripping whitespace.

        Args:
            v: Candidate string value.

        Returns:
            The validated, stripped string.

        Raises:
            ValueError: If the string is empty or whitespace-only.
        """
        if not v.strip():
            raise ValueError("Field must not be empty or whitespace-only.")
        return v


class _ExtractedClaimList(BaseModel):
    """
    Wrapper schema enforcing the top-level structured-output shape
    returned by the LLM: a bounded list of `_ExtractedClaim` objects.

    Attributes:
        claims: The list of extracted claims. May be empty if the source
            text contains no verifiable factual assertions.
    """

    model_config = ConfigDict(extra="forbid")

    claims: List[_ExtractedClaim] = Field(
        default_factory=list,
        description="List of atomic claims extracted from the input text.",
    )


class ClaimExtractor:
    """
    Decomposes raw draft text into a list of `AtomicClaim` objects using a
    structured-output-enforced LLM call.

    Uses `instructor` patched onto `litellm.completion` to guarantee the
    model's response conforms to `_ExtractedClaimList`, with automatic
    reprompting/repair on the client side for malformed JSON, and an
    outer `tenacity` retry loop for transient network/provider failures.

    Attributes:
        provider: LLM provider identifier used for extraction.
        model_name: Model identifier passed to `litellm`.
        temperature: Sampling temperature, sourced from
            `settings.claim_extraction_temperature`.
        confidence_threshold: Minimum per-claim confidence required to
            survive filtering, sourced from
            `settings.claim_extraction_confidence`.
        max_retries: Maximum retry attempts for both structured-output
            repair and transient failures.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the claim extractor and its instructor-patched LLM
        client.

        Args:
            provider: Optional override for the LLM provider. Defaults to
                `settings.default_llm_provider`.
            model_name: Optional override for the model identifier.
                Defaults to `settings.default_model_name`.

        Raises:
            ExtractionError: If the `litellm` or `instructor` packages are
                not installed.
        """
        try:
            import litellm  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ExtractionError(
                "The 'litellm' package is required for ClaimExtractor. "
                "Install it with `pip install litellm`."
            ) from exc

        self.provider: str = provider or settings.default_llm_provider
        self.model_name: str = model_name or settings.default_model_name
        self.temperature: float = settings.claim_extraction_temperature
        self.confidence_threshold: float = settings.claim_extraction_confidence
        self.max_retries: int = settings.max_retries
        self.request_timeout: float = settings.request_timeout_seconds

        self._resolved_model = self._resolve_model_string(self.provider, self.model_name)

        try:
            self._client = instructor.from_litellm(litellm.completion)
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(
                f"Failed to initialize instructor-patched litellm client: {exc}"
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
            A model string formatted as `litellm` expects (e.g.,
            'ollama/llama3', 'groq/llama3-70b-8192'). OpenAI model names
            pass through unmodified.
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
    def _call_llm_structured(self, draft: str) -> _ExtractedClaimList:
        """
        Issue a single structured-output completion request that returns
        a validated `_ExtractedClaimList`.

        `instructor` handles JSON-schema enforcement and will internally
        reprompt the model up to `max_retries` times if the raw response
        fails Pydantic validation before this method's caller ever sees a
        malformed object.

        Args:
            draft: The raw draft text to decompose into atomic claims.

        Returns:
            A validated `_ExtractedClaimList` instance.

        Raises:
            RuntimeError: Wraps any exception raised by the client
                (network failure, rate limit, exhausted instructor
                repair attempts) so the outer tenacity retry can catch it
                uniformly.
        """
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
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": draft},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - normalize all client/provider errors
            logger.warning(
                "Structured claim extraction call failed (provider=%s, model=%s): %s",
                self.provider,
                self._resolved_model,
                exc,
            )
            raise RuntimeError(f"Structured extraction call failed: {exc}") from exc

        if not isinstance(result, _ExtractedClaimList):
            raise RuntimeError(
                f"Instructor returned an unexpected type: {type(result)!r}"
            )

        return result

    def extract_claims(self, draft: str) -> List[AtomicClaim]:
        """
        Decompose a draft into a filtered list of `AtomicClaim` objects.

        Claims whose model-reported confidence falls below
        `self.confidence_threshold` are discarded as noise (opinion,
        rhetorical filler, or ambiguous statements) before conversion to
        the downstream `AtomicClaim` contract.

        Args:
            draft: The raw LLM-generated draft text to decompose.

        Returns:
            A list of `AtomicClaim` objects, each with a freshly generated
            `claim_id`. Returns an empty list if the draft contains no
            claims meeting the confidence threshold.

        Raises:
            ExtractionError: If `draft` is empty/whitespace-only, or if
                the LLM fails to produce a valid, schema-conformant claim
                list after exhausting all retry attempts.
        """
        cleaned_draft = draft.strip()
        if not cleaned_draft:
            raise ExtractionError("extract_claims called with an empty draft.")

        try:
            extracted = self._call_llm_structured(cleaned_draft)
        except Exception as exc:  # noqa: BLE001 - final catch after retries exhausted
            raise ExtractionError(
                f"Claim extraction failed after {self.max_retries} attempts "
                f"(provider={self.provider}, model={self._resolved_model}): {exc}"
            ) from exc

        atomic_claims: List[AtomicClaim] = []
        for item in extracted.claims:
            if item.confidence < self.confidence_threshold:
                logger.debug(
                    "Discarding low-confidence claim (confidence=%.3f < %.3f): %r",
                    item.confidence,
                    self.confidence_threshold,
                    item.text,
                )
                continue
            try:
                atomic_claims.append(
                    AtomicClaim(
                        claim_id=str(uuid.uuid4()),
                        text=item.text,
                        source_sentence=item.source_sentence,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - defensive re-validation
                raise ExtractionError(
                    f"Failed to convert extracted claim to AtomicClaim: {exc}"
                ) from exc

        return atomic_claims