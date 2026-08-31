"""
Zero-Shot NLI Judge agent for the Multi-Agent LLM Hallucination
Supervisor.

Adjudicates each atomic claim strictly against its retrieved evidence,
with no reliance on the model's parametric/outside knowledge, producing a
deterministic `ClaimVerdict`.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import instructor
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import settings
from ..schemas import AtomicClaim, ClaimEvidence, ClaimVerdict

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict logical judge performing Natural Language Inference. "
    "Compare the claim strictly against the provided evidence. Do not use "
    "outside knowledge, prior training data, or assumptions beyond what is "
    "explicitly stated in the evidence. "
    "Output SUPPORTED if the evidence explicitly confirms the claim. "
    "Output CONTRADICTED if the evidence explicitly refutes or conflicts "
    "with the claim. "
    "Output UNVERIFIABLE if the evidence is absent, irrelevant, or "
    "insufficient to confirm or deny the claim either way. "
    "Provide a confidence_score in [0.0, 1.0] reflecting your certainty in "
    "the assigned status, and a rationale that explicitly cites which part "
    "of the evidence (or the absence thereof) led to your verdict."
)

_NO_EVIDENCE_RATIONALE = (
    "Insufficient evidence was retrieved for this claim (fewer than the "
    "configured minimum of {min_snippets} snippet(s)); the claim is "
    "automatically marked UNVERIFIABLE without invoking the judge model, "
    "per the zero-outside-knowledge policy."
)


class JudgmentError(Exception):
    """
    Raised when the NLI Judge agent fails to produce a valid,
    schema-conformant verdict after exhausting all configured retry
    attempts.
    """


class _JudgeOutput(BaseModel):
    """
    Internal, LLM-facing schema for a single NLI judgment.

    Distinct from `schemas.ClaimVerdict` because the LLM must not be
    responsible for populating `claim_id`, which is assigned
    deterministically by the pipeline from the input `AtomicClaim` to
    prevent ID-hallucination/mismatch.

    Attributes:
        status: The categorical NLI verdict.
        confidence_score: The judge's confidence in `status`.
        rationale: Chain-of-thought explanation citing specific evidence.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"] = Field(
        ...,
        description="Categorical NLI verdict for the claim given the evidence.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Judge's confidence in the assigned status.",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Chain-of-thought explanation citing specific evidence.",
    )

    @field_validator("rationale")
    @classmethod
    def _reject_empty_rationale(cls, v: str) -> str:
        """Ensure the rationale is non-empty after whitespace stripping.

        Args:
            v: The candidate rationale string.

        Returns:
            The validated, stripped rationale string.

        Raises:
            ValueError: If the rationale is empty or whitespace-only.
        """
        if not v.strip():
            raise ValueError("Rationale must not be empty or whitespace-only.")
        return v


class NLIVerifier:
    """
    Performs strict, zero-shot Natural Language Inference judgment of a
    single atomic claim against its retrieved evidence.

    Enforces `settings.judge_temperature` (default 0.0) on every call to
    maximize determinism and grading reproducibility.

    Attributes:
        provider: LLM provider identifier used for judgment.
        model_name: Model identifier passed to `litellm`.
        temperature: Sampling temperature, fixed to
            `settings.judge_temperature`.
        min_evidence_snippets: Minimum retrieved snippets required before
            a definitive (non-UNVERIFIABLE) verdict may be rendered.
        max_retries: Maximum retry attempts for transient failures.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the NLI verifier and its instructor-patched LLM client.

        Args:
            provider: Optional override for the LLM provider. Defaults to
                `settings.default_llm_provider`.
            model_name: Optional override for the model identifier.
                Defaults to `settings.default_model_name`.

        Raises:
            JudgmentError: If the `litellm` package is not installed.
        """
        try:
            import litellm  # type: ignore[import-untyped]
        except ImportError as exc:
            raise JudgmentError(
                "The 'litellm' package is required for NLIVerifier. "
                "Install it with `pip install litellm`."
            ) from exc

        self.provider: str = provider or settings.default_llm_provider
        self.model_name: str = model_name or settings.default_model_name
        self.temperature: float = settings.judge_temperature
        self.min_evidence_snippets: int = settings.min_evidence_snippets
        self.max_retries: int = settings.max_retries
        self.request_timeout: float = settings.request_timeout_seconds

        self._resolved_model = self._resolve_model_string(self.provider, self.model_name)

        try:
            self._client = instructor.from_litellm(litellm.completion)
        except Exception as exc:  # noqa: BLE001
            raise JudgmentError(
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

    @staticmethod
    def _format_evidence_block(evidence: ClaimEvidence) -> str:
        """
        Render a `ClaimEvidence` object into a plain-text block suitable
        for inclusion in the judge's user prompt.

        Args:
            evidence: The evidence bundle to format.

        Returns:
            A formatted string enumerating each retrieved context
            snippet, or an explicit "no evidence" marker if none were
            retrieved.
        """
        if not evidence.retrieved_context:
            return "[NO EVIDENCE RETRIEVED]"

        blocks = [
            f"Evidence Snippet {i + 1}:\n{snippet}"
            for i, snippet in enumerate(evidence.retrieved_context)
        ]
        return "\n\n".join(blocks)

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(
            multiplier=settings.backoff_multiplier_seconds, min=1, max=20
        ),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, RuntimeError)),
    )
    def _call_judge_llm(self, claim_text: str, evidence_block: str) -> _JudgeOutput:
        """
        Issue a single structured NLI judgment completion request.

        Args:
            claim_text: The atomic claim text under adjudication.
            evidence_block: The formatted evidence text to judge the claim
                against.

        Returns:
            A validated `_JudgeOutput` instance.

        Raises:
            RuntimeError: Wraps any exception raised by the client so
                tenacity's retry predicate can catch it uniformly.
        """
        user_prompt = (
            f"CLAIM:\n{claim_text}\n\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            "Adjudicate the claim strictly against the evidence above."
        )
        try:
            result = self._client.chat.completions.create(
                model=self._resolved_model,
                response_model=_JudgeOutput,
                max_retries=self.max_retries,
                temperature=self.temperature,
                timeout=self.request_timeout,
                api_key=self._resolve_api_key(),
                api_base=settings.ollama_base_url if self.provider == "ollama" else None,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("NLI judgment call failed: %s", exc)
            raise RuntimeError(f"NLI judgment call failed: {exc}") from exc

        if not isinstance(result, _JudgeOutput):
            raise RuntimeError(f"Unexpected judge result type: {type(result)!r}")

        return result

    def evaluate_claim(self, claim: AtomicClaim, evidence: ClaimEvidence) -> ClaimVerdict:
        """
        Adjudicate a single atomic claim against its retrieved evidence.

        If the number of retrieved evidence snippets falls below
        `self.min_evidence_snippets`, the claim is automatically marked
        UNVERIFIABLE without invoking the LLM, since there is nothing
        substantive for a zero-outside-knowledge judge to reason over.

        Args:
            claim: The `AtomicClaim` to adjudicate. Must have a
                `claim_id` matching `evidence.claim_id`.
            evidence: The `ClaimEvidence` bundle retrieved for this claim.

        Returns:
            A `ClaimVerdict` with `claim_id` set from `claim.claim_id`
            (never from LLM output, to prevent ID mismatches).

        Raises:
            JudgmentError: If `claim.claim_id` does not match
                `evidence.claim_id`, or if the LLM fails to produce a
                valid verdict after exhausting all retry attempts.
        """
        if claim.claim_id != evidence.claim_id:
            raise JudgmentError(
                f"claim_id mismatch: claim.claim_id={claim.claim_id!r} "
                f"!= evidence.claim_id={evidence.claim_id!r}"
            )

        if len(evidence.retrieved_context) < self.min_evidence_snippets:
            logger.info(
                "claim_id=%s has insufficient evidence (%d < %d); "
                "auto-marking UNVERIFIABLE.",
                claim.claim_id,
                len(evidence.retrieved_context),
                self.min_evidence_snippets,
            )
            return ClaimVerdict(
                claim_id=claim.claim_id,
                status="UNVERIFIABLE",
                confidence_score=1.0,
                rationale=_NO_EVIDENCE_RATIONALE.format(
                    min_snippets=self.min_evidence_snippets
                ),
            )

        evidence_block = self._format_evidence_block(evidence)

        try:
            judge_output = self._call_judge_llm(claim.text, evidence_block)
        except Exception as exc:  # noqa: BLE001 - final catch after retries exhausted
            raise JudgmentError(
                f"NLI judgment failed after {self.max_retries} attempts for "
                f"claim_id={claim.claim_id}: {exc}"
            ) from exc

        return ClaimVerdict(
            claim_id=claim.claim_id,
            status=judge_output.status,
            confidence_score=judge_output.confidence_score,
            rationale=judge_output.rationale,
        )