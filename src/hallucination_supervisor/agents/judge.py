from __future__ import annotations

import logging
from typing import Literal, Optional

import instructor
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.hallucination_supervisor.config import settings
from src.hallucination_supervisor.schemas import AtomicClaim, ClaimEvidence, ClaimVerdict

logger = logging.getLogger(__name__)

JUDGE_PROMPT = (
    "You are a strict logical judge doing natural language inference. Compare the claim "
    "only against the provided evidence, no outside knowledge, no assumptions. "
    "SUPPORTED if the evidence explicitly confirms it. CONTRADICTED if the evidence "
    "explicitly refutes it. UNVERIFIABLE if the evidence is missing, irrelevant, or not "
    "enough to decide either way. Give a confidence score in [0.0, 1.0] and a rationale "
    "that points to the specific part of the evidence (or lack of it) behind your call."
)

NO_EVIDENCE_RATIONALE = (
    "Fewer than {min_snippets} evidence snippet(s) were retrieved for this claim, so it's "
    "auto-marked UNVERIFIABLE without calling the judge model."
)


class JudgmentError(Exception):
    """Raised when the judge fails to produce a verdict after all retries."""


class _JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=1)

    @field_validator("rationale")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rationale must not be empty")
        return v


class NLIVerifier:
    """Judges a claim against its evidence and returns a verdict. Temperature is pinned low for consistency."""

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None) -> None:
        try:
            import litellm
        except ImportError as exc:
            raise JudgmentError("litellm is required. pip install litellm") from exc

        self.provider = provider or settings.default_llm_provider
        self.model_name = model_name or settings.default_model_name
        self.temperature = settings.judge_temperature
        self.min_evidence_snippets = settings.min_evidence_snippets
        self.max_retries = settings.max_retries
        self.request_timeout = settings.request_timeout_seconds

        self._resolved_model = self._resolve_model_string(self.provider, self.model_name)

        try:
            self._client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)
        except Exception as exc:
            raise JudgmentError(f"Failed to init instructor client: {exc}") from exc

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

    @staticmethod
    def _format_evidence(evidence: ClaimEvidence) -> str:
        if not evidence.retrieved_context:
            return "[NO EVIDENCE RETRIEVED]"
        return "\n\n".join(
            f"Evidence Snippet {i + 1}:\n{snippet}" for i, snippet in enumerate(evidence.retrieved_context)
        )

    def _call_llm(self, claim_text: str, evidence_block: str) -> _JudgeOutput:
        prompt = f"CLAIM:\n{claim_text}\n\nEVIDENCE:\n{evidence_block}\n\nAdjudicate the claim against the evidence above."
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
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            logger.warning("Judgment call failed: %s", exc)
            raise RuntimeError(f"Judgment call failed: {exc}") from exc

        return result

    def evaluate_claim(self, claim: AtomicClaim, evidence: ClaimEvidence) -> ClaimVerdict:
        """Adjudicates one claim against its evidence and returns a verdict."""
        if claim.claim_id != evidence.claim_id:
            raise JudgmentError(
                f"claim_id mismatch: {claim.claim_id!r} vs {evidence.claim_id!r}"
            )

        if len(evidence.retrieved_context) < self.min_evidence_snippets:
            return ClaimVerdict(
                claim_id=claim.claim_id,
                status="UNVERIFIABLE",
                confidence_score=1.0,
                rationale=NO_EVIDENCE_RATIONALE.format(min_snippets=self.min_evidence_snippets),
            )

        try:
            output = self._call_llm(claim.text, self._format_evidence(evidence))
        except Exception as exc:
            raise JudgmentError(
                f"Judgment failed after {self.max_retries} attempts for claim {claim.claim_id}: {exc}"
            ) from exc

        return ClaimVerdict(
            claim_id=claim.claim_id,
            status=output.status,
            confidence_score=output.confidence_score,
            rationale=output.rationale,
        )