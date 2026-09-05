import uuid
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AtomicClaim(BaseModel):
    """A single, self-contained factual statement pulled out of a draft."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1)
    source_sentence: str = Field(..., min_length=1)

    @field_validator("text", "source_sentence")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class ClaimEvidence(BaseModel):
    """Search queries + raw snippets gathered for one claim."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    claim_id: str = Field(..., min_length=1)
    search_queries: List[str] = Field(default_factory=list)
    retrieved_context: List[str] = Field(default_factory=list)

    @field_validator("search_queries")
    @classmethod
    def _dedupe(cls, v: List[str]) -> List[str]:
        seen = set()
        out = []
        for q in v:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out


class ClaimVerdict(BaseModel):
    """The judge's call on a single claim."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    claim_id: str = Field(..., min_length=1)
    status: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=1)

    @field_validator("rationale")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rationale must not be empty")
        return v

    @property
    def hallucination_probability(self) -> float:
        """
        Rough per-claim hallucination probability, derived from status + confidence.
        CONTRADICTED counts fully, UNVERIFIABLE at half weight, SUPPORTED at zero weight.
        ReportCalibrator uses for the aggregate score, just applied per claim.
        """
        if self.status == "CONTRADICTED":
            return self.confidence_score
        if self.status == "UNVERIFIABLE":
            return 0.5 * self.confidence_score
        return 1.0 - self.confidence_score


class VerificationReport(BaseModel):
    """Final aggregated output for one draft: overall score + flagged spans."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    original_draft: str = Field(..., min_length=1)
    total_claims: int = Field(..., ge=0)
    supported_ratio: float = Field(..., ge=0.0, le=1.0)
    hallucination_percentage: float = Field(..., ge=0.0, le=100.0)
    hallucinated_spans: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _spans_must_exist_in_draft(self) -> "VerificationReport":
        for span in self.hallucinated_spans:
            if span not in self.original_draft:
                raise ValueError(f"hallucinated_spans entry not found in original_draft: {span!r}")
        return self

    @model_validator(mode="after")
    def _spans_cant_exceed_claims(self) -> "VerificationReport":
        if len(self.hallucinated_spans) > self.total_claims:
            raise ValueError("more hallucinated_spans than total_claims -- something's off")
        return self