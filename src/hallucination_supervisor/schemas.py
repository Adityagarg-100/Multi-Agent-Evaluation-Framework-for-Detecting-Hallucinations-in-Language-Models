from __future__ import annotations

import uuid
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AtomicClaim(BaseModel):

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    claim_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier correlating this claim across the pipeline.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Self-contained, atomic factual statement extracted from the draft.",
    )
    source_sentence: str = Field(
        ...,
        min_length=1,
        description="Verbatim source sentence/span from the original draft this claim came from.",
    )

    @field_validator("text", "source_sentence")
    @classmethod
    def _reject_empty_after_strip(cls, v: str) -> str:
        """Ensure text fields are non-empty after whitespace stripping.

        Args:
            v: The candidate string value.

        Returns:
            The validated, stripped string.

        Raises:
            ValueError: If the string is empty or whitespace-only.
        """
        if not v.strip():
            raise ValueError("Field must not be empty or whitespace-only.")
        return v


class ClaimEvidence(BaseModel):
    """
    Evidence gathered from external sources (e.g., web search) in support
    of, or contradiction to, a single `AtomicClaim`.

    Attributes:
        claim_id: Identifier of the `AtomicClaim` this evidence bundle
            corresponds to. Must match an `AtomicClaim.claim_id`.
        search_queries: The list of search queries issued by the Evidence
            Retrieval agent to gather context for this claim.
        retrieved_context: Raw text snippets returned by the search
            provider for the above queries, in the order retrieved.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    claim_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the AtomicClaim this evidence bundle corresponds to.",
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="Search queries issued to gather context for this claim.",
    )
    retrieved_context: List[str] = Field(
        default_factory=list,
        description="Raw text snippets returned by the search provider, in retrieval order.",
    )

    @field_validator("search_queries")
    @classmethod
    def _dedupe_queries(cls, v: List[str]) -> List[str]:
        """Remove duplicate search queries while preserving original order.

        Args:
            v: The raw list of search query strings.

        Returns:
            A de-duplicated list of non-empty search queries.
        """
        seen: set[str] = set()
        deduped: List[str] = []
        for query in v:
            cleaned = query.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)
        return deduped


class ClaimVerdict(BaseModel):
    """
    The Judge agent's final determination on a single `AtomicClaim`, based
    on its associated `ClaimEvidence`.

    Attributes:
        claim_id: Identifier of the `AtomicClaim` being adjudicated. Must
            match an `AtomicClaim.claim_id`.
        status: The verdict category:
            - 'SUPPORTED': Evidence corroborates the claim.
            - 'CONTRADICTED': Evidence directly refutes the claim.
            - 'UNVERIFIABLE': Insufficient or inconclusive evidence.
        confidence_score: The Judge's confidence in `status`, in [0.0, 1.0].
        rationale: A chain-of-thought explanation of how the Judge arrived
            at `status`, citing specific evidence snippets where possible.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    claim_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the AtomicClaim being adjudicated.",
    )
    status: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"] = Field(
        ...,
        description="The Judge agent's categorical verdict for this claim.",
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
        description="Chain-of-thought explanation supporting the verdict.",
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


class VerificationReport(BaseModel):
    """
    The final, aggregated output of the hallucination supervision pipeline
    for a single draft document.

    Attributes:
        original_draft: The full, unmodified original LLM-generated text
            that was analyzed.
        total_claims: Total number of atomic claims extracted from the
            draft.
        supported_ratio: Fraction of claims (0.0-1.0) that received a
            'SUPPORTED' verdict.
        hallucination_percentage: Percentage (0.0-100.0) of claims deemed
            hallucinated, defined as claims with status in
            {'CONTRADICTED', 'UNVERIFIABLE'} whose confidence_score meets
            or exceeds `hallucination_confidence_threshold`.
        hallucinated_spans: List of verbatim substrings from
            `original_draft` corresponding to claims flagged as
            hallucinated, suitable for direct highlighting in a UI.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    original_draft: str = Field(
        ...,
        min_length=1,
        description="Full, unmodified original draft text that was analyzed.",
    )
    total_claims: int = Field(
        ...,
        ge=0,
        description="Total number of atomic claims extracted from the draft.",
    )
    supported_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of claims that received a SUPPORTED verdict.",
    )
    hallucination_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of claims deemed hallucinated per the confidence threshold.",
    )
    hallucinated_spans: List[str] = Field(
        default_factory=list,
        description="Verbatim substrings of original_draft flagged as hallucinated, for UI highlighting.",
    )

    @model_validator(mode="after")
    def _validate_spans_are_substrings(self) -> "VerificationReport":
        """Ensure every flagged hallucinated span actually occurs in the draft.

        Returns:
            The validated `VerificationReport` instance.

        Raises:
            ValueError: If any entry in `hallucinated_spans` is not a
                verbatim substring of `original_draft`.
        """
        for span in self.hallucinated_spans:
            if span not in self.original_draft:
                raise ValueError(
                    f"hallucinated_spans entry is not a substring of original_draft: {span!r}"
                )
        return self

    @model_validator(mode="after")
    def _validate_span_count_bound(self) -> "VerificationReport":
        """Ensure the number of hallucinated spans does not exceed total_claims.

        Returns:
            The validated `VerificationReport` instance.

        Raises:
            ValueError: If `hallucinated_spans` contains more entries than
                `total_claims`, which would be logically inconsistent.
        """
        if len(self.hallucinated_spans) > self.total_claims:
            raise ValueError(
                "hallucinated_spans cannot contain more entries than total_claims."
            )
        return self