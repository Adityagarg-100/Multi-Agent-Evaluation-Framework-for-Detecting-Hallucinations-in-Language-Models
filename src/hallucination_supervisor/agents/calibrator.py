"""
Report Calibration agent for the Multi-Agent LLM Hallucination
Supervisor.

Aggregates per-claim `ClaimVerdict` objects into a single, quantified
`VerificationReport`, including exact-match span-mapping of contradicted
claims back onto the original draft for UI highlighting.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from ..config import settings
from ..schemas import AtomicClaim, ClaimVerdict, VerificationReport

logger = logging.getLogger(__name__)


class CalibrationError(Exception):
    """
    Raised when the Report Calibrator receives structurally inconsistent
    input (e.g., a verdict referencing a claim_id absent from the supplied
    claims list) that would make scoring or span-mapping unreliable.
    """


class ReportCalibrator:
    """
    Aggregates atomic claims and their verdicts into a final
    `VerificationReport`.

    Scoring algorithm:
        hallucination_percentage =
            100 * (n_contradicted + 0.5 * n_unverifiable) / total_claims

    where `n_contradicted` and `n_unverifiable` only count verdicts whose
    `confidence_score` meets or exceeds
    `settings.hallucination_confidence_threshold`, to avoid low-confidence
    negative verdicts inflating the reported hallucination rate.

    Span-mapping: for every `CONTRADICTED` verdict (regardless of the
    confidence-threshold gate, since even a low-confidence contradiction
    is worth surfacing visually) whose confidence meets the threshold,
    the corresponding claim's verbatim `source_sentence` is exact-matched
    against `original_draft` and, if found, added to
    `hallucinated_spans`.

    Attributes:
        hallucination_confidence_threshold: Minimum confidence for a
            negative verdict to count toward scoring/spans, sourced from
            `settings.hallucination_confidence_threshold`.
    """

    def __init__(self) -> None:
        """Initialize the calibrator with thresholds sourced from `settings`."""
        self.hallucination_confidence_threshold: float = (
            settings.hallucination_confidence_threshold
        )

    @staticmethod
    def _index_claims_by_id(claims: List[AtomicClaim]) -> Dict[str, AtomicClaim]:
        """
        Build a lookup table mapping `claim_id` to its `AtomicClaim`.

        Args:
            claims: The full list of atomic claims for this draft.

        Returns:
            A dict mapping `claim_id` to the corresponding `AtomicClaim`.

        Raises:
            CalibrationError: If two claims share the same `claim_id`
                (which would indicate an upstream extraction bug).
        """
        index: Dict[str, AtomicClaim] = {}
        for claim in claims:
            if claim.claim_id in index:
                raise CalibrationError(
                    f"Duplicate claim_id detected in claims list: {claim.claim_id!r}"
                )
            index[claim.claim_id] = claim
        return index

    def generate_report(
        self,
        original_draft: str,
        claims: List[AtomicClaim],
        verdicts: List[ClaimVerdict],
    ) -> VerificationReport:
        """
        Aggregate claims and verdicts into a final `VerificationReport`.

        Note: span-mapping requires access to each claim's verbatim
        `source_sentence`, which is not present on `ClaimVerdict` alone
        (by design — see `schemas.py`, where `ClaimVerdict` intentionally
        carries no reference back to the source text). This method
        therefore requires the original `claims` list alongside
        `verdicts`, rather than `verdicts` in isolation.

        Args:
            original_draft: The full, unmodified original draft text that
                was analyzed.
            claims: The complete list of `AtomicClaim` objects extracted
                from `original_draft` (used for span-mapping).
            verdicts: The list of `ClaimVerdict` objects produced by the
                Judge agent, one per claim in `claims`.

        Returns:
            A fully populated, schema-validated `VerificationReport`.

        Raises:
            CalibrationError: If `verdicts` is empty while `original_draft`
                is non-empty (nothing to score), if any verdict's
                `claim_id` does not correspond to an entry in `claims`, or
                if the claims list contains duplicate `claim_id` values.
        """
        cleaned_draft = original_draft.strip()
        if not cleaned_draft:
            raise CalibrationError("generate_report called with an empty original_draft.")

        claims_by_id = self._index_claims_by_id(claims)

        total_claims = len(verdicts)
        if total_claims == 0:
            logger.info("No verdicts supplied; emitting a zero-claim VerificationReport.")
            return VerificationReport(
                original_draft=cleaned_draft,
                total_claims=0,
                supported_ratio=0.0,
                hallucination_percentage=0.0,
                hallucinated_spans=[],
            )

        n_supported = 0
        n_contradicted_counted = 0
        n_unverifiable_counted = 0
        hallucinated_spans: List[str] = []
        seen_spans: set[str] = set()

        for verdict in verdicts:
            claim = claims_by_id.get(verdict.claim_id)
            if claim is None:
                raise CalibrationError(
                    f"ClaimVerdict.claim_id={verdict.claim_id!r} does not correspond "
                    "to any AtomicClaim in the supplied claims list."
                )

            meets_threshold = verdict.confidence_score >= self.hallucination_confidence_threshold

            if verdict.status == "SUPPORTED":
                n_supported += 1
            elif verdict.status == "CONTRADICTED":
                if meets_threshold:
                    n_contradicted_counted += 1
                self._try_add_span(
                    claim=claim,
                    original_draft=cleaned_draft,
                    meets_threshold=meets_threshold,
                    hallucinated_spans=hallucinated_spans,
                    seen_spans=seen_spans,
                )
            elif verdict.status == "UNVERIFIABLE":
                if meets_threshold:
                    n_unverifiable_counted += 1
            else:
                # Defensive: schema's Literal type should make this unreachable.
                raise CalibrationError(f"Unrecognized verdict status: {verdict.status!r}")

        supported_ratio = n_supported / total_claims
        hallucination_percentage = 100.0 * (
            (n_contradicted_counted + 0.5 * n_unverifiable_counted) / total_claims
        )
        # Clamp for floating-point safety against the schema's [0, 100] bound.
        hallucination_percentage = max(0.0, min(100.0, hallucination_percentage))

        logger.info(
            "Calibration complete: total_claims=%d, supported=%d, contradicted(≥thresh)=%d, "
            "unverifiable(≥thresh)=%d, hallucination_percentage=%.2f%%, spans_flagged=%d",
            total_claims,
            n_supported,
            n_contradicted_counted,
            n_unverifiable_counted,
            hallucination_percentage,
            len(hallucinated_spans),
        )

        return VerificationReport(
            original_draft=cleaned_draft,
            total_claims=total_claims,
            supported_ratio=round(supported_ratio, 6),
            hallucination_percentage=round(hallucination_percentage, 4),
            hallucinated_spans=hallucinated_spans,
        )

    @staticmethod
    def _try_add_span(
        claim: AtomicClaim,
        original_draft: str,
        meets_threshold: bool,
        hallucinated_spans: List[str],
        seen_spans: set[str],
    ) -> None:
        """
        Attempt to exact-match a contradicted claim's source sentence
        against the original draft and append it to `hallucinated_spans`
        if found and not already flagged.

        Args:
            claim: The `AtomicClaim` associated with a CONTRADICTED
                verdict.
            original_draft: The full original draft text to search
                within.
            meets_threshold: Whether the verdict's confidence meets the
                calibrator's hallucination confidence threshold. Spans
                are only added when this is True, keeping span-flagging
                consistent with the numeric score.
            hallucinated_spans: The accumulator list to append matched
                spans to (mutated in place).
            seen_spans: A set tracking already-added spans to prevent
                duplicate entries when multiple claims share a source
                sentence (mutated in place).
        """
        if not meets_threshold:
            return

        span = claim.source_sentence.strip()
        if not span:
            return

        if span in seen_spans:
            return

        if span in original_draft:
            hallucinated_spans.append(span)
            seen_spans.add(span)
        else:
            logger.warning(
                "CONTRADICTED claim_id=%s source_sentence could not be exact-matched "
                "against original_draft; skipping span flag. source_sentence=%r",
                claim.claim_id,
                span,
            )