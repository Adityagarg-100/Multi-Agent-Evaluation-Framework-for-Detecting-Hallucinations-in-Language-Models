from __future__ import annotations

import logging
from typing import Dict, List

from src.hallucination_supervisor.config import settings
from src.hallucination_supervisor.schemas import AtomicClaim, ClaimVerdict, VerificationReport

logger = logging.getLogger(__name__)


class CalibrationError(Exception):
    """Raised when claims/verdicts don't line up cleanly enough to score."""


class ReportCalibrator:
    """
    Turns a batch of per-claim verdicts into one aggregated report.

    Scoring:
        hallucination % = 100 * (contradicted + 0.5 * unverifiable) / total_claims

    Only verdicts with confidence >= hallucination_confidence_threshold count
    toward the score.
    """

    def __init__(self) -> None:
        self.hallucination_confidence_threshold = settings.hallucination_confidence_threshold

    @staticmethod
    def _index_claims_by_id(claims: List[AtomicClaim]) -> Dict[str, AtomicClaim]:
        index: Dict[str, AtomicClaim] = {}
        for claim in claims:
            if claim.claim_id in index:
                raise CalibrationError(f"Duplicate claim_id: {claim.claim_id!r}")
            index[claim.claim_id] = claim
        return index

    def generate_report(
        self,
        original_draft: str,
        claims: List[AtomicClaim],
        verdicts: List[ClaimVerdict],
    ) -> VerificationReport:
        draft = original_draft.strip()
        if not draft:
            raise CalibrationError("generate_report got an empty original_draft.")

        claims_by_id = self._index_claims_by_id(claims)
        total = len(verdicts)

        if total == 0:
            return VerificationReport(
                original_draft=draft,
                total_claims=0,
                supported_ratio=0.0,
                hallucination_percentage=0.0,
                hallucinated_spans=[],
            )

        supported = 0
        contradicted = 0
        unverifiable = 0
        spans: List[str] = []
        seen_spans: set[str] = set()

        for verdict in verdicts:
            claim = claims_by_id.get(verdict.claim_id)
            if claim is None:
                raise CalibrationError(
                    f"Verdict references claim_id {verdict.claim_id!r} not present in claims list."
                )

            meets_threshold = verdict.confidence_score >= self.hallucination_confidence_threshold

            if verdict.status == "SUPPORTED":
                supported += 1
            elif verdict.status == "CONTRADICTED":
                if meets_threshold:
                    contradicted += 1
                self._flag_span(claim, draft, meets_threshold, spans, seen_spans)
            elif verdict.status == "UNVERIFIABLE":
                if meets_threshold:
                    unverifiable += 1
            else:
                raise CalibrationError(f"Unknown verdict status: {verdict.status!r}")

        supported_ratio = supported / total
        hallucination_pct = 100.0 * ((contradicted + 0.5 * unverifiable) / total)
        hallucination_pct = max(0.0, min(100.0, hallucination_pct))

        logger.info(
            "Scored %d claims -> supported=%d contradicted=%d unverifiable=%d, hallucination=%.2f%%",
            total, supported, contradicted, unverifiable, hallucination_pct,
        )

        return VerificationReport(
            original_draft=draft,
            total_claims=total,
            supported_ratio=round(supported_ratio, 6),
            hallucination_percentage=round(hallucination_pct, 4),
            hallucinated_spans=spans,
        )

    @staticmethod
    def _flag_span(
        claim: AtomicClaim,
        draft: str,
        meets_threshold: bool,
        spans: List[str],
        seen_spans: set,
    ) -> None:
        # Only flag spans for verdicts confident enough to count toward the score,
        # so the highlighted text always matches the number we report.
        if not meets_threshold:
            return

        span = claim.source_sentence.strip()
        if not span or span in seen_spans:
            return

        if span in draft:
            spans.append(span)
            seen_spans.add(span)
        else:
            # Happens when the model paraphrases instead of quoting -- can't
            # highlight text that isn't actually in the draft verbatim.
            logger.warning("Couldn't find source_sentence in draft for claim %s", claim.claim_id)