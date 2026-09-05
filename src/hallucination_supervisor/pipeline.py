import asyncio
import logging
import sys
import time
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from src.hallucination_supervisor.agents.base_generator import BaseDraftGenerator, DraftGenerationError
from src.hallucination_supervisor.agents.calibrator import CalibrationError, ReportCalibrator
from src.hallucination_supervisor.agents.deconstructor import ClaimExtractor, ExtractionError
from src.hallucination_supervisor.agents.investigator import EvidenceInvestigator, InvestigationError
from src.hallucination_supervisor.agents.judge import JudgmentError, NLIVerifier
from src.hallucination_supervisor.schemas import AtomicClaim, ClaimEvidence, ClaimVerdict, VerificationReport

# avoids a known ResourceWarning quirk with ProactorEventLoop on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when a pipeline stage fails and we can't produce a report."""


class PipelineRunResult(BaseModel):
    """
    Everything a single pipeline run produced: draft, claims, evidence,
    verdicts, and the final report. VerificationReport only carries the
    aggregate numbers, so use this when you need per-claim detail.
    """

    model_config = ConfigDict(extra="forbid")

    draft: str
    claims: List[AtomicClaim]
    evidence: List[ClaimEvidence]
    verdicts: List[ClaimVerdict]
    report: VerificationReport

    def verdict_for(self, claim_id: str) -> Optional[ClaimVerdict]:
        for v in self.verdicts:
            if v.claim_id == claim_id:
                return v
        return None


class SupervisorPipeline:
    """Runs the full chain: generate -> extract claims -> gather evidence -> judge -> score."""

    def __init__(
        self,
        generator: Optional[BaseDraftGenerator] = None,
        extractor: Optional[ClaimExtractor] = None,
        investigator: Optional[EvidenceInvestigator] = None,
        judge: Optional[NLIVerifier] = None,
        calibrator: Optional[ReportCalibrator] = None,
    ) -> None:
        try:
            self.generator = generator or BaseDraftGenerator()
            self.extractor = extractor or ClaimExtractor()
            self.investigator = investigator or EvidenceInvestigator()
            self.judge = judge or NLIVerifier()
            self.calibrator = calibrator or ReportCalibrator()
        except Exception as exc:
            raise PipelineError(f"Failed to set up pipeline agents: {exc}") from exc

        logger.info("Pipeline ready: %s", {
            "generator": type(self.generator).__name__,
            "extractor": type(self.extractor).__name__,
            "investigator": type(self.investigator).__name__,
            "judge": type(self.judge).__name__,
        })

    def _stage_generate_draft(self, prompt: str) -> str:
        logger.info("STAGE 1/5: generating draft")
        start = time.perf_counter()
        try:
            draft = self.generator.generate_draft(prompt)
        except DraftGenerationError as exc:
            raise PipelineError(f"Draft generation failed: {exc}") from exc
        logger.info("STAGE 1/5 done in %.2fs (%d chars)", time.perf_counter() - start, len(draft))
        return draft

    def _stage_extract_claims(self, draft: str) -> List[AtomicClaim]:
        logger.info("STAGE 2/5: extracting claims")
        start = time.perf_counter()
        try:
            claims = self.extractor.extract_claims(draft)
        except ExtractionError as exc:
            raise PipelineError(f"Claim extraction failed: {exc}") from exc
        logger.info("STAGE 2/5 done in %.2fs (%d claims)", time.perf_counter() - start, len(claims))
        return claims

    def _stage_investigate(self, claims: List[AtomicClaim]) -> List[ClaimEvidence]:
        logger.info("STAGE 3/5: gathering evidence for %d claim(s)", len(claims))
        start = time.perf_counter()
        try:
            evidence = asyncio.run(self.investigator.investigate(claims))
        except InvestigationError as exc:
            raise PipelineError(f"Evidence investigation failed: {exc}") from exc
        logger.info("STAGE 3/5 done in %.2fs", time.perf_counter() - start)
        return evidence

    def _stage_judge(self, claims: List[AtomicClaim], evidence: List[ClaimEvidence]) -> List[ClaimVerdict]:
        if len(claims) != len(evidence):
            raise PipelineError(f"claims/evidence length mismatch: {len(claims)} vs {len(evidence)}")

        logger.info("STAGE 4/5: judging %d claim(s)", len(claims))
        start = time.perf_counter()
        verdicts = []
        for claim, ev in zip(claims, evidence):
            try:
                verdicts.append(self.judge.evaluate_claim(claim, ev))
            except JudgmentError as exc:
                raise PipelineError(f"Judgment failed on claim {claim.claim_id}: {exc}") from exc
        logger.info("STAGE 4/5 done in %.2fs", time.perf_counter() - start)
        return verdicts

    def _stage_calibrate(self, draft: str, claims: List[AtomicClaim], verdicts: List[ClaimVerdict]) -> VerificationReport:
        logger.info("STAGE 5/5: scoring report")
        start = time.perf_counter()
        try:
            report = self.calibrator.generate_report(original_draft=draft, claims=claims, verdicts=verdicts)
        except CalibrationError as exc:
            raise PipelineError(f"Calibration failed: {exc}") from exc
        logger.info("STAGE 5/5 done in %.2fs -- hallucination=%.2f%%", time.perf_counter() - start, report.hallucination_percentage)
        return report

    def _run(self, prompt: str) -> PipelineRunResult:
        prompt = prompt.strip()
        if not prompt:
            raise PipelineError("Prompt is empty.")

        draft = self._stage_generate_draft(prompt)
        claims = self._stage_extract_claims(draft)

        if not claims:
            report = self._stage_calibrate(draft, [], [])
            return PipelineRunResult(draft=draft, claims=[], evidence=[], verdicts=[], report=report)

        evidence = self._stage_investigate(claims)
        verdicts = self._stage_judge(claims, evidence)
        report = self._stage_calibrate(draft, claims, verdicts)

        return PipelineRunResult(draft=draft, claims=claims, evidence=evidence, verdicts=verdicts, report=report)

    def run_verification(self, prompt: str) -> VerificationReport:
        """Runs the full pipeline, returns just the aggregated report."""
        return self._run(prompt).report

    def run_verification_detailed(self, prompt: str) -> PipelineRunResult:
        """Runs the full pipeline, returns draft + claims + evidence + verdicts + report."""
        return self._run(prompt)