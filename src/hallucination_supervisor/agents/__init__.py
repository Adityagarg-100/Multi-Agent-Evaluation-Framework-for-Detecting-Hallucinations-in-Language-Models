"""
Public interface for the `agents` subpackage.

Exposes the five pipeline agents, in their chronological execution order,
along with their corresponding custom exception types so callers can
catch stage-specific failures without reaching into submodules directly.
"""

from ..agents.base_generator import BaseDraftGenerator, DraftGenerationError
from ..agents.deconstructor import ClaimExtractor, ExtractionError
from ..agents.investigator import EvidenceInvestigator, InvestigationError
from ..agents.judge import JudgmentError, NLIVerifier
from ..agents.calibrator import CalibrationError, ReportCalibrator

__all__ = [
    "BaseDraftGenerator",
    "DraftGenerationError",
    "ClaimExtractor",
    "ExtractionError",
    "EvidenceInvestigator",
    "InvestigationError",
    "NLIVerifier",
    "JudgmentError",
    "ReportCalibrator",
    "CalibrationError",
]