"""
Configuration module for the Multi-Agent LLM Hallucination Supervisor.

Defines strongly-typed, environment-driven settings using pydantic-settings.
All agents and tools MUST source their configuration exclusively from an
instance of `Settings` (see the `settings` singleton at the bottom of this
file) to guarantee a single source of truth across the pipeline.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized, environment-backed configuration for the hallucination
    supervisor pipeline.

    Values are loaded from environment variables and/or a local `.env` file
    (if present). All API keys are stored as `SecretStr` to prevent
    accidental leakage via logs, `repr()`, or exception tracebacks.

    Attributes:
        openai_api_key: API key for OpenAI-hosted models (used by the
            Claim Extraction and Judge agents by default).
        groq_api_key: API key for Groq-hosted inference (used for
            low-latency claim decomposition or fallback inference).
        ollama_base_url: Base URL of a local/self-hosted Ollama server,
            used as an offline fallback LLM provider.
        search_provider: Which web search backend the Evidence Retrieval
            agent should use ('tavily', 'serpapi', or 'duckduckgo').
        tavily_api_key: API key for the Tavily search API.
        serpapi_api_key: API key for the SerpAPI search API.
        max_retries: Maximum number of retry attempts for any LLM or
            external API call before raising to the caller.
        request_timeout_seconds: Hard timeout, in seconds, applied to every
            outbound LLM or search API call.
        backoff_multiplier_seconds: Base multiplier (seconds) for
            exponential backoff between retries.
        judge_temperature: Sampling temperature enforced on the Judge
            agent's LLM calls. Fixed near 0.0 to maximize determinism and
            reduce verdict variance.
        claim_extraction_temperature: Sampling temperature for the Claim
            Extraction agent. Kept low but non-zero to allow for natural
            claim segmentation without excessive rigidity.
        claim_extraction_confidence: Minimum confidence (0.0-1.0) an
            extracted atomic claim must meet to be forwarded to the
            Evidence Retrieval agent. Claims below this threshold are
            discarded as noise (e.g., stylistic filler, not factual
            assertions).
        min_evidence_snippets: Minimum number of retrieved evidence
            snippets required before the Judge agent may render a
            'SUPPORTED' or 'CONTRADICTED' verdict; otherwise the claim is
            forced to 'UNVERIFIABLE'.
        hallucination_confidence_threshold: Minimum confidence score
            required for a 'CONTRADICTED' or 'UNVERIFIABLE' verdict to be
            counted as a true hallucination in the final report (guards
            against low-confidence false positives inflating the
            hallucination_percentage).
        default_llm_provider: Which provider backs the primary pipeline
            LLM calls by default.
        default_model_name: Model identifier passed to the default
            provider's client.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

     # --- API Keys -----------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama server.",
    )
    search_provider: Literal["duckduckgo"] = Field(
        default="duckduckgo",
        description="Web search backend.",
    )

    # --- Resilience / Retry Parameters ---------------------------------
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for any LLM or external API call.",
    )
    request_timeout_seconds: float = Field(
        default=180.0,
        gt=0.0,
        description="Hard timeout (seconds) applied to every outbound API call.",
    )
    backoff_multiplier_seconds: float = Field(
        default=1.5,
        gt=0.0,
        description="Base multiplier (seconds) for exponential backoff between retries.",
    )

    # --- Agent Behavior Thresholds --------------------------------------
    judge_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the Judge agent (kept near 0 for determinism).",
    )
    claim_extraction_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the Claim Extraction agent.",
    )
    claim_extraction_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for an extracted claim to be forwarded downstream.",
    )
    min_evidence_snippets: int = Field(
        default=1,
        ge=0,
        description="Minimum retrieved evidence snippets required before a definitive verdict.",
    )
    hallucination_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for a negative verdict to count toward the hallucination rate.",
    )

    # --- Model Selection --------------------------------------------------
    default_llm_provider: Literal["ollama"] = Field(
        default="ollama",
        description="Default LLM provider.",
    )
    default_model_name: str = Field(
        default="qwen2.5:7b",
        description="Model identifier passed to the default provider's client.",
    )

    @field_validator("ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached, process-wide singleton instance of `Settings`.

    Using `lru_cache` ensures the `.env` file and environment variables are
    parsed exactly once per process, avoiding redundant I/O and guaranteeing
    all agents observe identical configuration values.

    Returns:
        The singleton `Settings` instance.
    """
    return Settings()


settings: Settings = get_settings()