from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config, loaded from .env. Every agent reads from the shared `settings` instance below."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API keys
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    groq_api_key: Optional[SecretStr] = None
    ollama_base_url: str = "http://localhost:11434"

    # search is locked to duckduckgo
    search_provider: Literal["duckduckgo"] = "duckduckgo"

    # retry / timeout behavior
    max_retries: int = Field(default=3, ge=1, le=10)
    request_timeout_seconds: float = Field(default=180.0, gt=0.0)
    backoff_multiplier_seconds: float = Field(default=1.5, gt=0.0)

    # agent thresholds
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    claim_extraction_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    claim_extraction_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    min_evidence_snippets: int = Field(default=1, ge=0)
    hallucination_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # model selection
    default_llm_provider: Literal["openai", "anthropic", "groq", "ollama"] = "ollama"
    default_model_name: str = "llama3.1:8b"

    @field_validator("ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()