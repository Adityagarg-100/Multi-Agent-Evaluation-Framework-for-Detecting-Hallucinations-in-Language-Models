"""
Provider-agnostic base draft generator for the Multi-Agent LLM
Hallucination Supervisor.

This module produces the initial candidate draft that the rest of the
pipeline (claim extraction, evidence retrieval, judging) will subsequently
verify for hallucinations. It is deliberately decoupled from any single
LLM vendor via `litellm`.
"""

from __future__ import annotations

import logging
from typing import Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import settings

logger = logging.getLogger(__name__)


class DraftGenerationError(Exception):
    """
    Raised when draft generation fails after exhausting all configured
    retry attempts, or when the LLM provider returns an empty/invalid
    response.
    """


class BaseDraftGenerator:
    """
    Generates an initial LLM draft response for a given prompt, using
    `litellm` to remain agnostic to the underlying provider (OpenAI, Groq,
    Ollama, Anthropic, etc.).

    Attributes:
        provider: LLM provider identifier (e.g., 'openai', 'groq', 'ollama').
        model_name: Model identifier passed to `litellm.completion`.
        temperature: Sampling temperature for draft generation.
        max_retries: Maximum retry attempts on transient failures.
        request_timeout: Per-request timeout, in seconds.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize the draft generator.

        Args:
            provider: Optional override for the LLM provider. Defaults to
                `settings.default_llm_provider`.
            model_name: Optional override for the model identifier.
                Defaults to `settings.default_model_name`.
            temperature: Sampling temperature for generation. Higher
                values produce more varied (and hallucination-prone)
                drafts, which is desirable for stress-testing the
                verification pipeline.

        Raises:
            DraftGenerationError: If the `litellm` package is not
                installed.
        """
        try:
            import litellm  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DraftGenerationError(
                "The 'litellm' package is required for BaseDraftGenerator. "
                "Install it with `pip install litellm`."
            ) from exc

        self._litellm = litellm
        self._litellm.drop_params = True  # silently drop unsupported kwargs per-provider

        self.provider: str = provider or settings.default_llm_provider
        self.model_name: str = model_name or settings.default_model_name
        self.temperature: float = temperature
        self.max_retries: int = settings.max_retries
        self.request_timeout: float = settings.request_timeout_seconds

        self._resolved_model = self._resolve_model_string(self.provider, self.model_name)

    @staticmethod
    def _resolve_model_string(provider: str, model_name: str) -> str:
        """
        Construct the `litellm`-compatible model identifier from a
        provider/model pair.

        Args:
            provider: LLM provider identifier.
            model_name: Bare model name (e.g., 'gpt-4o-mini', 'llama3').

        Returns:
            A model string formatted as `litellm` expects (e.g.,
            'ollama/llama3', 'groq/llama3-70b-8192'). OpenAI model names
            are passed through unmodified since that is `litellm`'s
            default routing target.
        """
        if provider == "openai":
            return model_name
        if provider == "groq":
            return f"groq/{model_name}"
        if provider == "ollama":
            return f"ollama/{model_name}"
        # Unknown provider: pass through as-is and let litellm raise if invalid.
        return model_name

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(
            multiplier=settings.backoff_multiplier_seconds, min=1, max=20
        ),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, RuntimeError)),
    )
    def _call_llm(self, prompt: str) -> str:
        """
        Issue a single completion request to the configured LLM provider.

        Args:
            prompt: The user prompt to generate a draft response for.

        Returns:
            The raw text content of the model's response.

        Raises:
            RuntimeError: Wraps any provider-level exception (timeouts,
                rate limits, API outages, malformed responses) so
                tenacity's retry predicate can catch it uniformly.
        """
        try:
            response = self._litellm.completion(
                model=self._resolved_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                timeout=self.request_timeout,
                api_key=self._resolve_api_key(),
                api_base=settings.ollama_base_url if self.provider == "ollama" else None,
            )
        except Exception as exc:  # noqa: BLE001 - normalize all provider errors
            logger.warning(
                "LLM call failed (provider=%s, model=%s): %s",
                self.provider,
                self._resolved_model,
                exc,
            )
            raise RuntimeError(f"LLM completion failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError) as exc:
            raise RuntimeError(
                f"Malformed LLM response structure: {response!r}"
            ) from exc

        if not content or not content.strip():
            raise RuntimeError("LLM returned an empty completion.")

        return content

    def _resolve_api_key(self) -> Optional[str]:
        """
        Select the appropriate secret API key for the active provider.

        Returns:
            The plaintext API key string for `self.provider`, or `None`
            when the provider requires no key (e.g., a local Ollama
            server).
        """
        if self.provider == "openai" and settings.openai_api_key is not None:
            return settings.openai_api_key.get_secret_value()
        if self.provider == "groq" and settings.groq_api_key is not None:
            return settings.groq_api_key.get_secret_value()
        return None

    def generate_draft(self, prompt: str) -> str:
        """
        Generate a candidate draft response for the given prompt.

        This is the primary public entry point consumed by downstream
        orchestration code. All transient failures are retried internally
        with exponential backoff before surfacing a `DraftGenerationError`.

        Args:
            prompt: The user-facing prompt/instruction to generate a
                response for.

        Returns:
            The generated draft text.

        Raises:
            DraftGenerationError: If `prompt` is empty/whitespace-only, or
                if generation fails after exhausting all retry attempts.
        """
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise DraftGenerationError("generate_draft called with an empty prompt.")

        try:
            return self._call_llm(cleaned_prompt)
        except Exception as exc:  # noqa: BLE001 - final catch after retries exhausted
            raise DraftGenerationError(
                f"Draft generation failed after {self.max_retries} attempts "
                f"(provider={self.provider}, model={self._resolved_model}): {exc}"
            ) from exc