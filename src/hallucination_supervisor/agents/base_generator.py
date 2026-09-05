from __future__ import annotations

import logging
from typing import Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.hallucination_supervisor.config import settings

logger = logging.getLogger(__name__)

class DraftGenerationError(Exception):
    """Raised when draft generation fails after all retries are exhausted."""


class BaseDraftGenerator:
    """
    Generates the initial draft response that gets fact-checked downstream.
    Uses litellm so we can point this at OpenAI, Anthropic, Groq, or a local
    Ollama model.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
    ) -> None:
        try:
            import litellm
        except ImportError as exc:
            raise DraftGenerationError(
                "litellm is required for BaseDraftGenerator. pip install litellm"
            ) from exc

        self._litellm = litellm
        self._litellm.drop_params = True

        self.provider = provider or settings.default_llm_provider
        self.model_name = model_name or settings.default_model_name
        self.temperature = temperature
        self.max_retries = settings.max_retries
        self.request_timeout = settings.request_timeout_seconds

        self._resolved_model = self._resolve_model_string(self.provider, self.model_name)

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

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.backoff_multiplier_seconds, min=1, max=20),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, RuntimeError)),
    )
    def _call_llm(self, prompt: str) -> str:
        try:
            response = self._litellm.completion(
                model=self._resolved_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                timeout=self.request_timeout,
                api_key=self._resolve_api_key(),
                api_base=settings.ollama_base_url if self.provider == "ollama" else None,
            )
        except Exception as exc:
            logger.warning("LLM call failed (%s/%s): %s", self.provider, self._resolved_model, exc)
            raise RuntimeError(f"LLM completion failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError) as exc:
            raise RuntimeError(f"Malformed LLM response: {response!r}") from exc

        if not content or not content.strip():
            raise RuntimeError("LLM returned an empty completion.")

        return content

    def generate_draft(self, prompt: str) -> str:
        """Generate a draft response for the given prompt."""
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise DraftGenerationError("generate_draft called with an empty prompt.")

        try:
            return self._call_llm(cleaned_prompt)
        except Exception as exc:
            raise DraftGenerationError(
                f"Draft generation failed after {self.max_retries} attempts "
                f"({self.provider}/{self._resolved_model}): {exc}"
            ) from exc