"""Provider-agnostic LLM interface.

The pipeline never imports a vendor SDK directly. Everything goes through ``LLMClient``,
which keeps three things possible: swapping providers without touching the RAG code,
running the whole system with no API key at all, and recording in every evaluation
report exactly which model produced the numbers.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0


class LLMClient(ABC):
    provider: str = "base"
    model: str = ""

    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse: ...

    def describe(self) -> dict[str, str]:
        return {"provider": self.provider, "model": self.model}

    @property
    def is_generative(self) -> bool:
        """False for the extractive engine, which cannot synthesise new text."""
        return True


class NullLLM(LLMClient):
    """Placeholder used when no provider is configured.

    It never pretends to answer: the generator checks ``is_generative`` and routes to
    the extractive engine instead. Making this explicit avoids the common failure of
    silently producing an ungrounded answer from a half-configured client.
    """

    provider = "none"
    model = "extractive"

    def complete(self, system: str, user: str) -> LLMResponse:  # pragma: no cover
        raise RuntimeError(
            "No LLM provider configured. Set POLICYLENS_LLM_PROVIDER and the matching "
            "API key, or use the extractive engine."
        )

    @property
    def is_generative(self) -> bool:
        return False


def _has(*env_keys: str) -> bool:
    return any(os.getenv(k) for k in env_keys)


def build_llm(config: LLMConfig | None = None) -> LLMClient:
    """Construct the configured provider.

    ``auto`` probes for credentials in a fixed order and falls back to the extractive
    engine. The order is deliberate: whichever key is present is almost certainly the
    one the user intends.
    """
    cfg = config or LLMConfig()
    provider = cfg.provider.lower().strip()

    def _anthropic() -> LLMClient:
        from .anthropic_backend import AnthropicLLM

        return AnthropicLLM(cfg)

    def _openai() -> LLMClient:
        from .openai_backend import OpenAILLM

        return OpenAILLM(cfg)

    def _gemini() -> LLMClient:
        from .gemini_backend import GeminiLLM

        return GeminiLLM(cfg)

    def _ollama() -> LLMClient:
        from .ollama_backend import OllamaLLM

        return OllamaLLM(cfg)

    builders = {
        "anthropic": _anthropic,
        "claude": _anthropic,
        "openai": _openai,
        "gemini": _gemini,
        "google": _gemini,
        "ollama": _ollama,
        "none": lambda: NullLLM(),
        "extractive": lambda: NullLLM(),
    }

    if provider in builders:
        return builders[provider]()

    if provider != "auto":
        raise ValueError(f"Unknown LLM provider: {cfg.provider}")

    candidates: list[tuple[str, bool]] = [
        ("anthropic", _has("ANTHROPIC_API_KEY")),
        ("openai", _has("OPENAI_API_KEY")),
        ("gemini", _has("GEMINI_API_KEY", "GOOGLE_API_KEY")),
        ("ollama", _has("POLICYLENS_USE_OLLAMA")),
    ]
    for name, available in candidates:
        if not available:
            continue
        try:
            return builders[name]()
        except Exception as exc:  # noqa: BLE001
            logger.warning("provider %s configured but unusable (%s)", name, exc)

    logger.info("no LLM provider configured; using the keyless extractive engine")
    return NullLLM()
