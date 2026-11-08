"""Anthropic Claude backend. The documented default."""

from __future__ import annotations

import os
import time

from ..config import LLMConfig
from .base import LLMClient, LLMResponse


class AnthropicLLM(LLMClient):
    provider = "anthropic"

    def __init__(self, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig()
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        import anthropic

        self.cfg = cfg
        self.model = cfg.anthropic_model
        self._client = anthropic.Anthropic(timeout=cfg.timeout_s)

    def complete(self, system: str, user: str) -> LLMResponse:
        start = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            input_tokens=getattr(message.usage, "input_tokens", None),
            output_tokens=getattr(message.usage, "output_tokens", None),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
