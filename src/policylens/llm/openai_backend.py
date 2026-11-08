"""OpenAI backend."""

from __future__ import annotations

import os
import time

from ..config import LLMConfig
from .base import LLMClient, LLMResponse


class OpenAILLM(LLMClient):
    provider = "openai"

    def __init__(self, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        self.cfg = cfg
        self.model = cfg.openai_model
        self._client = OpenAI(timeout=cfg.timeout_s)

    def complete(self, system: str, user: str) -> LLMResponse:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=response.choices[0].message.content or "",
            provider=self.provider,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
