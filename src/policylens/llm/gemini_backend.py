"""Google Gemini backend."""

from __future__ import annotations

import os
import time

from ..config import LLMConfig
from .base import LLMClient, LLMResponse


class GeminiLLM(LLMClient):
    provider = "gemini"

    def __init__(self, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        from google import genai

        self.cfg = cfg
        self.model = cfg.gemini_model
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def complete(self, system: str, user: str) -> LLMResponse:
        from google.genai import types

        start = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.cfg.temperature,
                max_output_tokens=self.cfg.max_tokens,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=response.text or "",
            provider=self.provider,
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
