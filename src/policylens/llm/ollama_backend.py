"""Ollama backend, for a fully local model with no API key.

Uses the HTTP API directly so no extra dependency is needed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from ..config import LLMConfig
from .base import LLMClient, LLMResponse


class OllamaLLM(LLMClient):
    provider = "ollama"

    def __init__(self, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig()
        self.cfg = cfg
        self.model = cfg.ollama_model
        self.host = cfg.ollama_host.rstrip("/")

    def complete(self, system: str, user: str) -> LLMResponse:
        payload = json.dumps(
            {
                "model": self.model,
                "system": system,
                "prompt": user,
                "stream": False,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.max_tokens,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running?"
            ) from exc

        return LLMResponse(
            text=body.get("response", ""),
            provider=self.provider,
            model=self.model,
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
