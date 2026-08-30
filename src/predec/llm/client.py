"""Thin OpenAI client wrapper.

Two LLM calls in the whole pipeline:
  1. Sycophancy detector (judge LLM): is response A agreeing with the prompt's premise?
  2. Report writer: turn 4 detector outputs + dataset summary into a recommendation.

All other components are statistical. We use a structured-output JSON mode
so downstream code can parse the response deterministically.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = os.environ.get("PREDEC_MODEL", "gpt-4o-mini")


@dataclass
class LLMCall:
    """Record of a single LLM invocation."""

    actor: str
    purpose: str
    prompt: str
    response_text: str
    response_json: dict[str, Any] | None
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: float


class LLMClient:
    """Minimal client that records every call (for the trajectory log)."""

    # Pricing per 1M tokens, gpt-4o-mini (Aug 2026)
    PRICE_IN = 0.15 / 1_000_000
    PRICE_OUT = 0.60 / 1_000_000

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.calls: list[LLMCall] = []
        if self.api_key:
            self._client = OpenAI(api_key=self.api_key)
        else:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def call_json(
        self,
        actor: str,
        purpose: str,
        system: str,
        user: str,
        schema_hint: dict[str, Any] | None = None,
        max_tokens: int = 800,
    ) -> LLMCall:
        """Make a structured JSON call. Returns the call record."""
        if not self._client:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in the environment or pass api_key=... "
                "to LLMClient()."
            )

        t0 = time.time()
        # Use JSON mode for reliable downstream parsing
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=0.0,
        )
        dt = (time.time() - t0) * 1000

        text = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        cost = in_tok * self.PRICE_IN + out_tok * self.PRICE_OUT

        call = LLMCall(
            actor=actor,
            purpose=purpose,
            prompt=user,
            response_text=text,
            response_json=parsed,
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            duration_ms=dt,
        )
        self.calls.append(call)
        return call
