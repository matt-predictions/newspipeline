"""Shared agent plumbing.

Every LLM-calling agent in the pipeline (brief writer, JJJ editor,
conversation panel, market proposer) routes through this module for
retry / backoff / token accounting. Keeps the surface area tiny:

- ``anthropic_call_with_retry`` — Anthropic primary→fallback model walk
  with exponential backoff over overload / rate-limit / 5xx errors.
- ``openai_chat_with_retry`` — OpenAI chat-completions with the same shape.
- ``rough_cost_cents`` — coarse $/token estimator used by the run manifest
  + spend ledger.
- ``AgentResult`` / ``make_result`` — light dataclass agents return so the
  orchestrator can keep a uniform per-step manifest entry.

The old POC carried a ``SpendTracker``, ``BudgetExceeded`` raise, a
``gather_agents`` helper, and a ``with_anthropic_retry`` shim — none of
them survived the refactor and they were removed in the modernization
pass. The per-day spend table in ``app.core.db`` is still wired (manifest
writes go through it) but is no longer wrapped by a tracker class.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


_TRANSIENT_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 529}


async def anthropic_call_with_retry(
    client: AsyncAnthropic,
    *,
    primary_model: str,
    fallback_model: str | None,
    max_tokens: int,
    messages: list[dict[str, Any]],
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Call Anthropic with backoff + model fallback.

    Tries ``primary_model`` for up to ``attempts`` rounds with exponential
    backoff; if every attempt fails on a transient error and a
    ``fallback_model`` is set, retries with the fallback before raising.
    """
    last_err: Exception | None = None
    candidates = [primary_model]
    if fallback_model and fallback_model != primary_model:
        candidates.append(fallback_model)
    extra = extra or {}
    for model_id in candidates:
        delay = base_delay
        for _ in range(attempts):
            try:
                return await client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    messages=messages,  # type: ignore[arg-type]
                    **extra,
                )
            except AnthropicAPIStatusError as e:
                last_err = e
                code = getattr(e, "status_code", None)
                if code in _TRANSIENT_HTTP:
                    await asyncio.sleep(delay + random.random() * 0.4)
                    delay = min(delay * 2.0, max_delay)
                    continue
                raise
            except Exception as e:
                last_err = e
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, max_delay)
    assert last_err is not None
    raise last_err


async def openai_chat_with_retry(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: dict[str, Any] | None = None,
    temperature: float = 0.4,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    extra: dict[str, Any] | None = None,
) -> Any:
    """OpenAI chat completions with the same retry shape as Anthropic."""
    last_err: Exception | None = None
    delay = base_delay
    extra = extra or {}
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **extra,
    }
    if response_format:
        kwargs["response_format"] = response_format
    for _ in range(attempts):
        try:
            return await client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            code = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None
            )
            if code in _TRANSIENT_HTTP or code is None:
                await asyncio.sleep(delay + random.random() * 0.4)
                delay = min(delay * 2.0, max_delay)
                continue
            raise
    assert last_err is not None
    raise last_err


# Rough public list-price snapshots at planning time (cents per 1M tokens).
# Used only for the run-manifest cost estimate; exact billing comes from
# the provider invoice.
_PRICE_PER_M_TOKEN: dict[str, tuple[int, int]] = {
    "opus": (1500, 7500),
    "sonnet": (300, 1500),
    "haiku": (80, 400),
    "gpt-top": (500, 1500),
    "gpt-mini": (15, 60),
    "embed": (2, 0),
}


def rough_cost_cents(
    in_tok: int, out_tok: int, model_class: str = "sonnet"
) -> int:
    in_rate, out_rate = _PRICE_PER_M_TOKEN.get(
        model_class, _PRICE_PER_M_TOKEN["sonnet"]
    )
    cents = (in_tok * in_rate + out_tok * out_rate) // 1_000_000
    return max(1, int(cents))


def classify_model(model_id: str) -> str:
    m = (model_id or "").lower()
    if "opus" in m:
        return "opus"
    if "haiku" in m:
        return "haiku"
    if "sonnet" in m or "claude" in m:
        return "sonnet"
    if "embedding" in m:
        return "embed"
    if "mini" in m or "nano" in m:
        return "gpt-mini"
    return "gpt-top"


@dataclass
class AgentResult:
    """What every agent should return so the orchestrator can keep a uniform trace."""

    role: str
    model: str
    provider: str  # "anthropic" | "openai"
    input_summary: str
    output_summary: str
    spend_cents: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


def text_from_anthropic(msg: Any) -> str:
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def make_result(
    role: str,
    *,
    model: str,
    provider: str,
    input_summary: str,
    output_summary: str,
    spend_cents: int = 0,
    extras: dict[str, Any] | None = None,
) -> AgentResult:
    return AgentResult(
        role=role,
        model=model,
        provider=provider,
        input_summary=input_summary[:240],
        output_summary=output_summary[:520],
        spend_cents=spend_cents,
        extras=extras or {},
    )
