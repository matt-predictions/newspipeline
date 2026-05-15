"""Shared agent plumbing.

Every specialist agent (researcher, director, critic, editor, moderator,
analyst, ...) used to bake its own retry/backoff/fallback logic inline. This
module is the single shared substrate so future agents inherit the same
hardening for free.

Contents:
- ``anthropic_call_with_retry`` — Anthropic primary→fallback model walk with
  exponential backoff over overload / rate-limit / 5xx errors.
- ``openai_chat_with_retry`` — OpenAI chat-completions with the same shape.
- ``rough_cost_cents`` — coarse $/token estimator so SpendTracker isn't blind.
- ``SpendTracker`` — in-process counter that records into the SQLite spend
  table; raises ``BudgetExceeded`` when the daily cap is exceeded.
- ``AgentResult`` — light dataclass agents return so the orchestrator can keep
  a structured debate-trace + spend ledger without each agent re-inventing it.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable

from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.db import add_spend_cents, get_spend_day


_TRANSIENT_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class BudgetExceeded(RuntimeError):
    """Raised when the per-day spend cap would be crossed."""


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
                    messages=messages,
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
    response_format: dict[str, str] | None = None,
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


_PRICE_PER_M_TOKEN = {
    # rough public list-price snapshots at planning time (cents per 1M tokens)
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


class SpendTracker:
    """Records cents into the SQLite spend table per UTC day.

    Use as ``await tracker.record("director", cents)``. Before kicking off an
    expensive agent, call ``await tracker.assert_budget(extra=cents)`` to fail
    fast when we'd cross today's cap.
    """

    def __init__(self, *, day: str | None = None, cap_cents: int | None = None):
        s = get_settings()
        self._day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._cap = cap_cents if cap_cents is not None else s.max_spend_cents_per_day
        self._in_memory_total: int = 0
        self._by_role: dict[str, int] = {}

    @property
    def day(self) -> str:
        return self._day

    @property
    def total_cents(self) -> int:
        return self._in_memory_total

    @property
    def cap_cents(self) -> int:
        return self._cap

    @property
    def by_role(self) -> dict[str, int]:
        return dict(self._by_role)

    async def assert_budget(self, *, extra: int = 0) -> None:
        used = await get_spend_day(self._day)
        if used + extra > self._cap:
            raise BudgetExceeded(
                f"daily spend cap reached: used={used}c extra={extra}c cap={self._cap}c"
            )

    async def record(self, role: str, cents: int) -> None:
        if cents <= 0:
            return
        self._in_memory_total += cents
        self._by_role[role] = self._by_role.get(role, 0) + cents
        await add_spend_cents(self._day, cents)


async def gather_agents(
    coros: Iterable[Awaitable[AgentResult]],
) -> list[AgentResult]:
    """asyncio.gather over agent coros, preserving order."""
    return list(await asyncio.gather(*coros))


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


# Re-export the legacy callable name some modules still import.
async def with_anthropic_retry(*args, **kwargs):  # pragma: no cover - shim
    return await anthropic_call_with_retry(*args, **kwargs)


# Stable list of callables for callers that want to enumerate agents.
AgentCoro = Callable[..., Awaitable[AgentResult]]
