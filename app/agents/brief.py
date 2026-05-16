"""One-shot brief writer.

Single LLM call that takes a clustered news story (source articles + verbatim
posts) and returns everything we need to produce a video:

- ``per_outlet_angle`` — what each outlet uniquely emphasizes
- ``divergent_framings`` — non-obvious framing splits across outlets
- ``public_opinion_sway`` — which cohorts move which direction
- ``story_grade`` — 1-10 wire weight
- ``hook`` — one-line headline of the brief
- ``higgsfield`` — {prompt, camera_move, aspect_ratio} ready for the Higgsfield API
- ``hero_image_prompt`` — what to render for the reference still

Replaces the old researcher/director/critic/editor/cross_source_analyst stack.
POC-grade: one call, one JSON output, no debate loops.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.agents.base import (
    anthropic_call_with_retry,
    classify_model,
    openai_chat_with_retry,
    rough_cost_cents,
    text_from_anthropic,
)
from app.core.config import get_settings
from app.core.jsonx import extract_json_object
from app.prompts import load_prompt


_HIGGSFIELD_CAMERA_MOVES = (
    "crash_zoom",
    "dolly_in",
    "dolly_out",
    "tracking_shot",
    "orbit",
    "static",
    "handheld_push_in",
    "vertigo",
)


def _user_prompt(
    sources: list[dict[str, Any]],
    *,
    market_context: str = "",
    today: datetime | None = None,
) -> str:
    today = today or datetime.now(timezone.utc)
    today_iso = today.strftime("%Y-%m-%d")
    rows: list[str] = []
    for s in sources[:18]:
        oid = s.get("outlet_id", "?")
        lean = s.get("outlet_lean", "")
        title = (s.get("title") or "").strip().replace("\n", " ")
        summary = (s.get("summary") or "")[:240].replace("\n", " ").strip()
        rows.append(f"[{oid}|{lean}] {title}\n    {summary}")
    cluster = "\n".join(rows)
    market_block = ""
    if market_context:
        market_block = f"\nMARKET CONTEXT (optional flavor only — story is the spine):\n{market_context}\n"
    moves = " | ".join(_HIGGSFIELD_CAMERA_MOVES)
    return load_prompt(
        "brief.user",
        today_iso=today_iso,
        cluster=cluster,
        market_block=market_block,
        moves=moves,
    )


def _default_brief(reason: str) -> dict[str, Any]:
    return {
        "hook": reason,
        "per_outlet_angle": {},
        "convergent_facts": [],
        "divergent_framings": [],
        "public_opinion_sway": {
            "shift_direction": "muted",
            "magnitude_pp": 0,
            "duration_days": 0,
            "cohorts_moved_positive": [],
            "cohorts_moved_negative": [],
            "rationale": reason,
        },
        "story_grade": 0,
        "what_to_watch_next": reason,
        "higgsfield": {
            "prompt": reason,
            "camera_move": "static",
            "aspect_ratio": "9:16",
            "duration_s": 5,
        },
        "hero_image_prompt": reason,
    }


_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def _scrub_stale_years(obj: Any, *, current_year: int, target_year: int) -> Any:
    """Recursively bump any year < current_year to target_year.

    The LLM is told to use today's date, but its training prior occasionally
    leaks through. One regex sweep keeps the output self-consistent.
    """
    if isinstance(obj, str):
        return _YEAR_RE.sub(
            lambda m: str(target_year) if int(m.group(0)) < current_year else m.group(0),
            obj,
        )
    if isinstance(obj, list):
        return [_scrub_stale_years(x, current_year=current_year, target_year=target_year) for x in obj]
    if isinstance(obj, dict):
        return {k: _scrub_stale_years(v, current_year=current_year, target_year=target_year) for k, v in obj.items()}
    return obj


async def _write_brief_openai(
    client: AsyncOpenAI, prompt: str
) -> tuple[str, int, int, str]:
    s = get_settings()
    r = await openai_chat_with_retry(
        client,
        model=s.openai_model_top,
        messages=[
            {"role": "system", "content": load_prompt("brief.system").strip()},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    text = r.choices[0].message.content or "{}"
    usage = getattr(r, "usage", None)
    tin = getattr(usage, "prompt_tokens", 0) if usage else 0
    tout = getattr(usage, "completion_tokens", 0) if usage else 0
    return text, tin, tout, s.openai_model_top


async def _write_brief_anthropic(
    client: AsyncAnthropic, prompt: str
) -> tuple[str, int, int, str]:
    """Anthropic-only fallback. Same prompt, same JSON shape."""
    s = get_settings()
    system = load_prompt("brief.system").strip()
    msg = await anthropic_call_with_retry(
        client,
        primary_model=s.anthropic_model_sonnet,
        fallback_model=s.anthropic_model_opus,
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{system}\n\n"
                    "Return ONLY a single JSON object — no markdown fences, no prose.\n\n"
                    f"{prompt}"
                ),
            }
        ],
    )
    text = text_from_anthropic(msg) or "{}"
    usage = getattr(msg, "usage", None)
    tin = getattr(usage, "input_tokens", 0) if usage else 0
    tout = getattr(usage, "output_tokens", 0) if usage else 0
    return text, tin, tout, s.anthropic_model_sonnet


async def write_brief(
    client: AsyncOpenAI | AsyncAnthropic | None,
    sources: list[dict[str, Any]],
    *,
    market_context: str = "",
) -> dict[str, Any]:
    """One LLM call → dict with hook, cross-outlet analysis, and Higgsfield prompt.

    Routes through OpenAI when ``OPENAI_API_KEY`` is set; falls back to
    Anthropic Sonnet when only ``ANTHROPIC_API_KEY`` is set. Output shape is
    identical so the rest of the pipeline doesn't care which provider ran.
    Passing ``client=None`` lets us construct the right client for the
    configured provider; passing an explicit client of the right type is also
    fine (used by tests / callers that already have a pool).
    """
    s = get_settings()
    today = datetime.now(timezone.utc)
    if not sources:
        return _default_brief("no sources in cluster")
    prompt = _user_prompt(sources, market_context=market_context, today=today)

    if s.has_openai:
        oc = client if isinstance(client, AsyncOpenAI) else AsyncOpenAI(api_key=s.openai_api_key)
        text, tin, tout, model = await _write_brief_openai(oc, prompt)
    elif s.has_anthropic:
        ac = client if isinstance(client, AsyncAnthropic) else AsyncAnthropic(api_key=s.anthropic_api_key)
        text, tin, tout, model = await _write_brief_anthropic(ac, prompt)
    else:
        return _default_brief("no API keys configured")

    data = extract_json_object(text) or _default_brief("empty LLM output")
    for key, default in _default_brief("missing").items():
        data.setdefault(key, default)
    data = _scrub_stale_years(
        data, current_year=today.year, target_year=today.year + 1
    )
    data["_meta"] = {
        "model": model,
        "tokens_in": tin,
        "tokens_out": tout,
        "spend_cents": rough_cost_cents(tin, tout, classify_model(model)),
    }
    return data
