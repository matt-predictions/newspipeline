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


_SYSTEM = (
    "You are a senior news-desk editor and video creative director. "
    "You read multiple outlets covering the same story, identify how their "
    "framings diverge, estimate public-opinion sway, and write a single "
    "video prompt for the Higgsfield API (image-to-video). "
    "Return STRICT JSON only — no prose outside the object."
)


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
    return f"""TODAY IS {today_iso}. Use this date for any horizon — never carry years over from training data.

CLUSTER (one story, multiple outlets, verbatim posts):
{cluster}
{market_block}
Return STRICT JSON with EXACTLY these keys:

- "hook": one-line headline summarizing the story in the news-desk's voice (max 110 chars).
- "per_outlet_angle": object mapping outlet_id -> one-sentence summary of what THIS outlet uniquely emphasizes (cite their wording). Mark wires (AFP/AP/Reuters) as "wire repeat" when they're not adding distinctive framing.
- "convergent_facts": array of 3-6 facts every outlet agrees on (verbatim-grounded).
- "divergent_framings": array of objects {{outlets:[...], frame:"..."}} showing real framing splits.
- "public_opinion_sway": {{shift_direction: "positive"|"negative"|"polarizing"|"muted", magnitude_pp: 1-25 integer, duration_days: integer, cohorts_moved_positive: [1-4 audience labels], cohorts_moved_negative: [1-4 audience labels], rationale: 1-2 sentences quoting verbatim cluster phrasing}}.
- "story_grade": integer 1-10 — wire weight (10=top of every front page; 3=filler).
- "what_to_watch_next": 1-2 sentences on the next inflection point.
- "higgsfield": {{
    prompt: 2-4 sentence cinematic prompt for Higgsfield video API. NEVER name living public figures — use symbolic stand-ins (silhouettes, empty podiums, name plates, flags). Include lighting and composition cues, but no rendered text overlays.
    camera_move: one of [{moves}]
    aspect_ratio: "9:16" | "16:9" | "1:1"
    duration_s: 5 | 8 | 12 (integer)
  }}
- "hero_image_prompt": one paragraph for the reference still. Off-black background (#0F1115) edge-to-edge, single cyan accent (#1AB4E0) on the focal element, warm orange (#E07A2F) only on risk/alert beats, solid black silhouettes with cyan rim-light if any figures appear (NEVER named public figures, NEVER faces). ≤8 words of text in the frame.

Hard rules:
- All horizons / dates referenced anywhere must be on or after {today_iso}.
- Never invent quotes — only echo phrasing that appears verbatim in the cluster.
- The video prompt and hero prompt must be visually different (the video moves; the still freezes a beat).

No fields may be missing. No prose outside the JSON.
"""


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
            {"role": "system", "content": _SYSTEM},
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
    msg = await anthropic_call_with_retry(
        client,
        primary_model=s.anthropic_model_sonnet,
        fallback_model=s.anthropic_model_opus,
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{_SYSTEM}\n\n"
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
