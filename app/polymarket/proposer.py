"""Proposed-market generator agent.

When the live matcher returns ``None`` we still want the brief to surface a
*plausible* Polymarket market spec — slug, outcomes, resolution criteria,
horizon — so the closing CTA isn't a dead end. This module asks Claude to
generate that spec, grounded ONLY in the research output (no clickbait
fabrication, no fake horizons).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.base import (
    AgentResult,
    anthropic_call_with_retry,
    classify_model,
    make_result,
    openai_chat_with_retry,
    rough_cost_cents,
    text_from_anthropic,
)
from app.core.config import get_settings
from app.core.jsonx import extract_json_object
from app.core.models import ResearchOut


class ProposedMarket(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    slug: str = ""
    outcome_type: str = Field(
        default="binary",
        description="binary | multi | scalar",
    )
    outcomes: list[str] = Field(default_factory=list)
    resolution_criteria: str = ""
    resolution_source_hint: str = ""
    horizon: str = ""
    horizon_iso: str | None = None
    confidence_market_attracts_volume: float = 0.5
    rationale: str = ""

    @field_validator("confidence_market_attracts_volume", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return min(1.0, max(0.0, f))

    def as_polymarket_style_url(self) -> str:
        slug = (self.slug or "").strip("/")
        if not slug:
            return ""
        return f"https://polymarket.com/proposed/{slug}"


PROPOSER_SYSTEM = (
    "You are a Polymarket market-design analyst. You propose only well-formed, "
    "objectively resolvable markets grounded in the news cluster the user pastes. "
    "You IGNORE your training-data prior on the current date — the user always pins "
    "TODAY explicitly. All horizons must be in the future relative to that pinned date. "
    "If the cluster does NOT support a clean binary or multi-outcome market with a "
    "concrete resolution date, you set outcome_type to 'multi' with a 'no clean market' "
    "outcome and confidence below 0.2. Return STRICT JSON only."
)


def _user_prompt(
    research: ResearchOut,
    *,
    brand_kit_block: str = "",
    today: datetime | None = None,
    stale_feedback: str = "",
) -> str:
    research_blob = research.model_dump()
    today = today or datetime.now(timezone.utc)
    today_iso = today.strftime("%Y-%m-%d")
    year_now = today.year
    min_horizon = (today + timedelta(days=7)).strftime("%Y-%m-%d")
    stale_block = ""
    if stale_feedback:
        stale_block = (
            "\nPREVIOUS ATTEMPT FAILED FRESHNESS CHECK:\n"
            f"{stale_feedback}\n"
            "Regenerate with a strictly future horizon.\n"
        )
    return f"""TODAY IS {today_iso}. The current year is {year_now}. Any horizon you propose MUST be on or after {min_horizon}. Past years are an automatic fail and the pipeline will reject and re-prompt.

Generate a proposed Polymarket market spec from the research below.
{stale_block}
REQUIRED JSON KEYS:
- title: short, plain English. If the title references a year, that year MUST be {year_now} or later (e.g. "Will the Fed cut rates by 50bp before end of Q3 {year_now}?"). NEVER use a past year.
- slug: kebab-case ASCII, ≤ 60 chars. If the slug references a year, that year MUST be {year_now} or later.
- outcome_type: binary | multi | scalar
- outcomes: list of strings (for binary: ["Yes", "No"])
- resolution_criteria: ONE paragraph quoting the headline ledger and naming a source-of-truth (e.g. "FOMC statement on …", "AP confirmed …"). No speculation.
- resolution_source_hint: where a resolver would look (e.g. "FOMC press release", "OFAC sanctions list", "AP/Reuters wire confirmation")
- horizon: human-readable, in the future ("within 30 days", "by end of Q3 {year_now}", "by {year_now + 1}-03-31"). NEVER reference a year before {year_now}.
- horizon_iso: ISO 8601 deadline strictly AFTER {today_iso} (must be ≥ {min_horizon}T00:00:00Z). If you cannot pin one, return null.
- confidence_market_attracts_volume: 0..1 float (lower = niche / unlikely to clear)
- rationale: 2-3 sentences calling out why this market is grounded and what the brief should highlight

HARD RULES:
- You MAY NOT propose markets the cluster cannot resolve (no "will X resign" if cluster doesn't mention resignation).
- You MAY NOT invent statistics. If the cluster only says "tensions rise", confidence stays ≤ 0.2 and outcome_type is "multi".
- Title may NOT mention named individuals' likeness — use roles ("the chair", "the CEO") if unavoidable.
- DO NOT carry over any year from your training data. The clock you trust is the TODAY pin above.

{brand_kit_block.strip()}

RESEARCH:
{json.dumps(research_blob, ensure_ascii=False)[:8000]}
"""


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _check_freshness(
    pm: ProposedMarket, *, today: datetime | None = None
) -> tuple[bool, str]:
    """Return (is_fresh, reason_if_stale).

    Stale = any embedded year < current year in title / slug / horizon, OR an
    explicit horizon_iso that's already in the past.
    """
    today = today or datetime.now(timezone.utc)
    year_now = today.year
    today_date = today.date()
    issues: list[str] = []

    for field, val in (
        ("title", pm.title or ""),
        ("slug", pm.slug or ""),
        ("horizon", pm.horizon or ""),
    ):
        for m in _YEAR_RE.finditer(val):
            y = int(m.group(0))
            if y < year_now:
                issues.append(f"{field} contains stale year {y}")
            elif y > year_now + 5:
                issues.append(f"{field} contains implausibly far year {y}")

    iso = (pm.horizon_iso or "").strip()
    if iso:
        try:
            parsed = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if parsed.date() <= today_date:
                issues.append(f"horizon_iso {iso} is on or before today ({today_date})")
        except ValueError:
            issues.append(f"horizon_iso {iso} is not parseable ISO 8601")

    if issues:
        return False, "; ".join(issues)
    return True, ""


def _scrub_stale_years(pm: ProposedMarket, *, today: datetime) -> ProposedMarket:
    """Last-resort fix: replace stale year tokens with a sensible future year.

    Runs only when the regen loop still produced a stale spec. We bump every
    stale-year occurrence to ``current_year + 1`` (a reasonable "in the next
    year" hedge) rather than stripping, so grammar stays intact.
    """
    year_now = today.year
    target_year = year_now + 1

    def _bump_year(s: str) -> str:
        if not s:
            return s

        def _sub(m: re.Match[str]) -> str:
            y = int(m.group(0))
            return str(target_year) if y < year_now else m.group(0)

        return _YEAR_RE.sub(_sub, s)

    new = pm.model_copy()
    new.title = _bump_year(pm.title)
    new.slug = _bump_year(pm.slug)
    new.horizon = _bump_year(pm.horizon)
    iso = (pm.horizon_iso or "").strip()
    if iso:
        try:
            parsed = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if parsed.year < year_now or parsed.date() <= today.date():
                bumped = parsed.replace(year=target_year)
                new.horizon_iso = bumped.isoformat().replace("+00:00", "Z")
        except ValueError:
            new.horizon_iso = None
    if not new.horizon.strip():
        new.horizon = "within 90 days"
    if not new.title.strip():
        new.title = (
            f"Will this story produce a wire-confirmed outcome by end of {target_year}?"
        )
    if not new.slug.strip():
        new.slug = f"wire-confirmed-outcome-{target_year}"
    return new


async def _call_anthropic(
    client: AsyncAnthropic, prompt: str
) -> tuple[str, int, int]:
    s = get_settings()
    msg = await anthropic_call_with_retry(
        client,
        primary_model=s.anthropic_model_sonnet,
        fallback_model=s.anthropic_model_opus,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = text_from_anthropic(msg)
    usage = getattr(msg, "usage", None)
    tin = getattr(usage, "input_tokens", 0) if usage else 0
    tout = getattr(usage, "output_tokens", 0) if usage else 0
    return text, tin, tout


async def _call_openai(
    client: AsyncOpenAI, prompt: str
) -> tuple[str, int, int]:
    s = get_settings()
    r = await openai_chat_with_retry(
        client,
        model=s.openai_model_top,
        messages=[
            {"role": "system", "content": PROPOSER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    text = r.choices[0].message.content or "{}"
    usage = getattr(r, "usage", None)
    tin = getattr(usage, "prompt_tokens", 0) if usage else 0
    tout = getattr(usage, "completion_tokens", 0) if usage else 0
    return text, tin, tout


async def propose_market(
    research: ResearchOut,
    *,
    client: AsyncAnthropic | AsyncOpenAI | None = None,
    brand_kit_block: str = "",
) -> tuple[ProposedMarket, AgentResult]:
    """Generate a proposed Polymarket spec.

    Routes through Anthropic Sonnet when ``ANTHROPIC_API_KEY`` is set
    (preferred for clean market-design prose); falls back to OpenAI
    ``gpt-4o`` with JSON-mode when only ``OPENAI_API_KEY`` is set. The
    freshness check + one-shot regen + last-resort year-bump are identical
    in both paths.
    """
    s = get_settings()
    today = datetime.now(timezone.utc)
    prompt = _user_prompt(research, brand_kit_block=brand_kit_block, today=today)

    use_anthropic = s.has_anthropic if not isinstance(client, AsyncOpenAI) else False
    if isinstance(client, AsyncAnthropic):
        use_anthropic = True
    if client is None:
        if use_anthropic:
            client = AsyncAnthropic(api_key=s.anthropic_api_key)
        else:
            client = AsyncOpenAI(api_key=s.openai_api_key)

    provider = "anthropic" if use_anthropic else "openai"
    model_name = s.anthropic_model_sonnet if use_anthropic else s.openai_model_top

    async def _call(p: str) -> tuple[str, int, int]:
        if use_anthropic:
            return await _call_anthropic(client, p)  # type: ignore[arg-type]
        return await _call_openai(client, p)  # type: ignore[arg-type]

    text, tin, tout = await _call(prompt)
    data: dict[str, Any] = extract_json_object(text) or {}
    pm = ProposedMarket.model_validate(data)

    extras: dict[str, Any] = {}
    is_fresh, why = _check_freshness(pm, today=today)
    if not is_fresh:
        extras["freshness_regen"] = why
        regen_prompt = _user_prompt(
            research,
            brand_kit_block=brand_kit_block,
            today=today,
            stale_feedback=why,
        )
        text2, tin2, tout2 = await _call(regen_prompt)
        data2: dict[str, Any] = extract_json_object(text2) or {}
        pm2 = ProposedMarket.model_validate(data2)
        is_fresh2, why2 = _check_freshness(pm2, today=today)
        if is_fresh2:
            pm = pm2
            text = text2
            tin += tin2
            tout += tout2
        else:
            extras["freshness_scrubbed"] = why2
            pm = _scrub_stale_years(pm2, today=today)
            text = pm.model_dump_json()
            tin += tin2
            tout += tout2

    cents = rough_cost_cents(tin, tout, classify_model(model_name))
    result = make_result(
        "market_proposer",
        model=model_name,
        provider=provider,
        input_summary=prompt,
        output_summary=text,
        spend_cents=cents,
        extras={"tokens_in": tin, "tokens_out": tout, **extras},
    )
    return pm, result
