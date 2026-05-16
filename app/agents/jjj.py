"""JJJ — post-panel editor agent.

After the persona panel concludes, JJJ takes the draft brief and the panel
transcript and rewrites two fields:

- ``hook`` — punchier, ≤ 100 chars, declarative (no hedging)
- ``higgsfield.prompt`` — sharper video prompt, tighter language, leaning
  into whichever framing WON the debate (DA-swayed → DA's framing; room-
  swayed → consensus framing; stalemate → the higher-stakes framing)

ONE LLM call. Provider routing matches ``brief.py`` (OpenAI preferred,
Anthropic fallback). JJJ preserves facts and entities verbatim — they only
sharpen voice and structure. The hero_image_prompt is intentionally left
alone (the hero has already been considered and we don't want JJJ to
re-roll the visual brief).

Editor notes are stashed at ``brief["_jjj"]`` so we can surface them in the
event README footer.
"""

from __future__ import annotations

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


def _compelling_turns(transcript: dict[str, Any], *, k: int = 4) -> list[str]:
    """Pick a small set of representative turns for JJJ.

    Heuristic: keep the DA's last move + the turn(s) that cite a probability
    closest to the final consensus, padded with the longest substantive
    turns. We want JJJ to see voice + the decisive arguments, not the whole
    14-turn log.
    """
    turns = transcript.get("turns") or []
    if not turns:
        return []
    da_id = transcript.get("devil_advocate_id")
    final_pct = transcript.get("consensus_probability_pct")

    scored: list[tuple[float, dict[str, Any]]] = []
    for t in turns:
        text = (t.get("text") or "").strip()
        if not text:
            continue
        score = float(len(text))
        if da_id and t.get("persona_id") == da_id:
            score += 800.0
        pct = t.get("probability_pct")
        if pct is not None and final_pct is not None:
            score += max(0.0, 200.0 - abs(int(pct) - int(final_pct)) * 10.0)
        scored.append((score, t))
    scored.sort(key=lambda x: -x[0])
    picked = scored[:k]
    picked.sort(key=lambda x: int(x[1].get("order", 0)))
    lines: list[str] = []
    for _, t in picked:
        pct = t.get("probability_pct")
        pct_str = f" [{pct}c]" if pct is not None else ""
        da_badge = " (DA)" if da_id and t.get("persona_id") == da_id else ""
        lines.append(
            f"- {t.get('persona_id', '?')}{da_badge}{pct_str}: "
            f"{(t.get('text') or '').strip()[:320]}"
        )
    return lines


def _user_prompt(brief: dict[str, Any], transcript: dict[str, Any]) -> str:
    current_hook = (brief.get("hook") or "").strip()
    higgs = brief.get("higgsfield") or {}
    current_prompt = (higgs.get("prompt") or "").strip()
    ended = str(transcript.get("ended_reason") or "unknown")
    consensus = transcript.get("consensus_probability_pct")
    consensus_str = f"{consensus}c YES" if consensus is not None else "no consensus"
    da_id = transcript.get("devil_advocate_id") or "—"
    quotes = "\n".join(_compelling_turns(transcript)) or "(no transcript turns)"
    return load_prompt(
        "jjj.user",
        current_hook_len=len(current_hook),
        current_hook=current_hook,
        current_prompt=current_prompt,
        ended=ended,
        consensus_str=consensus_str,
        da_id=da_id,
        quotes=quotes,
    )


async def _edit_openai(
    client: AsyncOpenAI, prompt: str
) -> tuple[str, int, int, str]:
    s = get_settings()
    r = await openai_chat_with_retry(
        client,
        model=s.openai_model_top,
        messages=[
            {"role": "system", "content": load_prompt("jjj.system").strip()},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.35,
    )
    text = r.choices[0].message.content or "{}"
    usage = getattr(r, "usage", None)
    tin = getattr(usage, "prompt_tokens", 0) if usage else 0
    tout = getattr(usage, "completion_tokens", 0) if usage else 0
    return text, tin, tout, s.openai_model_top


async def _edit_anthropic(
    client: AsyncAnthropic, prompt: str
) -> tuple[str, int, int, str]:
    s = get_settings()
    system = load_prompt("jjj.system").strip()
    msg = await anthropic_call_with_retry(
        client,
        primary_model=s.anthropic_model_sonnet,
        fallback_model=s.anthropic_model_opus,
        max_tokens=1200,
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


async def edit_brief(
    client: AsyncOpenAI | AsyncAnthropic | None,
    brief: dict[str, Any],
    panel_transcript: dict[str, Any],
) -> dict[str, Any]:
    """Mutate ``brief`` in-place with JJJ's polished hook + higgsfield.prompt.

    Returns the same dict for caller convenience. On any failure JJJ leaves
    the brief untouched and stashes a warning under ``brief["_jjj"].error`` —
    callers should keep moving (the original brief from ``brief.py`` is
    perfectly usable on its own).
    """
    s = get_settings()
    if not brief:
        return brief
    if not (s.has_openai or s.has_anthropic):
        brief.setdefault("_jjj", {})["error"] = "no API keys configured"
        return brief
    if not panel_transcript or not panel_transcript.get("turns"):
        brief.setdefault("_jjj", {})["error"] = "no panel transcript"
        return brief

    prompt = _user_prompt(brief, panel_transcript)
    try:
        if s.has_openai:
            oc = client if isinstance(client, AsyncOpenAI) else AsyncOpenAI(api_key=s.openai_api_key)
            text, tin, tout, model = await _edit_openai(oc, prompt)
        else:
            ac = client if isinstance(client, AsyncAnthropic) else AsyncAnthropic(api_key=s.anthropic_api_key)
            text, tin, tout, model = await _edit_anthropic(ac, prompt)
    except Exception as exc:
        brief.setdefault("_jjj", {})["error"] = f"{type(exc).__name__}: {exc}"
        return brief

    data = extract_json_object(text) or {}
    new_hook = str(data.get("hook") or "").strip()
    new_prompt = str(data.get("higgsfield_prompt") or "").strip()
    notes = str(data.get("edit_notes") or "").strip()

    if new_hook:
        brief["hook"] = new_hook[:200]
    if new_prompt:
        higgs = dict(brief.get("higgsfield") or {})
        higgs["prompt"] = new_prompt
        brief["higgsfield"] = higgs
    brief["_jjj"] = {
        "notes": notes[:400],
        "model": model,
        "tokens_in": tin,
        "tokens_out": tout,
        "spend_cents": rough_cost_cents(tin, tout, classify_model(model)),
    }
    return brief
