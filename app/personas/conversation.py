"""Round-robin persona conversation.

Personas argue PROBABILITIES, not feelings. Each turn:

- max 3 sentences
- must state a probability OR directly engage with the previous probability
- may ask at most one question to a named persona on the same panel
- voice is anchored to the persona's ``betting_voice``

The transcript is BOTH a debug artifact saved to ``conversation.json`` AND a
creative asset — the editor agent is allowed to lift verbatim lines into the
final script (kinetic captions or VO).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

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


CONSENSUS_RANGE_CENTS: int = 5
DEFAULT_MAX_TURNS: int = 8
REPETITION_JACCARD: float = 0.78


@dataclass
class ConversationTurn:
    order: int
    persona_id: str
    text: str
    probability_pct: int | None
    asks_persona: str | None = None
    is_consensus_call: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationTranscript:
    panel: list[str]
    turns: list[ConversationTurn] = field(default_factory=list)
    consensus_probability_pct: int | None = None
    ended_reason: str = "unknown"
    moderator_take: str = ""
    market_context_summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel": self.panel,
            "turns": [t.as_dict() for t in self.turns],
            "consensus_probability_pct": self.consensus_probability_pct,
            "ended_reason": self.ended_reason,
            "moderator_take": self.moderator_take,
            "market_context_summary": self.market_context_summary,
        }

    def verbatim_lines_for_editor(self, *, max_lines: int = 6) -> list[str]:
        """Quotable one-liners the editor can lift into the final script."""
        out: list[str] = []
        for t in self.turns:
            first_sent = re.split(r"(?<=[.!?])\s", t.text.strip(), maxsplit=1)[0]
            if 30 <= len(first_sent) <= 160:
                out.append(f"{t.persona_id}: \"{first_sent.strip()}\"")
            if len(out) >= max_lines:
                break
        return out


_PROB_RE = re.compile(
    r"(?:(\d{1,3})\s*(?:c|%|cent)|\b(\d{1,3})\s*(?:percent)\b|\bI'?d\s+(?:bet|price)\s+(?:that\s+at\s+)?(\d{1,3}))",
    re.IGNORECASE,
)


def parse_probability(text: str) -> int | None:
    """Return a probability in 0..100 if the text mentions one, else None."""
    for m in _PROB_RE.finditer(text):
        for grp in m.groups():
            if grp:
                try:
                    v = int(grp)
                except ValueError:
                    continue
                if 0 <= v <= 100:
                    return v
    return None


def _jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"[a-z]{3,}", a.lower()))
    sb = set(re.findall(r"[a-z]{3,}", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _detect_repetition(turns: list[ConversationTurn]) -> bool:
    if len(turns) < 3:
        return False
    a = turns[-1].text
    b = turns[-2].text
    return _jaccard(a, b) >= REPETITION_JACCARD


def _detect_consensus(turns: list[ConversationTurn]) -> int | None:
    probs = [t.probability_pct for t in turns[-3:] if t.probability_pct is not None]
    if len(probs) < 3:
        return None
    if max(probs) - min(probs) <= CONSENSUS_RANGE_CENTS:
        return int(round(sum(probs) / len(probs)))
    return None


def _turn_user_prompt(
    *,
    persona: dict[str, Any],
    summary: str,
    market_context: str,
    history: list[ConversationTurn],
    other_ids: list[str],
) -> str:
    history_block = "\n".join(
        f"{t.persona_id}: {t.text} "
        + (f"[priced {t.probability_pct}c]" if t.probability_pct is not None else "")
        for t in history[-6:]
    ) or "(this is the opening turn)"
    return f"""You are {persona['id']}. Card:
{persona.get('card', '')}

Dialect: {persona.get('dialect', '')}
Bias targets: {persona.get('bias_targets', [])}
Betting voice (anchor your delivery here): {persona.get('betting_voice', '')}

You're on a Polymarket-style probability panel debating ONE news story.

RULES (hard):
- ≤3 sentences
- MUST state a probability you'd bet in either "Xc on YES" / "Y%" form AT LEAST ONCE, OR explicitly engage with the previous probability ("I think 35c is rich, fade it to 22c").
- MAY ask one direct question to another named panelist (one of: {other_ids}) — but only one.
- NO breathless hype. No 'breaking', 'shocking', 'epic'.
- Sound like YOU (use dialect + betting_voice). Don't impersonate the others.

NEWS SUMMARY:
{summary[:3500]}

{market_context.strip()}

PRIOR TURNS:
{history_block}

Return JSON ONLY:
{{
  "text": "your turn (≤3 sentences)",
  "probability_pct": int 0-100 or null,
  "asks_persona": "persona_id or null"
}}
"""


async def _one_turn(
    client: AsyncAnthropic | AsyncOpenAI,
    persona: dict[str, Any],
    *,
    summary: str,
    market_context: str,
    history: list[ConversationTurn],
    other_ids: list[str],
    order: int,
) -> tuple[ConversationTurn, int]:
    s = get_settings()
    prompt = _turn_user_prompt(
        persona=persona,
        summary=summary,
        market_context=market_context,
        history=history,
        other_ids=other_ids,
    )
    if isinstance(client, AsyncAnthropic):
        msg = await anthropic_call_with_retry(
            client,
            primary_model=s.anthropic_model_sonnet,
            fallback_model=s.anthropic_model_opus,
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}],
        )
        text = text_from_anthropic(msg)
        usage = getattr(msg, "usage", None)
        tin = getattr(usage, "input_tokens", 0) if usage else 0
        tout = getattr(usage, "output_tokens", 0) if usage else 0
        model_for_cost = s.anthropic_model_sonnet
    else:
        r = await openai_chat_with_retry(
            client,
            model=s.openai_model_top,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You roleplay one named Polymarket-style panelist. "
                        "Stay in voice. Return STRICT JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
        )
        text = r.choices[0].message.content or "{}"
        usage = getattr(r, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) if usage else 0
        tout = getattr(usage, "completion_tokens", 0) if usage else 0
        model_for_cost = s.openai_model_top

    data = extract_json_object(text) or {}
    body = str(data.get("text") or "").strip()
    prob = data.get("probability_pct")
    if not isinstance(prob, int):
        prob = parse_probability(body)
    asks = data.get("asks_persona") or None
    cents = rough_cost_cents(tin, tout, classify_model(model_for_cost))
    return (
        ConversationTurn(
            order=order,
            persona_id=persona["id"],
            text=body,
            probability_pct=prob,
            asks_persona=asks if asks in other_ids else None,
        ),
        cents,
    )


async def run_conversation(
    panel_personas: list[dict[str, Any]],
    *,
    summary: str,
    market_context: str = "",
    max_turns: int = DEFAULT_MAX_TURNS,
) -> tuple[ConversationTranscript, AgentResult]:
    """Round-robin conversation across the picked panel. Returns transcript + cost trace.

    Provider routing: Anthropic Sonnet when ``ANTHROPIC_API_KEY`` is present
    (preferred — better at sustained-voice persona writing), otherwise OpenAI
    ``gpt-4o`` with JSON-mode. Same prompt either way.
    """
    s = get_settings()
    ids = [p["id"] for p in panel_personas]
    transcript = ConversationTranscript(
        panel=list(ids), market_context_summary=market_context[:400]
    )
    provider = "anthropic" if s.has_anthropic else "openai"
    model_name = s.anthropic_model_sonnet if s.has_anthropic else s.openai_model_top
    if not panel_personas:
        transcript.ended_reason = "no_panel"
        return transcript, make_result(
            "conversation",
            model=model_name,
            provider=provider,
            input_summary=summary[:200],
            output_summary="no_panel",
            spend_cents=0,
            extras={"turns": 0},
        )

    client: AsyncAnthropic | AsyncOpenAI
    if s.has_anthropic:
        client = AsyncAnthropic(api_key=s.anthropic_api_key)
    else:
        client = AsyncOpenAI(api_key=s.openai_api_key)
    total_cents = 0
    order = 0
    while len(transcript.turns) < max_turns:
        persona = panel_personas[len(transcript.turns) % len(panel_personas)]
        other_ids = [pid for pid in ids if pid != persona["id"]]
        turn, cents = await _one_turn(
            client,
            persona,
            summary=summary,
            market_context=market_context,
            history=transcript.turns,
            other_ids=other_ids,
            order=order,
        )
        transcript.turns.append(turn)
        total_cents += cents
        order += 1
        consensus = _detect_consensus(transcript.turns)
        if consensus is not None:
            transcript.consensus_probability_pct = consensus
            transcript.ended_reason = "consensus"
            break
        if _detect_repetition(transcript.turns):
            transcript.ended_reason = "repetition"
            break
    else:
        transcript.ended_reason = "max_turns"

    result = make_result(
        "conversation",
        model=model_name,
        provider=provider,
        input_summary=summary[:200],
        output_summary=json.dumps(
            {"turns": len(transcript.turns), "ended": transcript.ended_reason}
        ),
        spend_cents=total_cents,
        extras={"turn_count": len(transcript.turns)},
    )
    return transcript, result
