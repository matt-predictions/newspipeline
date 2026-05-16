"""Round-robin persona conversation with a devil's advocate.

Each regular persona speaks first; their first-round probabilities give us a
"room median". Then the devil's advocate (any persona with ``role:
devil_advocate`` in its YAML card) joins, anchored to argue the OPPOSITE
direction by at least 35pp, citing concrete historical precedent.

End conditions (in order):
- ``da_swayed``   — devil's advocate moved ≥ 20pp toward the room median.
- ``room_swayed`` — room median moved ≥ 15pp toward the DA's latest probability.
- ``stalemate``  — neither side budged after ``max_turns``.
- ``repetition`` — back-to-back turns are too similar (groupthink fail-safe).
- ``consensus``  — legacy fast-converge for panels with no DA loaded.

The transcript is BOTH a debug artifact saved to ``conversation.json`` AND a
creative asset — the editor agent is allowed to lift verbatim lines into the
final script.
"""

from __future__ import annotations

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
DEFAULT_MAX_TURNS: int = 14
REPETITION_JACCARD: float = 0.78

DA_ROLE: str = "devil_advocate"
DA_SWAY_THRESHOLD_PP: int = 20
ROOM_SWAY_THRESHOLD_PP: int = 15
DA_OPENING_DELTA_PP: int = 35


@dataclass
class ConversationTurn:
    order: int
    persona_id: str
    text: str
    probability_pct: int | None
    asks_persona: str | None = None
    is_consensus_call: bool = False
    is_devil_advocate: bool = False

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
    devil_advocate_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel": self.panel,
            "turns": [t.as_dict() for t in self.turns],
            "consensus_probability_pct": self.consensus_probability_pct,
            "ended_reason": self.ended_reason,
            "moderator_take": self.moderator_take,
            "market_context_summary": self.market_context_summary,
            "devil_advocate_id": self.devil_advocate_id,
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


def _is_da(persona: dict[str, Any]) -> bool:
    return str(persona.get("role", "")).lower() == DA_ROLE


def _split_panel(
    panel: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    regulars = [p for p in panel if not _is_da(p)]
    da = next((p for p in panel if _is_da(p)), None)
    return regulars, da


def _median_int(xs: list[int]) -> int | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) // 2


def _last_prob_per_persona(turns: list[ConversationTurn]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in turns:
        if t.probability_pct is not None:
            out[t.persona_id] = t.probability_pct
    return out


def _room_median(
    turns: list[ConversationTurn], regular_ids: set[str]
) -> int | None:
    last = _last_prob_per_persona([t for t in turns if t.persona_id in regular_ids])
    return _median_int(list(last.values())) if len(last) >= 2 else None


def _check_sway(
    turns: list[ConversationTurn],
    *,
    da_id: str,
    regular_ids: set[str],
) -> tuple[str, int] | None:
    """Returns ``(reason, final_consensus_pct)`` once a sway condition fires, else None.

    Reasons: ``"da_swayed"`` | ``"room_swayed"``.
    """
    da_turns = [
        t for t in turns if t.persona_id == da_id and t.probability_pct is not None
    ]

    # DA side: did the devil's advocate move toward the room?
    if len(da_turns) >= 2:
        first_da = da_turns[0].probability_pct
        last_da = da_turns[-1].probability_pct
        room_at_last_da = _room_median(
            [t for t in turns if t.order < da_turns[-1].order], regular_ids
        )
        if (
            first_da is not None
            and last_da is not None
            and room_at_last_da is not None
        ):
            moved = last_da - first_da
            toward_room = room_at_last_da - first_da
            if (
                moved != 0
                and (moved > 0) == (toward_room > 0)
                and abs(moved) >= DA_SWAY_THRESHOLD_PP
            ):
                return ("da_swayed", room_at_last_da)

    # Room side: did the room move toward the DA after she joined?
    if da_turns:
        first_da_order = da_turns[0].order
        regs_before = [
            t
            for t in turns
            if t.persona_id in regular_ids and t.order < first_da_order
        ]
        regs_after = [
            t
            for t in turns
            if t.persona_id in regular_ids and t.order > first_da_order
        ]
        initial_median = _median_int(list(_last_prob_per_persona(regs_before).values()))
        current_median = _median_int(list(_last_prob_per_persona(regs_after).values()))
        last_da_prob = da_turns[-1].probability_pct
        if (
            initial_median is not None
            and current_median is not None
            and last_da_prob is not None
        ):
            moved = current_median - initial_median
            toward_da = last_da_prob - initial_median
            if (
                moved != 0
                and (moved > 0) == (toward_da > 0)
                and abs(moved) >= ROOM_SWAY_THRESHOLD_PP
            ):
                return ("room_swayed", last_da_prob)

    return None


def _devil_advocate_prefix(
    *, room_median: int | None, is_opening: bool
) -> str:
    if room_median is None:
        return (
            "DEVIL'S ADVOCATE BRIEF:\n"
            "You are this story's devil's advocate. The panel hasn't priced yet — "
            "open with a sharp, evidence-backed contrarian read so the others have "
            "something concrete to argue against.\n\n"
        )
    if is_opening:
        if room_median >= 50:
            target = max(3, room_median - DA_OPENING_DELTA_PP)
            hint = (
                f"opening probability MUST be ≤ {target}c (i.e. well below the room)."
            )
        else:
            target = min(97, room_median + DA_OPENING_DELTA_PP)
            hint = (
                f"opening probability MUST be ≥ {target}c (i.e. well above the room)."
            )
        return (
            "DEVIL'S ADVOCATE BRIEF (read FIRST, before persona card):\n"
            f"The panel has converged near {room_median}c. Argue the OPPOSITE "
            "direction with a SPECIFIC historical precedent or structural "
            "mechanism — not contrarianism for its own sake. Hold the position "
            "until the panel either sways you with new evidence, or you sway "
            "them.\n"
            f"On THIS opening turn: {hint}\n\n"
        )
    return (
        "DEVIL'S ADVOCATE BRIEF (read FIRST, before persona card):\n"
        f"The panel is currently around {room_median}c. Continue arguing the "
        "opposite direction with concrete precedent. Move ONLY if a panelist "
        "gives you a new mechanism — not because they repeated themselves "
        "louder.\n\n"
    )


def _turn_user_prompt(
    *,
    persona: dict[str, Any],
    summary: str,
    market_context: str,
    history: list[ConversationTurn],
    other_ids: list[str],
    devil_advocate_meta: dict[str, Any] | None = None,
) -> str:
    history_block = "\n".join(
        f"{t.persona_id}: {t.text} "
        + (f"[priced {t.probability_pct}c]" if t.probability_pct is not None else "")
        for t in history[-6:]
    ) or "(this is the opening turn)"
    prefix = ""
    if devil_advocate_meta is not None:
        prefix = _devil_advocate_prefix(
            room_median=devil_advocate_meta.get("room_median"),
            is_opening=bool(devil_advocate_meta.get("is_opening", False)),
        )
    return f"""{prefix}You are {persona['id']}. Card:
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
    devil_advocate_meta: dict[str, Any] | None = None,
) -> tuple[ConversationTurn, int]:
    s = get_settings()
    prompt = _turn_user_prompt(
        persona=persona,
        summary=summary,
        market_context=market_context,
        history=history,
        other_ids=other_ids,
        devil_advocate_meta=devil_advocate_meta,
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
            is_devil_advocate=devil_advocate_meta is not None,
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
    """Round-robin conversation with optional devil's advocate.

    Provider routing: Anthropic Sonnet when ``ANTHROPIC_API_KEY`` is present
    (preferred — better at sustained-voice persona writing), otherwise OpenAI
    ``gpt-4o`` with JSON-mode. Same prompt either way.

    A panelist with ``role: devil_advocate`` in its YAML card is treated
    specially: it gets an opening anchor 35pp away from the room median and a
    persona-aware prompt prefix on every turn. Termination flips from
    fast-consensus to a sway/stalemate check (see module docstring).
    """
    s = get_settings()
    ids = [p["id"] for p in panel_personas]
    regulars, da_persona = _split_panel(panel_personas)
    regular_ids = {p["id"] for p in regulars}
    da_id = da_persona["id"] if da_persona else None

    transcript = ConversationTranscript(
        panel=list(ids),
        market_context_summary=market_context[:400],
        devil_advocate_id=da_id,
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
        is_da_turn = persona["id"] == da_id
        da_meta: dict[str, Any] | None = None
        if is_da_turn:
            room_med = _room_median(transcript.turns, regular_ids)
            already_spoke = any(t.persona_id == da_id for t in transcript.turns)
            da_meta = {"room_median": room_med, "is_opening": not already_spoke}
        other_ids = [pid for pid in ids if pid != persona["id"]]
        turn, cents = await _one_turn(
            client,
            persona,
            summary=summary,
            market_context=market_context,
            history=transcript.turns,
            other_ids=other_ids,
            order=order,
            devil_advocate_meta=da_meta,
        )
        transcript.turns.append(turn)
        total_cents += cents
        order += 1
        if _detect_repetition(transcript.turns):
            transcript.ended_reason = "repetition"
            break
        if da_id:
            sway = _check_sway(
                transcript.turns, da_id=da_id, regular_ids=regular_ids
            )
            if sway is not None:
                reason, pct = sway
                transcript.ended_reason = reason
                transcript.consensus_probability_pct = pct
                break
        else:
            consensus = _detect_consensus(transcript.turns)
            if consensus is not None:
                transcript.consensus_probability_pct = consensus
                transcript.ended_reason = "consensus"
                break
    else:
        transcript.ended_reason = "stalemate" if da_id else "max_turns"
    if transcript.consensus_probability_pct is None:
        # Even on stalemate/max_turns/repetition we want ONE summary number.
        if da_id:
            transcript.consensus_probability_pct = _room_median(
                transcript.turns, regular_ids
            )
        else:
            probs = [
                t.probability_pct
                for t in transcript.turns
                if t.probability_pct is not None
            ]
            transcript.consensus_probability_pct = _median_int(probs)

    result = make_result(
        "conversation",
        model=model_name,
        provider=provider,
        input_summary=summary[:200],
        output_summary=json.dumps(
            {
                "turns": len(transcript.turns),
                "ended": transcript.ended_reason,
                "da": da_id,
            }
        ),
        spend_cents=total_cents,
        extras={"turn_count": len(transcript.turns), "devil_advocate_id": da_id},
    )
    return transcript, result
