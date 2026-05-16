"""Round-robin persona conversation with a devil's advocate.

Two-phase state machine so debates actually go deep instead of dying after
3 turns when somebody nods at the contrarian:

1. **Debate phase** — every persona (regulars + DA) round-robins until BOTH:
   - each has spoken at least ``MIN_TURNS_PER_PERSONA`` times (default 3),
     so a 5-person panel runs at least 15 debate turns
   - the last-priced probability per persona has spread ≤ ``CONSENSUS_SPREAD_CENTS``
     (default 10pp)
2. **Agreement phase** — one explicit final round, panel order with the DA
   LAST so they're the most-tested holdout. Each persona answers a separate
   "lock-in or break consensus" prompt and pins a final probability.

End conditions:
- ``consensus``  — every persona completed their agreement turn AND every
                   agreement-round pct is within ``AGREEMENT_TOLERANCE_PP``
                   of the converged median.
- ``stalemate``  — agreement broke once, re-entry to debate burned, and
                   the next agreement still failed; OR ``max_turns`` hit
                   before agreement phase even started.

If an agreement turn falls outside the converged band, the conversation
**re-enters debate** for at least ``RE_ENTRY_DEBATE_TURNS`` more turns, then
re-attempts agreement. Only ONE re-entry is allowed; a second failure
collapses to stalemate.

The devil's advocate is NOT a dedicated persona — it's a role assigned at
runtime to one of the existing panelists, chosen deterministically by
hashing the event id so re-runs of the same cluster reproduce the same DA.
The chosen panelist keeps their normal voice; we only bolt on a "for THIS
debate, argue the opposite direction" brief in the prompt.

``convergence_note`` is descriptive narrative ("room moved from 65c to 38c
toward the DA's 25c anchor"); it does NOT drive control flow.

The transcript is BOTH a debug artifact saved to ``conversation.json`` AND a
creative asset — the editor agent is allowed to lift verbatim lines into the
final script.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

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
from app.agents.schemas import PanelTurnOut, StructuredOutputError, parse_structured
from app.core.config import get_settings
from app.prompts import load_prompt


# Bumped from 50 → 60 so the agreement round + an optional re-entry cycle
# always fit even when the debate ran long. `_ready_for_agreement` also
# refuses to enter the agreement phase if there's no budget left for a
# full panel-sized agreement round (see below).
DEFAULT_MAX_TURNS: int = 60

MIN_TURNS_PER_PERSONA: int = 3      # each persona must speak this many debate turns
CONSENSUS_SPREAD_CENTS: int = 10    # last-round pct max - min must be ≤ this to enter agreement
AGREEMENT_TOLERANCE_PP: int = 5     # agreement turn must be within this of converged median
RE_ENTRY_DEBATE_TURNS: int = 5      # if agreement breaks, debate at least this much before retry

REPETITION_JACCARD: float = 0.85
REPETITION_CONSECUTIVE: int = 3     # need 3 consecutive high-Jaccard pairs to fire

DA_OPENING_DELTA_PP: int = 35

HISTORY_TURNS_VISIBLE: int = 10     # show the LLM this many prior turns each call


@dataclass
class ConversationTurn:
    order: int
    persona_id: str
    text: str
    probability_pct: int | None
    asks_persona: str | None = None
    is_consensus_call: bool = False
    is_devil_advocate: bool = False
    is_agreement_turn: bool = False

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
    phase: str = "debate"
    convergence_note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel": self.panel,
            "turns": [t.as_dict() for t in self.turns],
            "consensus_probability_pct": self.consensus_probability_pct,
            "ended_reason": self.ended_reason,
            "moderator_take": self.moderator_take,
            "market_context_summary": self.market_context_summary,
            "devil_advocate_id": self.devil_advocate_id,
            "phase": self.phase,
            "convergence_note": self.convergence_note,
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
    """Fire only when ``REPETITION_CONSECUTIVE`` consecutive pairs all repeat.

    The previous "single-pair > 0.78 Jaccard" heuristic killed legitimate
    rebuttals where two personas reused the same key terminology. We now
    need a sustained pattern of near-identical messages back-to-back-to-back
    before declaring groupthink.
    """
    window = REPETITION_CONSECUTIVE + 1
    if len(turns) < window:
        return False
    tail = turns[-window:]
    pairs = list(zip(tail[:-1], tail[1:]))
    return all(_jaccard(a.text, b.text) >= REPETITION_JACCARD for a, b in pairs)


def _pick_devil_advocate(
    panel: list[dict[str, Any]], *, event_id: str | None
) -> dict[str, Any] | None:
    """Pick exactly one panelist to play devil's advocate for THIS debate.

    Selection is deterministic when ``event_id`` is provided (so re-running
    the same cluster produces the same DA), and uniformly random otherwise.
    Returns ``None`` for panels of size < 2 — you need at least one regular
    and one DA for the dynamic to make sense.
    """
    if len(panel) < 2:
        return None
    if event_id:
        seed = int(hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
    else:
        rng = random.Random()
    return rng.choice(panel)


def _persona_summary(persona: dict[str, Any]) -> str:
    """One-line distillation of the persona for the DA brief."""
    card = str(persona.get("card") or "").strip()
    first = re.split(r"(?<=[.!?])\s", card, maxsplit=1)[0]
    return " ".join(first.split())[:240] or persona.get("id", "panelist")


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


def _partial_room_median(
    turns: list[ConversationTurn], regular_ids: set[str]
) -> int | None:
    """Median across whoever has spoken so far — accepts a single voice.

    Used ONLY for the DA's opening-anchor calculation, where we want to
    anchor against whatever the room has already said even if only one
    regular has priced.
    """
    last = _last_prob_per_persona([t for t in turns if t.persona_id in regular_ids])
    if not last:
        return None
    return _median_int(list(last.values()))


def _build_agreement_order(
    panel_personas: list[dict[str, Any]], da_id: str | None
) -> list[dict[str, Any]]:
    """Panel order with DA last — they're the holdout we want most tested."""
    da = None
    rest: list[dict[str, Any]] = []
    for p in panel_personas:
        if p["id"] == da_id:
            da = p
        else:
            rest.append(p)
    return rest + ([da] if da else [])


def _debate_turns(turns: list[ConversationTurn]) -> list[ConversationTurn]:
    return [t for t in turns if not t.is_agreement_turn]


def _agreement_turns(turns: list[ConversationTurn]) -> list[ConversationTurn]:
    return [t for t in turns if t.is_agreement_turn]


def _ready_for_agreement(
    turns: list[ConversationTurn],
    *,
    panel_ids: set[str],
    min_debate_total: int,
    re_entry_debate_threshold: int | None,
    max_turns: int,
) -> bool:
    """True iff debate-phase conditions for advancing to agreement are met.

    Requires every persona to have spoken ``MIN_TURNS_PER_PERSONA`` times in
    the debate phase AND the most-recent-priced pct across all personas to
    have spread ≤ ``CONSENSUS_SPREAD_CENTS``. After re-entry, also requires
    at least ``RE_ENTRY_DEBATE_TURNS`` new debate turns since re-entry.

    Also refuses to advance when there isn't enough total-turn budget left
    to complete the full agreement round (so consensus can't fire at turn
    49 of 50 with a 5-person panel, only to be coerced to ``stalemate``).
    """
    debate = _debate_turns(turns)
    if len(debate) < min_debate_total:
        return False
    if re_entry_debate_threshold is not None and len(debate) < re_entry_debate_threshold:
        return False
    if max_turns - len(turns) < len(panel_ids):
        return False

    counts: dict[str, int] = {}
    for t in debate:
        counts[t.persona_id] = counts.get(t.persona_id, 0) + 1
    if any(counts.get(pid, 0) < MIN_TURNS_PER_PERSONA for pid in panel_ids):
        return False

    last = _last_prob_per_persona(debate)
    if len(last) < len(panel_ids):
        return False
    vals = list(last.values())
    return (max(vals) - min(vals)) <= CONSENSUS_SPREAD_CENTS


def _converged_median(turns: list[ConversationTurn], panel_ids: set[str]) -> int | None:
    """Median of each persona's most-recent debate-phase pct."""
    last = _last_prob_per_persona(_debate_turns(turns))
    relevant = [v for k, v in last.items() if k in panel_ids]
    return _median_int(relevant) if relevant else None


def _agreement_held(
    ag_turns: list[ConversationTurn],
    *,
    panel_ids: set[str],
    converged_median: int | None,
) -> bool:
    """All agreement turns landed within ``AGREEMENT_TOLERANCE_PP`` of the band."""
    if converged_median is None:
        return False
    if len({t.persona_id for t in ag_turns}) < len(panel_ids):
        return False
    for t in ag_turns:
        if t.probability_pct is None:
            return False
        if abs(t.probability_pct - converged_median) > AGREEMENT_TOLERANCE_PP:
            return False
    return True


def _convergence_note(
    turns: list[ConversationTurn],
    *,
    regular_ids: set[str],
    da_id: str | None,
) -> str | None:
    """Render a one-liner describing the trajectory of the debate.

    Heuristic only: first vs last priced pct per persona, room-median vs
    DA-anchor delta. Used in the README for narrative color — never gates
    control flow.
    """
    if not turns:
        return None
    debate = _debate_turns(turns)
    first_p: dict[str, int] = {}
    last_p: dict[str, int] = {}
    for t in debate:
        if t.probability_pct is None:
            continue
        first_p.setdefault(t.persona_id, t.probability_pct)
        last_p[t.persona_id] = t.probability_pct
    if not last_p:
        return None
    reg_first = [v for k, v in first_p.items() if k in regular_ids]
    reg_last = [v for k, v in last_p.items() if k in regular_ids]
    room_first = _median_int(reg_first)
    room_last = _median_int(reg_last)
    da_first = first_p.get(da_id) if da_id else None
    da_last = last_p.get(da_id) if da_id else None

    parts: list[str] = []
    if room_first is not None and room_last is not None:
        if abs(room_last - room_first) >= 3:
            parts.append(f"room moved from {room_first}c to {room_last}c")
        else:
            parts.append(f"room held near {room_first}c")
    if da_id and da_first is not None and da_last is not None:
        if abs(da_last - da_first) >= 3:
            parts.append(f"DA ({da_id}) moved from {da_first}c to {da_last}c")
        else:
            parts.append(f"DA ({da_id}) held near {da_first}c")
    if not parts:
        return None
    final_med = _median_int(list(last_p.values()))
    if final_med is not None:
        parts.append(f"final read {final_med}c")
    return "; ".join(parts)


def _devil_advocate_prefix(
    *,
    persona: dict[str, Any],
    room_median: int | None,
    is_opening: bool,
) -> str:
    """Render the DA framing block for a panelist drawn from the regular pool."""
    persona_summary = _persona_summary(persona)
    persona_id = persona.get("id", "panelist")
    if room_median is None:
        return load_prompt(
            "conversation.devil_advocate.unpriced",
            persona_summary=persona_summary,
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
        return load_prompt(
            "conversation.devil_advocate.opening",
            persona_id=persona_id,
            persona_summary=persona_summary,
            room_median=room_median,
            hint=hint,
        )
    return load_prompt(
        "conversation.devil_advocate.followup",
        persona_id=persona_id,
        persona_summary=persona_summary,
        room_median=room_median,
    )


def _agreement_prefix(*, median: int, turn_count: int) -> str:
    return load_prompt(
        "conversation.agreement", median=median, turn_count=turn_count
    )


def _turn_user_prompt(
    *,
    persona: dict[str, Any],
    summary: str,
    market_context: str,
    history: list[ConversationTurn],
    other_ids: list[str],
    devil_advocate_meta: dict[str, Any] | None = None,
    agreement_meta: dict[str, Any] | None = None,
) -> str:
    history_block = "\n".join(
        f"{t.persona_id}: {t.text} "
        + (f"[priced {t.probability_pct}c]" if t.probability_pct is not None else "")
        for t in history[-HISTORY_TURNS_VISIBLE:]
    ) or "(this is the opening turn)"
    prefix = ""
    if agreement_meta is not None:
        prefix = _agreement_prefix(
            median=int(agreement_meta["median"]),
            turn_count=int(agreement_meta["turn_count"]),
        )
    elif devil_advocate_meta is not None:
        prefix = _devil_advocate_prefix(
            persona=persona,
            room_median=devil_advocate_meta.get("room_median"),
            is_opening=bool(devil_advocate_meta.get("is_opening", False)),
        )
    return load_prompt(
        "conversation.turn",
        prefix=prefix,
        persona_id=persona["id"],
        persona_card=persona.get("card", ""),
        persona_dialect=persona.get("dialect", ""),
        persona_bias_targets=persona.get("bias_targets", []),
        persona_betting_voice=persona.get("betting_voice", ""),
        other_ids=other_ids,
        summary=summary[:3500],
        market_context=market_context.strip(),
        history_block=history_block,
    )


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
    agreement_meta: dict[str, Any] | None = None,
    is_da: bool = False,
) -> tuple[ConversationTurn, int]:
    s = get_settings()
    prompt = _turn_user_prompt(
        persona=persona,
        summary=summary,
        market_context=market_context,
        history=history,
        other_ids=other_ids,
        devil_advocate_meta=devil_advocate_meta,
        agreement_meta=agreement_meta,
    )
    if isinstance(client, AsyncAnthropic):
        msg = await anthropic_call_with_retry(
            client,
            primary_model=s.anthropic_model_sonnet,
            fallback_model=s.anthropic_model_opus,
            max_tokens=400,
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

    try:
        parsed = parse_structured(text, PanelTurnOut)
        body = parsed.text.strip()
        prob = parsed.probability_pct
        asks = parsed.asks_persona
    except StructuredOutputError:
        # If the LLM drifted off-schema, salvage the text and re-parse a
        # probability from it. We never want one bad turn to blow up the panel.
        body = (text or "").strip()
        prob = None
        asks = None
    if prob is None:
        prob = parse_probability(body)
    cents = rough_cost_cents(tin, tout, classify_model(model_for_cost))
    return (
        ConversationTurn(
            order=order,
            persona_id=persona["id"],
            text=body,
            probability_pct=prob,
            asks_persona=asks if asks in other_ids else None,
            is_devil_advocate=is_da,
            is_agreement_turn=agreement_meta is not None,
        ),
        cents,
    )


TurnCallback = Callable[[ConversationTurn, str, int, int], None]
"""Per-turn progress callback: ``(turn, phase, idx, total_so_far)``.

``phase`` is "debate" or "agreement". ``idx`` is the 1-indexed position
within that phase. ``total_so_far`` is ``len(transcript.turns)`` after the
turn is appended. Callbacks must be lightweight and non-blocking (the
pipeline uses them for stdout progress logs).
"""


async def run_conversation(
    panel_personas: list[dict[str, Any]],
    *,
    summary: str,
    market_context: str = "",
    max_turns: int = DEFAULT_MAX_TURNS,
    event_id: str | None = None,
    on_turn: TurnCallback | None = None,
) -> tuple[ConversationTranscript, AgentResult]:
    """Two-phase conversation: debate → agreement → consensus | stalemate.

    Provider routing: Anthropic Sonnet when ``ANTHROPIC_API_KEY`` is present
    (preferred — better at sustained-voice persona writing), otherwise OpenAI
    ``gpt-4o`` with JSON-mode. Same prompts either way.

    One panelist is drawn at runtime to play devil's advocate. With
    ``event_id`` set, the pick is deterministic (re-runs of the same cluster
    produce the same DA). The DA keeps their normal voice; the framing
    prompt tilts them contrarian.

    See the module docstring for end-condition details.
    """
    s = get_settings()
    ids = [p["id"] for p in panel_personas]
    panel_ids = set(ids)
    da_persona = _pick_devil_advocate(panel_personas, event_id=event_id)
    da_id = da_persona["id"] if da_persona else None
    regular_ids = {p["id"] for p in panel_personas if p["id"] != da_id}

    transcript = ConversationTranscript(
        panel=list(ids),
        market_context_summary=market_context[:400],
        devil_advocate_id=da_id,
    )
    provider = "anthropic" if s.has_anthropic else "openai"
    model_name = s.anthropic_model_sonnet if s.has_anthropic else s.openai_model_top
    if not panel_personas:
        transcript.ended_reason = "no_panel"
        transcript.phase = "ended"
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

    min_debate_total = MIN_TURNS_PER_PERSONA * len(panel_personas)
    total_cents = 0
    order = 0
    debate_turn_count = 0
    phase = "debate"
    re_entry_used = False
    re_entry_debate_threshold: int | None = None
    converged_median: int | None = None
    agreement_queue: list[dict[str, Any]] = []

    while len(transcript.turns) < max_turns and phase != "ended":
        if phase == "debate":
            persona = panel_personas[debate_turn_count % len(panel_personas)]
            is_da_turn = persona["id"] == da_id
            da_meta: dict[str, Any] | None = None
            if is_da_turn:
                room_med = _partial_room_median(
                    _debate_turns(transcript.turns), regular_ids
                )
                already_spoke_in_debate = any(
                    t.persona_id == da_id and not t.is_agreement_turn
                    for t in transcript.turns
                )
                da_meta = {
                    "room_median": room_med,
                    "is_opening": not already_spoke_in_debate,
                }
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
                is_da=is_da_turn,
            )
            transcript.turns.append(turn)
            debate_turn_count += 1
            total_cents += cents
            order += 1
            if on_turn is not None:
                try:
                    on_turn(turn, "debate", debate_turn_count, len(transcript.turns))
                except Exception:
                    # progress callbacks must never break the conversation
                    pass

            if _detect_repetition(transcript.turns):
                # 3 consecutive near-identical turns is genuine groupthink;
                # let the agreement phase still try to wrap it up if we're
                # already past the min-turns floor.
                if _ready_for_agreement(
                    transcript.turns,
                    panel_ids=panel_ids,
                    min_debate_total=min_debate_total,
                    re_entry_debate_threshold=re_entry_debate_threshold,
                    max_turns=max_turns,
                ):
                    converged_median = _converged_median(transcript.turns, panel_ids)
                    phase = "agreement"
                    agreement_queue = _build_agreement_order(panel_personas, da_id)
                    continue
                transcript.ended_reason = "repetition"
                phase = "ended"
                break

            if _ready_for_agreement(
                transcript.turns,
                panel_ids=panel_ids,
                min_debate_total=min_debate_total,
                re_entry_debate_threshold=re_entry_debate_threshold,
                max_turns=max_turns,
            ):
                converged_median = _converged_median(transcript.turns, panel_ids)
                phase = "agreement"
                agreement_queue = _build_agreement_order(panel_personas, da_id)
        else:
            if agreement_queue:
                persona = agreement_queue.pop(0)
                is_da_turn = persona["id"] == da_id
                other_ids = [pid for pid in ids if pid != persona["id"]]
                turn, cents = await _one_turn(
                    client,
                    persona,
                    summary=summary,
                    market_context=market_context,
                    history=transcript.turns,
                    other_ids=other_ids,
                    order=order,
                    agreement_meta={
                        "median": converged_median or 50,
                        "turn_count": len(transcript.turns),
                    },
                    is_da=is_da_turn,
                )
                transcript.turns.append(turn)
                total_cents += cents
                order += 1
                if on_turn is not None:
                    try:
                        ag_idx = len(_agreement_turns(transcript.turns))
                        on_turn(turn, "agreement", ag_idx, len(transcript.turns))
                    except Exception:
                        pass
            else:
                ag = _agreement_turns(transcript.turns)
                # Only count the MOST RECENT agreement attempt (post any re-entry).
                # After re-entry there will be exactly len(panel) fresh ag turns.
                latest_ag = ag[-len(panel_personas):]
                if _agreement_held(
                    latest_ag,
                    panel_ids=panel_ids,
                    converged_median=converged_median,
                ):
                    pcts = [
                        t.probability_pct
                        for t in latest_ag
                        if t.probability_pct is not None
                    ]
                    transcript.consensus_probability_pct = _median_int(pcts)
                    transcript.ended_reason = "consensus"
                    phase = "ended"
                elif not re_entry_used:
                    re_entry_used = True
                    re_entry_debate_threshold = (
                        len(_debate_turns(transcript.turns)) + RE_ENTRY_DEBATE_TURNS
                    )
                    phase = "debate"
                else:
                    transcript.ended_reason = "stalemate"
                    phase = "ended"

    if phase != "ended":
        # We exited via max_turns without locking in.
        transcript.ended_reason = (
            transcript.ended_reason
            if transcript.ended_reason not in ("unknown", "")
            else "stalemate"
        )
        phase = "ended"

    transcript.phase = phase
    transcript.convergence_note = _convergence_note(
        transcript.turns, regular_ids=regular_ids, da_id=da_id
    )

    if transcript.consensus_probability_pct is None:
        # Always report a number for the brief, even when we couldn't lock in.
        ag = _agreement_turns(transcript.turns)
        latest_ag = ag[-len(panel_personas):] if ag else []
        ag_pcts = [t.probability_pct for t in latest_ag if t.probability_pct is not None]
        if ag_pcts:
            transcript.consensus_probability_pct = _median_int(ag_pcts)
        else:
            last = _last_prob_per_persona(_debate_turns(transcript.turns))
            if last:
                transcript.consensus_probability_pct = _median_int(list(last.values()))

    result = make_result(
        "conversation",
        model=model_name,
        provider=provider,
        input_summary=summary[:200],
        output_summary=json.dumps(
            {
                "turns": len(transcript.turns),
                "ended": transcript.ended_reason,
                "phase": transcript.phase,
                "da": da_id,
                "agreement_turns": len(_agreement_turns(transcript.turns)),
            }
        ),
        spend_cents=total_cents,
        extras={
            "turn_count": len(transcript.turns),
            "debate_turn_count": len(_debate_turns(transcript.turns)),
            "agreement_turn_count": len(_agreement_turns(transcript.turns)),
            "devil_advocate_id": da_id,
            "re_entry_used": re_entry_used,
        },
    )
    return transcript, result
