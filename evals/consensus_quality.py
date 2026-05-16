"""LLM-judge: convergence, calibration, persona fidelity.

Given one ``conversation.json`` produced by the pipeline, score:

- ``convergence`` — did the panel actually move toward each other? Was
  the agreement round earned or rubber-stamped?
- ``calibration`` — does the final consensus probability reflect the
  evidence in the transcript? Penalize anchored-on-the-headline outputs
  and reward turns that update on real new info.
- ``persona_fidelity`` — does each panelist sound like themselves?
  Penalize same-voice drift (the "everyone-talks-like-Devon" failure).

Returns an ``EvalScore`` per axis plus an ``overall`` mean.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.agents.base import openai_chat_with_retry
from app.agents.schemas import StructuredOutputError, parse_structured
from app.core.config import get_settings
from evals import EvalScore

from pydantic import BaseModel, ConfigDict, Field


JUDGE_MODEL = "gpt-4o"


class _Out(BaseModel):
    model_config = ConfigDict(extra="ignore")
    convergence: float = Field(..., ge=0.0, le=10.0)
    calibration: float = Field(..., ge=0.0, le=10.0)
    persona_fidelity: float = Field(..., ge=0.0, le=10.0)
    rationale: str = ""


_SYSTEM = (
    "You are a senior pollster who judges Polymarket-style probability "
    "panels. You read a panel transcript and score three axes 0-10. "
    "Return STRICT JSON with keys convergence, calibration, "
    "persona_fidelity (each float 0-10) and rationale (≤ 4 sentences)."
)


def _user_prompt(conversation: dict[str, Any]) -> str:
    panel = conversation.get("panel") or []
    da = conversation.get("devil_advocate_id") or "—"
    ended = conversation.get("ended_reason") or "unknown"
    consensus = conversation.get("consensus_probability_pct")
    turns = conversation.get("turns") or []

    rendered: list[str] = []
    for t in turns:
        pid = t.get("persona_id", "?")
        pct = t.get("probability_pct")
        pct_s = f" [{pct}c]" if pct is not None else ""
        da_s = " (DA)" if t.get("is_devil_advocate") else ""
        ag_s = " (AG)" if t.get("is_agreement_turn") else ""
        rendered.append(f"- {pid}{da_s}{ag_s}{pct_s}: {(t.get('text') or '').strip()[:320]}")
    transcript = "\n".join(rendered)

    return (
        f"PANEL: {', '.join(panel)}\n"
        f"DEVIL'S ADVOCATE: {da}\n"
        f"ENDED_REASON: {ended}\n"
        f"FINAL CONSENSUS: {consensus}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        "Score:\n"
        "- convergence — did the panel actually move toward each other? "
        "Was the agreement round earned (with new info) or rubber-stamped?\n"
        "- calibration — does the final consensus probability reflect the "
        "evidence in the transcript? Penalize anchored-on-headline takes; "
        "reward turns that updated on real new info.\n"
        "- persona_fidelity — does each panelist sound like themselves? "
        "Penalize same-voice drift.\n\n"
        "Return STRICT JSON: "
        '{"convergence": float, "calibration": float, '
        '"persona_fidelity": float, "rationale": "..."}'
    )


async def score_conversation(conversation: dict[str, Any]) -> EvalScore:
    s = get_settings()
    if not s.has_openai:
        return EvalScore(
            name="consensus_quality",
            overall=0.0,
            rationale="OPENAI_API_KEY not set — eval skipped.",
        )
    client = AsyncOpenAI(api_key=s.openai_api_key)
    r = await openai_chat_with_retry(
        client,
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _user_prompt(conversation)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    text = r.choices[0].message.content or "{}"
    try:
        parsed = parse_structured(text, _Out)
    except StructuredOutputError as exc:
        return EvalScore(
            name="consensus_quality",
            overall=0.0,
            rationale=f"judge schema validation failed: {exc}"[:300],
        )
    assert isinstance(parsed, _Out)
    overall = (parsed.convergence + parsed.calibration + parsed.persona_fidelity) / 3.0
    return EvalScore(
        name="consensus_quality",
        overall=round(overall, 2),
        subscores={
            "convergence": parsed.convergence,
            "calibration": parsed.calibration,
            "persona_fidelity": parsed.persona_fidelity,
        },
        rationale=parsed.rationale,
        raw=parsed.model_dump(),
    )


async def score_event_folder(folder: Path) -> EvalScore:
    """Score the ``conversation.json`` inside an event folder."""
    convo = folder / "conversation.json"
    if not convo.exists():
        return EvalScore(
            name="consensus_quality",
            overall=0.0,
            rationale=f"no conversation.json at {convo}",
        )
    data = json.loads(convo.read_text(encoding="utf-8"))
    return await score_conversation(data)


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target is None:
        out = get_settings().output_dir
        candidates = sorted(
            (p for p in out.iterdir() if p.is_dir() and (p / "conversation.json").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        target = candidates[0] if candidates else None
    if target is None:
        print("no event folder found")
        raise SystemExit(2)
    score = asyncio.run(score_event_folder(target))
    print(json.dumps(score.as_dict(), indent=2, ensure_ascii=False))
