"""LLM-judge: declarative voice on the README + JJJ output.

Scores three axes:

- ``hook_punch`` — is the hook ≤ 100 chars, declarative, leads with the
  verb, no hedging ("may", "could", "might")? 10 if perfect, ≤ 4 if
  it's a wire-service rewrite.
- ``higgsfield_voice`` — does the higgsfield prompt commit to a single
  framing and lean into it? Penalize hedged "either X or Y" phrasings.
- ``readme_freshness`` — does the README's prose feel current
  (forward-looking horizons, no obvious training-data year drift)?
  Penalize "in 2023" / "in 2024" references that don't make sense.

Returns one ``EvalScore`` with the three subscores.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import openai_chat_with_retry
from app.agents.schemas import StructuredOutputError, parse_structured
from app.core.config import get_settings
from evals import EvalScore


JUDGE_MODEL = "gpt-4o"


class _Out(BaseModel):
    model_config = ConfigDict(extra="ignore")
    hook_punch: float = Field(..., ge=0.0, le=10.0)
    higgsfield_voice: float = Field(..., ge=0.0, le=10.0)
    readme_freshness: float = Field(..., ge=0.0, le=10.0)
    rationale: str = ""


_SYSTEM = (
    "You are an opinionated tabloid news editor evaluating event briefs. "
    "Score three axes 0-10. Be harsh: a 7 is a 'meets bar' brief; 10 is "
    'rare. Return STRICT JSON: {"hook_punch": float, '
    '"higgsfield_voice": float, "readme_freshness": float, "rationale": "..."}.'
)


def _user_prompt(readme_text: str, hook: str, higgsfield_prompt: str) -> str:
    return (
        f"HOOK ({len(hook)} chars):\n{hook}\n\n"
        f"HIGGSFIELD PROMPT:\n{higgsfield_prompt}\n\n"
        f"README (truncated):\n{readme_text[:6000]}\n\n"
        "Score:\n"
        "- hook_punch — ≤ 100 chars, declarative, leads with the verb, "
        "no hedging.\n"
        "- higgsfield_voice — commits to a single framing, no 'either/or' "
        "hedging.\n"
        "- readme_freshness — forward-looking horizons; flag any obvious "
        "training-data year drift ('in 2023' references that don't match "
        "the story's timeline).\n"
    )


async def score_event_folder(folder: Path) -> EvalScore:
    s = get_settings()
    if not s.has_openai:
        return EvalScore(
            name="style",
            overall=0.0,
            rationale="OPENAI_API_KEY not set — eval skipped.",
        )
    readme = folder / "README.md"
    higgs = folder / "higgsfield.json"
    if not readme.exists():
        return EvalScore(
            name="style",
            overall=0.0,
            rationale=f"no README.md at {folder}",
        )
    readme_text = readme.read_text(encoding="utf-8")
    hook = readme_text.splitlines()[0].lstrip("# ").strip()
    higgsfield_prompt = ""
    if higgs.exists():
        try:
            data = json.loads(higgs.read_text(encoding="utf-8"))
            higgsfield_prompt = str(data.get("prompt") or "")
        except json.JSONDecodeError:
            pass

    client = AsyncOpenAI(api_key=s.openai_api_key)
    r = await openai_chat_with_retry(
        client,
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _user_prompt(readme_text, hook, higgsfield_prompt)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    text = r.choices[0].message.content or "{}"
    try:
        parsed = parse_structured(text, _Out)
    except StructuredOutputError as exc:
        return EvalScore(
            name="style",
            overall=0.0,
            rationale=f"judge schema validation failed: {exc}"[:300],
        )
    assert isinstance(parsed, _Out)
    overall = (
        parsed.hook_punch + parsed.higgsfield_voice + parsed.readme_freshness
    ) / 3.0
    return EvalScore(
        name="style",
        overall=round(overall, 2),
        subscores={
            "hook_punch": parsed.hook_punch,
            "higgsfield_voice": parsed.higgsfield_voice,
            "readme_freshness": parsed.readme_freshness,
        },
        rationale=parsed.rationale,
        raw=parsed.model_dump(),
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target is None:
        out = get_settings().output_dir
        candidates = sorted(
            (p for p in out.iterdir() if p.is_dir() and (p / "README.md").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        target = candidates[0] if candidates else None
    if target is None:
        print("no event folder found")
        raise SystemExit(2)
    score = asyncio.run(score_event_folder(target))
    print(json.dumps(score.as_dict(), indent=2, ensure_ascii=False))
