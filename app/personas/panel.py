from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

import yaml
from anthropic import AsyncAnthropic
from anthropic import APIStatusError as AnthropicAPIStatusError

from app.core.config import get_settings
from app.core.jsonx import extract_json_object
from app.core.models import PanelVerdict, PersonaReaction


async def _create_with_retry(
    client: AsyncAnthropic,
    *,
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    attempts: int = 4,
) -> Any:
    delay = 0.8
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return await client.messages.create(
                model=model, max_tokens=max_tokens, messages=messages
            )
        except AnthropicAPIStatusError as e:
            last = e
            code = getattr(e, "status_code", None)
            if code in (429, 500, 502, 503, 504, 529):
                await asyncio.sleep(delay + random.random() * 0.3)
                delay = min(delay * 2.0, 8.0)
                continue
            raise
        except Exception as e:
            last = e
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 8.0)
    assert last is not None
    raise last


def load_personas(path: Path | None = None) -> list[dict[str, str]]:
    """Load personas: prefer per-card files in app/personas/cards/*.yml; fall back to legacy personas.yml at repo root."""
    s = get_settings()
    if path is not None:
        data = yaml.safe_load(path.read_text()) or {}
        return list(data.get("personas", []))
    cards_dir = Path(__file__).resolve().parent / "cards"
    out: list[dict[str, str]] = []
    if cards_dir.is_dir():
        for f in sorted(cards_dir.glob("*.yml")):
            try:
                card = yaml.safe_load(f.read_text()) or {}
                if isinstance(card, dict) and card.get("id"):
                    out.append(card)
            except Exception:
                continue
    if out:
        return out
    legacy = s.project_root / "personas.yml"
    if legacy.exists():
        data = yaml.safe_load(legacy.read_text()) or {}
        return list(data.get("personas", []))
    return []


PANEL_PROMPT = """You are simulating a real person with the demographic below.
Given the NEWS SUMMARY, respond in JSON only:
{{
  "persona_id": "{pid}",
  "would_stop_scroll": true or false,
  "stop_probability": 0.0 to 1.0,
  "would_share": true or false,
  "would_comment": true or false,
  "would_save": true or false,
  "hook_suggestion": "≤16 words — MUST cite a concrete noun/entity directly present in NEWS SUMMARY (no generic clichés)",
  "angle_of_interest": "short string tying that hook to THEIR identity",
  "emotional_response": "short string",
  "cliches_to_avoid": ["string"]
}}

NEWS SUMMARY:
{summary}

PERSONA:
{card}
"""


async def _one_persona(
    client: AsyncAnthropic, pid: str, card: str, summary: str, model: str
) -> PersonaReaction:
    s = get_settings()
    if s.dry_run:
        return PersonaReaction(
            persona_id=pid,
            would_stop_scroll=True,
            stop_probability=0.7,
            would_share=True,
            would_comment=False,
            would_save=False,
            hook_suggestion="dry run",
            angle_of_interest="dry run",
            emotional_response="neutral",
            cliches_to_avoid=[],
        )
    msg = await _create_with_retry(
        client,
        model=model,
        max_tokens=600,
        messages=[
            {
                "role": "user",
                "content": PANEL_PROMPT.format(
                    summary=summary[:4000], card=card, pid=pid
                ),
            }
        ],
    )
    text = ""
    for b in msg.content:
        if b.type == "text":
            text += b.text
    data = extract_json_object(text)
    data["persona_id"] = pid
    return PersonaReaction.model_validate(data)


async def run_initial_panel(summary: str) -> PanelVerdict:
    s = get_settings()
    personas = load_personas()
    akey = s.anthropic_api_key or "sk-ant-dry-run-localxxxxxxxxxxxxxxxxx"
    client = AsyncAnthropic(api_key=akey)
    model = s.anthropic_model_sonnet
    tasks = [
        _one_persona(
            client,
            p["id"],
            p.get("card", ""),
            summary,
            model,
        )
        for p in personas
    ]
    reactions = await asyncio.gather(*tasks)
    rs = list(reactions)
    stop_scores = [r.stop_probability for r in rs]
    overall = sum(stop_scores) / max(len(stop_scores), 1)
    breadth = sum(1 for r in rs if r.would_stop_scroll) / max(len(rs), 1)
    hooks = [r.hook_suggestion for r in rs if r.hook_suggestion][:5]
    avoid: list[str] = []
    for r in rs:
        avoid.extend(r.cliches_to_avoid)
    return PanelVerdict(
        reactions=rs,
        overall_stop_score=overall,
        breadth=breadth,
        cross_demographic_hooks=hooks,
        avoid_list=list(dict.fromkeys(avoid))[:20],
    )


async def run_panel_rerate(summary: str, draft_prompt: str) -> dict[str, Any]:
    """Second panel: judge the draft prompt."""
    s = get_settings()
    personas = load_personas()
    akey = s.anthropic_api_key or "sk-ant-dry-run-localxxxxxxxxxxxxxxxxx"
    client = AsyncAnthropic(api_key=akey)
    model = s.anthropic_model_sonnet

    async def one(p: dict[str, str]) -> dict[str, Any]:
        pid = p["id"]
        if s.dry_run:
            return {
                "persona_id": pid,
                "would_stop": True,
                "would_share": True,
                "would_comment": False,
                "would_save": False,
                "objections": [],
                "improvement": "",
            }
        pr = f"""NEWS SUMMARY:\n{summary[:2000]}\n\nVIDEO PROMPT DRAFT:\n{draft_prompt[:4000]}\n\n
Would THIS specific prompt make YOU stop scrolling, share, comment, save?
JSON only:
{{"persona_id":"{pid}","would_stop":bool,"would_share":bool,"would_comment":bool,"would_save":bool,
"objections":[],"improvement":""}}\nPERSONA:\n{p.get("card","")}"""
        msg = await _create_with_retry(
            client,
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": pr}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        data = extract_json_object(text)
        data["persona_id"] = pid
        return data

    rows = await asyncio.gather(*[one(p) for p in personas])
    return {"reactions": rows}
