"""Agent loader — reads ``app/agents/<id>/persona.md`` and ``skill.md``.

Each agent folder under ``app/agents/`` contains:

- ``persona.md`` — YAML frontmatter (id, dialect, bias_targets,
  betting_voice, ...) + the natural-language card in the body
- ``skill.md`` — natural-language description of what they argue
  best, plus voice tics and weaknesses

This loader returns the agents as plain dicts that match the shape the
conversation runner expects (``id``, ``card``, ``dialect``,
``bias_targets``, ``betting_voice``, plus a new ``skill`` field).

The conversation runner uses ``card`` + ``dialect`` + ``bias_targets``
+ ``betting_voice``. ``skill`` is currently surface-only (read by the
eval harness; available to future panel orchestration that wants to
match agents to topics).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_AGENTS_DIR = Path(__file__).parent
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Folders under ``app/agents/`` that aren't agent folders (system code).
_NON_AGENT_DIRS = {"_shared", "__pycache__"}


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    data = yaml.safe_load(m.group(1)) or {}
    if not isinstance(data, dict):
        return {}, text
    return data, text[m.end():]


@dataclass(frozen=True)
class AgentSpec:
    id: str
    name: str
    role: str
    card: str
    dialect: str
    bias_targets: list[str]
    responds_well_to: list[str]
    betting_voice: str
    skill: str
    raw_frontmatter: dict[str, Any]

    def as_persona_dict(self) -> dict[str, Any]:
        """Shape compatible with ``run_conversation``'s ``panel_personas``."""
        return {
            "id": self.id,
            "card": self.card,
            "dialect": self.dialect,
            "bias_targets": self.bias_targets,
            "responds_well_to": self.responds_well_to,
            "betting_voice": self.betting_voice,
            "skill": self.skill,
        }


def _read_optional(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _load_one(folder: Path) -> AgentSpec | None:
    persona_md = folder / "persona.md"
    if not persona_md.exists():
        return None
    raw = persona_md.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(raw)
    aid = str(fm.get("id") or folder.name).strip()
    if not aid:
        return None
    skill_body = _read_optional(folder / "skill.md").strip()
    bias_targets = fm.get("bias_targets") or []
    if not isinstance(bias_targets, list):
        bias_targets = []
    responds = fm.get("responds_well_to") or []
    if not isinstance(responds, list):
        responds = []
    return AgentSpec(
        id=aid,
        name=str(fm.get("name") or aid.title()),
        role=str(fm.get("role") or "panelist"),
        card=body.strip(),
        dialect=str(fm.get("dialect") or ""),
        bias_targets=[str(x) for x in bias_targets],
        responds_well_to=[str(x) for x in responds],
        betting_voice=str(fm.get("betting_voice") or ""),
        skill=skill_body,
        raw_frontmatter=fm,
    )


@lru_cache(maxsize=1)
def load_agents() -> list[AgentSpec]:
    """Return every agent under ``app/agents/`` sorted by id."""
    out: list[AgentSpec] = []
    if not _AGENTS_DIR.exists():
        return out
    for folder in sorted(_AGENTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in _NON_AGENT_DIRS or folder.name.startswith("_"):
            continue
        spec = _load_one(folder)
        if spec is not None:
            out.append(spec)
    return out


def load_panelists() -> list[AgentSpec]:
    """All agents with role 'panelist' (i.e. excludes JJJ + future editors)."""
    return [a for a in load_agents() if a.role == "panelist"]


def get_agent(agent_id: str) -> AgentSpec | None:
    """Look up one agent by id, or ``None``."""
    for a in load_agents():
        if a.id == agent_id:
            return a
    return None


__all__ = [
    "AgentSpec",
    "load_agents",
    "load_panelists",
    "get_agent",
]
