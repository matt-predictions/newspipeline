"""Persona loader (compatibility shim).

The actual agent definitions live under ``app/agents/<id>/persona.md``
since the folder-per-agent migration. This module is the thin shim that
returns the panel personas as plain dicts shaped the way
``app.personas.conversation.run_conversation`` expects.

Keeping the function name + return shape unchanged means the rest of
the pipeline never had to learn about ``AgentSpec``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.agents._loader import load_panelists
from app.core.config import get_settings


def load_personas(path: Path | None = None) -> list[dict[str, Any]]:
    """Return panelist dicts compatible with ``run_conversation``.

    Order matches the new ``app/agents/`` sorted-folder order. When the
    caller passes an explicit YAML ``path`` (used by tests), we still
    parse that legacy file.
    """
    if path is not None:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return list(data.get("personas", []))
    out = [a.as_persona_dict() for a in load_panelists()]
    if out:
        return out
    # Legacy fallback: `personas.yml` at repo root, kept for one release.
    s = get_settings()
    legacy = s.project_root / "personas.yml"
    if legacy.exists():
        data = yaml.safe_load(legacy.read_text(encoding="utf-8")) or {}
        return list(data.get("personas", []))
    return []
