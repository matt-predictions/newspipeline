"""Persona loader.

The old POC carried two extra panel stages (initial reactions + draft rerate)
but the simplified flow only uses the round-robin conversation in
``conversation.py`` and just needs to read persona cards off disk. Anything
beyond ``load_personas`` was dead code, so it's been removed.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.core.config import get_settings


def load_personas(path: Path | None = None) -> list[dict[str, str]]:
    """Prefer per-card files in ``app/personas/cards/*.yml``; fall back to legacy ``personas.yml``."""
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
