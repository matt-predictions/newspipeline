"""Prompt loader — heredocs as ``.md`` files for non-Python editing.

Every LLM-facing prompt in the pipeline now lives in this folder as a flat
``.md`` file. Python callsites pull them through ``load_prompt(name, **subs)``
where ``name`` is the basename (without ``.md``) and ``subs`` are
``str.format_map``-style substitutions.

Why ``str.format`` and not Jinja: the prompts are read-only after install,
substitutions are pure string interpolation, and ``format_map`` already
gives us positional-by-name with clean error messages. Adding a templating
engine to swap one heredoc per file would be overkill.

Literal ``{`` or ``}`` in a prompt body MUST be escaped as ``{{`` / ``}}``
in the ``.md`` file — same rule as Python f-strings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent


class _SafeDict(dict):
    """Format-map dict that raises a clear error on missing keys.

    The default ``str.format_map`` raises a bare ``KeyError(key)`` which is
    very hard to trace back to "which prompt was missing which substitution".
    This dict carries the prompt name so the message reads like a real
    diagnostic.
    """

    def __init__(self, mapping: dict[str, Any], *, prompt_name: str) -> None:
        super().__init__(mapping)
        self._prompt_name = prompt_name

    def __missing__(self, key: str) -> str:  # noqa: D401
        raise KeyError(
            f"prompt {self._prompt_name!r} missing substitution {{{key}}}"
        )


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    path = _DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"prompt {name!r} not found at {path} — available: "
            + ", ".join(p.stem for p in sorted(_DIR.glob("*.md")))
        )
    return path.read_text(encoding="utf-8")


def load_prompt(name: str, **subs: Any) -> str:
    """Read ``app/prompts/{name}.md`` and apply ``str.format_map`` substitutions.

    Reads are cached on first access. Missing substitutions raise a
    ``KeyError`` with the prompt name + missing field, so callsites fail
    loudly during development instead of shipping a half-rendered prompt to
    an LLM.
    """
    text = _read(name)
    if not subs:
        return text
    return text.format_map(_SafeDict(subs, prompt_name=name))


__all__ = ["load_prompt"]
