"""Prompt loader with YAML frontmatter.

Every LLM-facing prompt in the pipeline lives in this folder (or under an
agent folder via ``app.agents._loader``) as a ``.md`` file with an
optional YAML frontmatter block at the top. Frontmatter carries the
operational contract for the prompt; the body is the natural-language
template fed to the model.

Example::

    ---
    model: openai:gpt-4o
    temperature: 0.4
    purpose: Write the cross-outlet brief from a clustered news story
    inputs:
      today_iso: ISO date the run is pinned to
      cluster: Per-outlet headline ledger
      market_block: Optional live/proposed market context
      moves: Pipe-separated list of allowed Higgsfield camera moves
    output_format: json_object
    output_schema: app.agents.schemas:BriefOut
    owner: newspipeline-core
    version: 1.1.0
    ---
    TODAY IS {today_iso}.
    ...

Python callsites pull prompts through ``load_prompt(name, **subs)`` which
returns the rendered body. Frontmatter is fetched separately via
``prompt_meta(name)`` (returns a ``PromptMeta``); the loader also
validates that every declared ``input`` placeholder appears in the body
and that no undeclared ``{placeholder}`` slips through.

Why ``str.format`` and not Jinja: the prompts are read-only after
install, substitutions are pure string interpolation, and ``format_map``
already gives us positional-by-name with clean error messages.

Literal ``{`` or ``}`` in a prompt body MUST be escaped as ``{{`` /
``}}`` in the ``.md`` file — same rule as Python f-strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DIR = Path(__file__).parent


@dataclass(frozen=True)
class PromptMeta:
    """Frontmatter parse result for one prompt file."""

    name: str
    path: Path
    model: str = ""                 # "openai:gpt-4o", "anthropic:sonnet", or "" (caller decides)
    temperature: float | None = None
    purpose: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    output_format: str = ""         # "json_object" | "json_schema" | "text"
    output_schema: str = ""         # "module.path:ClassName" (when output_format=json_schema)
    owner: str = ""
    version: str = "1.0.0"
    raw: dict[str, Any] = field(default_factory=dict)


class _SafeDict(dict):
    """Format-map dict that raises a clear error on missing keys."""

    def __init__(self, mapping: dict[str, Any], *, prompt_name: str) -> None:
        super().__init__(mapping)
        self._prompt_name = prompt_name

    def __missing__(self, key: str) -> str:
        raise KeyError(
            f"prompt {self._prompt_name!r} missing substitution {{{key}}}"
        )


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_text = m.group(1)
    body = text[m.end():]
    try:
        data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("prompt frontmatter must be a YAML mapping")
    return data, body


@lru_cache(maxsize=None)
def _resolve_path(name: str) -> Path:
    """Resolve a prompt name to its on-disk file.

    Supports both flat prompts under ``app/prompts/`` (the legacy layout)
    and namespaced prompts under ``app/agents/<id>/prompts/`` (the
    folder-per-agent layout). Callers pass either ``"brief.user"`` or
    ``"agents/jjj/user"`` style names.
    """
    if name.startswith("agents/"):
        rel = name.split("/", 1)[1]
        agent_part, _, prompt_part = rel.partition("/")
        candidate = _DIR.parent / "agents" / agent_part / "prompts" / f"{prompt_part}.md"
        if not candidate.exists():
            candidate = _DIR.parent / "agents" / agent_part / f"{prompt_part}.md"
        if not candidate.exists():
            raise FileNotFoundError(
                f"prompt {name!r} not found at {candidate}"
            )
        return candidate
    path = _DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"prompt {name!r} not found at {path} — available: "
            + ", ".join(p.stem for p in sorted(_DIR.glob("*.md")))
        )
    return path


@lru_cache(maxsize=None)
def _load(name: str) -> tuple[PromptMeta, str]:
    path = _resolve_path(name)
    raw = path.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(raw)
    inputs = fm.get("inputs") or {}
    if not isinstance(inputs, dict):
        raise ValueError(
            f"prompt {name!r}: 'inputs' frontmatter must be a mapping, got {type(inputs).__name__}"
        )
    inputs = {str(k): str(v) for k, v in inputs.items()}
    temperature_raw = fm.get("temperature")
    temperature: float | None
    if temperature_raw is None:
        temperature = None
    else:
        try:
            temperature = float(temperature_raw)
        except (TypeError, ValueError):
            temperature = None
    meta = PromptMeta(
        name=name,
        path=path,
        model=str(fm.get("model") or ""),
        temperature=temperature,
        purpose=str(fm.get("purpose") or ""),
        inputs=inputs,
        output_format=str(fm.get("output_format") or ""),
        output_schema=str(fm.get("output_schema") or ""),
        owner=str(fm.get("owner") or ""),
        version=str(fm.get("version") or "1.0.0"),
        raw=fm,
    )
    if fm and fm.get("strict_inputs"):
        # Optional opt-in strict check: every placeholder in the body
        # must appear in `inputs:`. Off by default since several
        # prompts inline placeholders ({prefix}) that callers always
        # provide via the substitution dict.
        declared = set(inputs.keys())
        found = set(_PLACEHOLDER_RE.findall(body))
        undeclared = found - declared
        if undeclared:
            raise ValueError(
                f"prompt {name!r} body references placeholders "
                f"{sorted(undeclared)} not declared in frontmatter "
                "`inputs:` (set strict_inputs: false to disable)."
            )
    return meta, body


def prompt_meta(name: str) -> PromptMeta:
    """Return the parsed frontmatter for a prompt (no substitutions)."""
    meta, _ = _load(name)
    return meta


def load_prompt(name: str, **subs: Any) -> str:
    """Read ``app/prompts/{name}.md`` and apply ``str.format_map`` substitutions.

    Reads are cached on first access. Missing substitutions raise a
    ``KeyError`` with the prompt name + missing field, so callsites fail
    loudly during development instead of shipping a half-rendered prompt
    to an LLM.

    The returned string is the prompt BODY only — frontmatter (when
    present) is stripped. Fetch it via ``prompt_meta(name)``.
    """
    _, body = _load(name)
    if not subs:
        return body
    return body.format_map(_SafeDict(subs, prompt_name=name))


def list_prompts() -> list[str]:
    """List flat prompt names (under ``app/prompts/``) for quick discovery."""
    return sorted(p.stem for p in _DIR.glob("*.md"))


def validate_all_prompts() -> dict[str, PromptMeta]:
    """Parse every prompt under ``app/prompts/`` and assert frontmatter is sane.

    Useful as a CI / unit-test sanity check. Raises on the first prompt
    that fails to parse. Returns ``{name: meta}`` on success.
    """
    out: dict[str, PromptMeta] = {}
    for p in sorted(_DIR.glob("*.md")):
        name = p.stem
        meta, _ = _load(name)
        out[name] = meta
    return out


__all__ = [
    "PromptMeta",
    "load_prompt",
    "prompt_meta",
    "list_prompts",
    "validate_all_prompts",
]
