"""Pydantic schemas for every LLM boundary in the pipeline.

Each prompt declares its ``output_schema`` in frontmatter as
``app.agents.schemas:<ClassName>``; agents validate the LLM's JSON
output against the matching schema before using it. This replaces the
regex / string-split parsing that used to live inline in every caller.

The schemas are intentionally permissive on input (``extra="ignore"``,
``mode="before"`` coercers where the LLM tends to drift) and strict on
the contract — required keys are required, types are typed.

Why Pydantic and not JSON Schema directly: OpenAI's structured-outputs
feature accepts JSON Schema for ``response_format``, but Anthropic
doesn't, so we keep one validator that works for both and let
``response_format=json_object`` on the OpenAI side handle the
"return JSON" instruction.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.jsonx import extract_json_object


T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when an LLM response can't be coerced into the schema."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PanelTurnOut(_Base):
    """One panelist's JSON output for a single conversation turn."""

    text: str = ""
    probability_pct: int | None = None
    asks_persona: str | None = None

    @field_validator("probability_pct", mode="before")
    @classmethod
    def _coerce_pct(cls, v: Any) -> Any:
        if v in (None, "", "null", "None"):
            return None
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return None
        return max(0, min(100, iv))

    @field_validator("asks_persona", mode="before")
    @classmethod
    def _coerce_asks(cls, v: Any) -> Any:
        if v in (None, "", "null", "None"):
            return None
        return str(v)


class HiggsfieldBrief(_Base):
    """Higgsfield body fragment inside the brief (pre-render)."""

    prompt: str = ""
    camera_move: str = "static"
    aspect_ratio: str = "16:9"
    duration_s: int = 8

    @field_validator("duration_s", mode="before")
    @classmethod
    def _coerce_duration(cls, v: Any) -> Any:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return 8
        return max(3, min(20, iv))


class PublicOpinionSway(_Base):
    shift_direction: str = "muted"
    magnitude_pp: int = 0
    duration_days: int = 0
    cohorts_moved_positive: list[str] = Field(default_factory=list)
    cohorts_moved_negative: list[str] = Field(default_factory=list)
    rationale: str = ""


class DivergentFraming(_Base):
    outlets: list[str] = Field(default_factory=list)
    frame: str = ""


class BriefOut(_Base):
    """Full brief output produced by the one-shot brief writer."""

    hook: str = ""
    per_outlet_angle: dict[str, str] = Field(default_factory=dict)
    convergent_facts: list[str] = Field(default_factory=list)
    divergent_framings: list[DivergentFraming] = Field(default_factory=list)
    public_opinion_sway: PublicOpinionSway = Field(default_factory=PublicOpinionSway)
    story_grade: int = 0
    what_to_watch_next: str = ""
    higgsfield: HiggsfieldBrief = Field(default_factory=HiggsfieldBrief)
    hero_image_prompt: str = ""

    @field_validator("per_outlet_angle", mode="before")
    @classmethod
    def _coerce_outlet_angle(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return {}
        return {str(k): str(val) for k, val in v.items()}


class JJJEdit(_Base):
    """Output of the JJJ editor pass."""

    hook: str = ""
    higgsfield_prompt: str = ""
    edit_notes: str = ""


def parse_structured(text: str, model: type[T]) -> T:
    """Pull JSON out of an LLM response and validate against ``model``.

    Raises ``StructuredOutputError`` with the original text snippet on
    parse or validation failure — callers can fall back to safe defaults
    without juggling two exception types. Generic in the schema type so
    static analyzers see the concrete return type.
    """
    try:
        data = extract_json_object(text) or {}
    except Exception as exc:  # extract_json_object raises json.JSONDecodeError
        raise StructuredOutputError(
            f"could not extract JSON from response: {exc}\n--- response ---\n{text[:400]}"
        ) from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(
            f"response did not validate against {model.__name__}: {exc}\n"
            f"--- response ---\n{text[:400]}"
        ) from exc


__all__ = [
    "BriefOut",
    "DivergentFraming",
    "HiggsfieldBrief",
    "JJJEdit",
    "PanelTurnOut",
    "PublicOpinionSway",
    "StructuredOutputError",
    "parse_structured",
]
