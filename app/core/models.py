from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ArticleIn(BaseModel):
    outlet_id: str
    title: str
    link: str
    summary: str | None = None
    published: datetime | None = None


class Article(BaseModel):
    """Stored article row (RSS ingest)."""

    article_id: str
    url: str
    url_hash: str
    outlet_id: str
    outlet_lean: str = "center"
    outlet_trust: float = 0.8
    title: str
    summary: str
    published_at: datetime | None = None
    fetched_at: datetime | None = None


class Corroboration(BaseModel):
    outlets: list[str]
    outlet_count: int
    spans_political_spectrum: bool
    has_wire: bool
    first_seen_at: datetime | None = None
    window_minutes: int = 30


class PersonaReaction(BaseModel):
    persona_id: str
    would_stop_scroll: bool = False
    stop_probability: float = 0.0
    would_share: bool = False
    would_comment: bool = False
    would_save: bool = False
    hook_suggestion: str = ""
    angle_of_interest: str = ""
    emotional_response: str = ""
    cliches_to_avoid: list[str] = Field(default_factory=list)


class PanelVerdict(BaseModel):
    reactions: list[PersonaReaction] = Field(default_factory=list)
    overall_stop_score: float = 0.0
    breadth: float = 0.0
    cross_demographic_hooks: list[str] = Field(default_factory=list)
    avoid_list: list[str] = Field(default_factory=list)


class ResearchOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    confirmed_facts: list[str] = Field(default_factory=list)
    conflicting_claims: list[str] = Field(default_factory=list)
    most_likely_narrative: str = ""
    entities: list[str] = Field(default_factory=list)
    suggested_tone: str = ""
    visual_opportunities: list[str] = Field(default_factory=list)
    cultural_tensions: list[str] = Field(default_factory=list)
    target_demographics: list[str] = Field(default_factory=list)
    headline_ledger: list[str] = Field(
        default_factory=list,
        description="Verbatim ledger lines echoed from clustered headlines",
    )
    news_hook_line: str = ""
    whats_at_stake_viewer: str = ""
    thumbnail_concept_one_sentence: str = ""
    do_not_claim: list[str] = Field(default_factory=list)


class StoryboardScene(BaseModel):
    order: int
    duration_s: float
    type: str
    image_prompt: str
    narration: str | None = None
    on_screen_text: str | None = None


class StoryboardOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    aspect_ratio: str = "9:16"
    total_duration_s: float = 14.0
    engagement_driver: str = ""
    hook_pattern: str = ""
    format_convention: str = ""
    hero_scene_index: int = 0
    scenes: list[StoryboardScene] = Field(default_factory=list)


class CriticFeedback(BaseModel):
    model_config = ConfigDict(extra="ignore")

    weaknesses: list[str] = Field(default_factory=list)
    missed_opportunities: list[str] = Field(default_factory=list)
    factual_concerns: list[str] = Field(default_factory=list)
    cliches: list[str] = Field(default_factory=list)
    suggested_rewrite_notes: str = ""

    @field_validator("suggested_rewrite_notes", mode="before")
    @classmethod
    def _coerce_notes(cls, v):
        if v is None:
            return ""
        if isinstance(v, list):
            return "\n".join(str(x) for x in v if x is not None)
        return str(v)
    overall_grade: Literal["weak", "ok", "good", "strong"] = "ok"
    viral_score: int = 5
    viral_drivers: list[str] = Field(default_factory=list)
    viral_misses: list[str] = Field(default_factory=list)
    format_convention_used: str | None = None
    trend_alignment: list[str] = Field(default_factory=list)


class SlopExample(BaseModel):
    pattern: str
    replacement: str
    category: str
    editor_comment: str


class EditorPass(BaseModel):
    final_prompt: str
    final_rationale: str
    slop_caught: list[SlopExample] = Field(default_factory=list)
    angry_quips: list[str] = Field(default_factory=list)
    edits_made: list[str] = Field(default_factory=list)
    grade_before: int = 0
    grade_after: int = 0


class DebateTurn(BaseModel):
    role: str
    model: str
    provider: Literal["anthropic", "openai"]
    input_summary: str
    output_summary: str
    spend_cents: int = 0


class EventRecord(BaseModel):
    event_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    article_ids: list[str] = Field(default_factory=list)
    importance_score: float = 0.0
    corroboration: Corroboration | None = None
    panel_initial: PanelVerdict | None = None
    research: ResearchOut | None = None
    storyboard: StoryboardOut | None = None
    draft_prompt: str | None = None
    panel_rerate: dict[str, Any] | None = None
    critic_feedback: CriticFeedback | None = None
    editor_pass: EditorPass | None = None
    debate_trace: list[DebateTurn] = Field(default_factory=list)
    state: str = "new"
    spend_cents: int = 0
