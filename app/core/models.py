"""Pydantic models used by the pipeline.

Trimmed to the two types that survived the simplified flow:

- ``Article``    — one ingested article row (stored in SQLite, fed into
                   clustering).
- ``ResearchOut`` — minimal envelope the proposer consumes when no live
                   Polymarket market matches a cluster.

Everything else (the old researcher / director / critic / editor / panel
stack, storyboard scenes, debate-trace types) was removed when the
one-shot brief replaced the multi-agent debate. See git history for the
deleted types if you need to resurrect them.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Article(BaseModel):
    """One ingested article row.

    Identity is keyed on ``url_hash`` (canonical-URL sha256 prefix). The
    pipeline never touches the original article body — only title and
    feed-supplied summary — so we don't store body text.
    """

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


class ResearchOut(BaseModel):
    """Lightweight envelope the proposer reads.

    The brief writer's JSON output is shaped differently; this is the
    plumbing carrier that maps cluster headlines + a hook line into the
    fields the market-proposer prompt needs.
    """

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
