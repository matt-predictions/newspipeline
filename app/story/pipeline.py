"""End-to-end pipeline.

Linear flow, no debate loops, no relevance gating, no continuity index:

1. Embed and cluster recent articles.
2. Pick the top cohesion-passing candidate (or one matching a given dedup_hash).
3. Build the ``sources`` list (per-outlet article URLs).
4. One LLM call → cross-outlet brief + Higgsfield video prompt + hero image prompt.
5. Persona panel discusses the story.
6. Polymarket: live-match → propose-market if no live match.
7. Render the hero image.
8. Write the 4-file output folder + rebuild ``output/README.md``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.agents.brief import write_brief
from app.cluster.candidates import (
    StoryCandidate,
    discover_story_candidates,
    enrich_lean_from_feeds,
    sort_articles_newest_first,
)
from app.cluster.embedding import embed_texts
from app.core.config import get_settings
from app.core.db import init_db, list_articles
from app.imagegen import render_hero
from app.ingest.rss import poll_all as poll_rss
from app.ingest.twitter_nitter import poll_all_twitter
from app.personas.conversation import run_conversation
from app.personas.panel import load_personas
from app.polymarket.matcher import match_to_live_market
from app.polymarket.proposer import propose_market
from app.story.write import write_event, write_index


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(s: str, max_len: int = 80) -> str:
    out = _SLUG_RE.sub("-", (s or "").lower()).strip("-")
    return out[:max_len] or "untitled"


async def _build_sources(sub: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for a in sub:
        url = (a.get("url") or "").strip()
        if not url:
            continue
        sources.append({
            "outlet_id": a.get("outlet_id") or "",
            "outlet_lean": a.get("outlet_lean") or "",
            "outlet_trust": a.get("outlet_trust"),
            "title": (a.get("title") or "").strip(),
            "summary": ((a.get("summary") or "")[:280]).strip(),
            "url": url,
            "published_at": str(a.get("published_at") or ""),
        })
    return sources


async def _market_context_for(sub: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Try live Polymarket match → propose-market fallback. Returns (block, text-context)."""
    try:
        live = await match_to_live_market(sub)
    except Exception:
        live = None
    if live is not None:
        market = {
            "market_url": live.market_url,
            "market_id": live.market_id,
            "match_score": live.score,
            "price_cents": getattr(live, "price_cents", None),
            "question": getattr(live, "question", ""),
        }
        return market, f"Live Polymarket market: {live.question} ({live.market_url})"
    try:
        from app.core.models import ResearchOut

        proposed, _ = await propose_market(
            ResearchOut(
                news_hook_line=sub[0].get("title", "")[:160],
                headline_ledger=[a.get("title", "") for a in sub[:8]],
                entities=[],
            )
        )
    except Exception:
        return None, ""
    market = {
        "proposed_market_slug": proposed.slug,
        "proposed_market": proposed.model_dump(),
    }
    ctx = f"Proposed Polymarket market: {proposed.title} (slug: {proposed.slug}, horizon: {proposed.horizon})"
    return market, ctx


async def run_story(dedup_hash: str | None = None) -> dict[str, Any]:
    """Pick a cluster (by hash or top-ranked), produce the brief, write the folder."""
    s = get_settings()
    await init_db()
    raw = await list_articles(limit=600)
    if not raw:
        return {"ok": False, "reason": "no articles in DB — run `ingest` first"}
    articles = sort_articles_newest_first(
        enrich_lean_from_feeds([a.model_dump() if hasattr(a, "model_dump") else dict(a) for a in raw])
    )
    embeddings = await embed_texts(
        [(a.get("title") or "") + " " + (a.get("summary") or "")[:200] for a in articles]
    )
    candidates = discover_story_candidates(articles, embeddings, poc_mode=True)
    if not candidates:
        return {"ok": False, "reason": "no cohesion-passing candidates"}

    chosen: StoryCandidate | None = None
    if dedup_hash:
        for c in candidates:
            if c.dedup_hash == dedup_hash:
                chosen = c
                break
        if chosen is None:
            return {"ok": False, "reason": f"no candidate matches dedup_hash={dedup_hash}"}
    else:
        chosen = candidates[0]

    sub = [articles[i] for i in chosen.indices]
    sources = await _build_sources(sub)

    client = AsyncOpenAI(api_key=s.openai_api_key) if not s.dry_run else None
    market, market_context = await _market_context_for(sub)
    brief = await write_brief(
        client or AsyncOpenAI(api_key="dry-run"),
        sources,
        market_context=market_context,
    )

    cluster_summary = "\n".join(
        f"[{src['outlet_id']}] {src['title']}" for src in sources[:10]
    )
    try:
        personas = load_personas()
        panel = personas[:4] if personas else []
        if panel:
            transcript, _ = await run_conversation(
                panel,
                summary=f"{brief.get('hook', '')}\n\n{cluster_summary}",
                market_context=market_context or "",
            )
            conversation_dict = transcript.as_dict()
            conversation_dict["turns"] = [
                {
                    "order": t.order,
                    "persona_id": t.persona_id,
                    "text": t.text,
                    "probability_pct": t.probability_pct,
                }
                for t in transcript.turns
            ]
        else:
            conversation_dict = {"warning": "no personas loaded"}
    except Exception as exc:
        conversation_dict = {"error": f"{type(exc).__name__}: {exc}"}

    # Render hero image
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hook = brief.get("hook") or sub[0].get("title", "event")
    event_id = f"{day}-{_slugify(hook)}"
    folder = s.output_dir / event_id
    folder.mkdir(parents=True, exist_ok=True)
    hero_path = folder / "hero.png"
    try:
        await render_hero(brief.get("hero_image_prompt") or hook, hero_path)
    except Exception as exc:
        hero_path = None
        brief.setdefault("_warnings", []).append(f"hero render failed: {exc}")

    write_event(
        folder,
        event_id=event_id,
        brief=brief,
        sources=sources,
        conversation=conversation_dict,
        market=market,
        hero_path=hero_path,
    )
    write_index(s.output_dir)

    return {
        "ok": True,
        "event_id": event_id,
        "folder": str(folder),
        "dedup_hash": chosen.dedup_hash,
        "sources": len(sources),
        "outlets": list(chosen.outlets),
        "mean_pairwise_sim": chosen.mean_pairwise_sim,
    }


async def run_ingest() -> dict[str, Any]:
    """Poll RSS + X/Nitter once. Updates the sqlite DB."""
    await init_db()
    rss_count = await poll_rss()
    try:
        x_count = await poll_all_twitter()
    except Exception:
        x_count = 0
    return {"ok": True, "rss_new": rss_count, "x_new": x_count}
