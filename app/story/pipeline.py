"""End-to-end pipeline.

Linear flow, no debate loops, no relevance gating:

1. Embed and cluster recent articles.
2. Pick the top cohesion-passing candidate (or one matching ``dedup_hash``).
3. Build the ``sources`` list (per-outlet article URLs).
4. One LLM call → cross-outlet brief + Higgsfield video prompt + hero image prompt.
5. Persona panel discusses the story.
6. Polymarket: live-match → propose-market if no live match.
7. Render the hero image.
8. Optional: Higgsfield image-to-video render (off by default, hero is the
   reference frame).
9. Write the 5-file output folder + rebuild ``output/README.md``.

All progress is printed to stdout via ``app.core.log`` — single-line timestamps,
no JSON until the final summary. Errors squash to ``WARNING: <step> failed: ...``
and the pipeline keeps going where it sanely can.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.agents.brief import write_brief
from app.agents.jjj import edit_brief
from app.cluster.candidates import (
    StoryCandidate,
    discover_story_candidates,
    enrich_lean_from_feeds,
    sort_articles_newest_first,
)
from app.cluster.embedding import embed_texts
from app.core.config import get_settings
from app.core.db import init_db, list_articles
from app.core.log import step, warn
from app.imagegen import render_hero
from app.ingest.rss import poll_all as poll_rss
from app.ingest.twitter_nitter import poll_all_twitter
from app.personas.conversation import run_conversation
from app.personas.panel import load_personas
from app.polymarket.matcher import match_to_live_market
from app.polymarket.proposer import propose_market
from app.render.higgsfield import render_video
from app.story.write import write_event, write_index


_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Two candidate clusters with > this Jaccard overlap on their article-index
# sets are treated as sibling clusters of the same story (e.g. the two SCOTUS
# Virginia redistricting clusters that share 4 of 5 articles).
SIBLING_JACCARD_THRESHOLD: float = 0.5


def _slugify(s: str, max_len: int = 80) -> str:
    out = _SLUG_RE.sub("-", (s or "").lower()).strip("-")
    return out[:max_len] or "untitled"


def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _short_hash(h: str) -> str:
    return (h or "")[:8] or "????????"


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
    except Exception as exc:
        warn("polymarket.match", exc)
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
    except Exception as exc:
        warn("polymarket.propose", exc)
        return None, ""
    market = {
        "proposed_market_slug": proposed.slug,
        "proposed_market": proposed.model_dump(),
    }
    ctx = f"Proposed Polymarket market: {proposed.title} (slug: {proposed.slug}, horizon: {proposed.horizon})"
    return market, ctx


def _print_key_status() -> None:
    s = get_settings()
    oa = "yes" if s.has_openai else "no"
    an = "yes" if s.has_anthropic else "no"
    if s.has_openai and s.has_anthropic:
        suffix = "using both"
    elif s.has_openai:
        suffix = "routing conversation+proposer through OpenAI"
    elif s.has_anthropic:
        suffix = "routing brief through Anthropic (no hero image, no higgsfield)"
    else:
        suffix = "no providers — pipeline will fail"
    step(f"keys: openai={oa}, anthropic={an} ({suffix})")


async def _run_one(
    chosen: StoryCandidate,
    articles: list[dict[str, Any]],
    *,
    log_tag: str = "",
) -> dict[str, Any]:
    """Run one chosen candidate end-to-end.

    Every progress line is prefixed with ``log_tag`` (e.g. ``"[2/5]"``) so
    parallel runs stay readable as their output interleaves on stdout.
    """
    s = get_settings()

    def _say(msg: str, *, indent: int = 0) -> None:
        step(f"{log_tag} {msg}".strip() if log_tag else msg, indent=indent)

    def _warn(where: str, exc: BaseException | str) -> None:
        warn(f"{log_tag} {where}".strip() if log_tag else where, exc)

    sub = [articles[i] for i in chosen.indices]
    headline = (sub[0].get("title") or "").strip().split("\n", 1)[0][:70]
    _say(
        f"cluster {_short_hash(chosen.dedup_hash)} "
        f"({len(sub)} sources, sim {chosen.mean_pairwise_sim:.2f}): {headline}"
    )

    sources = await _build_sources(sub)
    client = AsyncOpenAI(api_key=s.openai_api_key) if s.has_openai else None

    market, market_context = await _market_context_for(sub)
    if market and market.get("market_url"):
        price = market.get("price_cents")
        price_s = f" @ {price}c" if price is not None else ""
        _say(f"polymarket: live match -> {market.get('question', '')[:60]}{price_s}", indent=1)
    elif market and market.get("proposed_market_slug"):
        _say(f"polymarket: proposed -> {market['proposed_market_slug']}", indent=1)
    else:
        _say("polymarket: no angle", indent=1)

    brief = await write_brief(client, sources, market_context=market_context)
    meta = brief.get("_meta") or {}
    _say(
        f"brief: {meta.get('model', 'openai')} "
        f"({meta.get('tokens_in', 0)} tokens in, {meta.get('tokens_out', 0)} out)",
        indent=1,
    )

    cluster_summary = "\n".join(
        f"[{src['outlet_id']}] {src['title']}" for src in sources[:10]
    )
    conversation_dict: dict[str, Any] | None
    try:
        personas = load_personas()
        panel = personas[:5]
        if panel:
            transcript, _ = await run_conversation(
                panel,
                summary=f"{brief.get('hook', '')}\n\n{cluster_summary}",
                market_context=market_context or "",
                event_id=chosen.dedup_hash,
            )
            consensus = transcript.consensus_probability_pct
            tag_da = " (incl. devil's advocate)" if transcript.devil_advocate_id else ""
            outcome = transcript.ended_reason
            if consensus is not None:
                outcome = f"{transcript.ended_reason} -> {consensus}c YES"
            _say(
                f"panel: {len(panel)} personas{tag_da}, {len(transcript.turns)} turns, {outcome}",
                indent=1,
            )
            conversation_dict = transcript.as_dict()
            conversation_dict["turns"] = [
                {
                    "order": t.order,
                    "persona_id": t.persona_id,
                    "text": t.text,
                    "probability_pct": t.probability_pct,
                    "is_devil_advocate": t.is_devil_advocate,
                }
                for t in transcript.turns
            ]
        else:
            conversation_dict = {"warning": "no personas loaded"}
            _warn("panel", "no personas loaded")
    except Exception as exc:
        _warn("panel", exc)
        conversation_dict = {"error": f"{type(exc).__name__}: {exc}"}

    # JJJ — editor pass between panel and hero. Polishes hook + higgsfield
    # prompt only; hero_image_prompt is intentionally left untouched.
    if conversation_dict and conversation_dict.get("turns"):
        try:
            await edit_brief(client, brief, conversation_dict)
            jjj = brief.get("_jjj") or {}
            if jjj.get("error"):
                _warn("jjj", jjj["error"])
            else:
                _say(
                    "jjj: edited hook + higgsfield prompt "
                    f"(tokens {jjj.get('tokens_in', 0)} in / "
                    f"{jjj.get('tokens_out', 0)} out)",
                    indent=1,
                )
        except Exception as exc:
            _warn("jjj", exc)

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hook = brief.get("hook") or sub[0].get("title", "event")
    event_id = f"{day}-{_slugify(hook)}"
    folder = s.output_dir / event_id
    folder.mkdir(parents=True, exist_ok=True)

    hero_path: Path | None = None
    hero_skip_reason: str | None = None
    if not s.has_openai:
        hero_skip_reason = "no OPENAI_API_KEY in .env"
        _say("hero: skipped (no OPENAI_API_KEY)", indent=1)
    else:
        candidate_hero = folder / "hero.png"
        try:
            result = await render_hero(brief.get("hero_image_prompt") or hook, candidate_hero)
            if result is not None:
                hero_path = result
                _say(f"hero: rendered {hero_path.name}", indent=1)
            else:
                hero_skip_reason = "hero render returned no image"
                _warn("hero", "renderer returned None")
        except Exception as exc:
            _warn("hero", exc)
            hero_skip_reason = f"image generation failed: {type(exc).__name__}"
            brief.setdefault("_warnings", []).append(f"hero render failed: {exc}")

    video_path: Path | None = None
    higgs = brief.get("higgsfield") or {}
    if not s.enable_higgsfield_render:
        _say("higgsfield: skipped (ENABLE_HIGGSFIELD_RENDER=false)", indent=1)
    elif not s.higgsfield_api_key.strip():
        _say("higgsfield: skipped (no HIGGSFIELD_API_KEY)", indent=1)
    elif hero_path is None:
        _say("higgsfield: skipped (no hero image)", indent=1)
    else:
        candidate_video = folder / "video.mp4"
        try:
            result = await render_video(
                prompt=str(higgs.get("prompt") or hook),
                hero_path=hero_path,
                out_path=candidate_video,
                camera_move=str(higgs.get("camera_move") or "static"),
                aspect_ratio=str(higgs.get("aspect_ratio") or "9:16"),
                duration_s=int(higgs.get("duration_s") or 8),
            )
            video_path = result
        except Exception as exc:
            _warn("higgsfield", exc)
            video_path = None

    # Drop transient meta before writing — it's not part of the deliverable.
    brief.pop("_meta", None)

    write_event(
        folder,
        event_id=event_id,
        brief=brief,
        sources=sources,
        conversation=conversation_dict,
        market=market,
        hero_path=hero_path,
        video_path=video_path,
        hero_skip_reason=hero_skip_reason,
    )
    _say(f"-> output/{folder.name}/", indent=1)

    return {
        "ok": True,
        "event_id": event_id,
        "folder": str(folder),
        "dedup_hash": chosen.dedup_hash,
        "sources": len(sources),
        "outlets": list(chosen.outlets),
        "mean_pairwise_sim": chosen.mean_pairwise_sim,
        "video_rendered": bool(video_path),
    }


async def _load_articles_and_candidates() -> tuple[list[dict[str, Any]], list[StoryCandidate]]:
    await init_db()
    raw = await list_articles(limit=600)
    if not raw:
        return [], []
    articles = sort_articles_newest_first(
        enrich_lean_from_feeds([a.model_dump() if hasattr(a, "model_dump") else dict(a) for a in raw])
    )
    embeddings = await embed_texts(
        [(a.get("title") or "") + " " + (a.get("summary") or "")[:200] for a in articles]
    )
    candidates = discover_story_candidates(articles, embeddings, poc_mode=True)
    return articles, candidates


async def run_story(dedup_hash: str | None = None) -> dict[str, Any]:
    """Pick a cluster (by hash or top-ranked), produce the brief, write the folder."""
    _print_key_status()
    articles, candidates = await _load_articles_and_candidates()
    if not articles:
        warn("cluster", "no articles in DB — run `ingest` first")
        return {"ok": False, "reason": "no articles in DB — run `ingest` first"}
    if not candidates:
        warn("cluster", "no cohesion-passing candidates")
        return {"ok": False, "reason": "no cohesion-passing candidates"}

    step(f"cluster: {len(articles)} articles -> {len(candidates)} candidates")

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

    result = await _run_one(chosen, articles, log_tag="")
    write_index(get_settings().output_dir)
    return result


async def run_multiple(n: int = 5) -> list[dict[str, Any]]:
    """Run up to ``n`` cohesion-passing candidates concurrently, skipping siblings.

    Two candidates whose article-index sets have Jaccard overlap above
    ``SIBLING_JACCARD_THRESHOLD`` are treated as the same story (one cluster
    drawn slightly differently) and we walk further down the ranked list to
    fill ``n`` truly distinct stories.

    Stories run inside an ``asyncio.gather`` bounded by an ``asyncio.Semaphore``
    sized to ``settings.max_parallel_stories`` (default 3) — so we get the
    speedup without hammering provider rate limits. Each story's ``step()``
    lines carry a ``[i/N]`` prefix so interleaved output is still followable,
    and the top-level ``output/README.md`` index is rebuilt exactly once after
    the batch completes.
    """
    s = get_settings()
    _print_key_status()
    articles, candidates = await _load_articles_and_candidates()
    if not articles:
        warn("cluster", "no articles in DB — run `ingest` first")
        return [{"ok": False, "reason": "no articles in DB — run `ingest` first"}]
    if not candidates:
        warn("cluster", "no cohesion-passing candidates")
        return [{"ok": False, "reason": "no cohesion-passing candidates"}]

    step(f"cluster: {len(articles)} articles -> {len(candidates)} candidates, picking top {n}")

    picked: list[StoryCandidate] = []
    skipped_siblings = 0
    for c in candidates:
        if len(picked) >= n:
            break
        if any(_jaccard(c.indices, p.indices) > SIBLING_JACCARD_THRESHOLD for p in picked):
            skipped_siblings += 1
            continue
        picked.append(c)
    if skipped_siblings:
        step(f"cluster: skipped {skipped_siblings} sibling cluster(s)")
    if not picked:
        return [{"ok": False, "reason": "no distinct candidates after sibling dedup"}]
    if len(picked) < n:
        step(f"cluster: only {len(picked)} distinct candidates available (wanted {n})")

    concurrency = max(1, int(s.max_parallel_stories))
    sem = asyncio.Semaphore(concurrency)
    step(f"parallel: running {len(picked)} stories, up to {concurrency} at once")

    async def _wrap(idx: int, candidate: StoryCandidate) -> dict[str, Any]:
        tag = f"[{idx}/{len(picked)}]"
        async with sem:
            try:
                return await _run_one(candidate, articles, log_tag=tag)
            except Exception as exc:
                warn(f"{tag} story", exc)
                return {
                    "ok": False,
                    "dedup_hash": candidate.dedup_hash,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    results: list[dict[str, Any]] = await asyncio.gather(
        *[_wrap(i, c) for i, c in enumerate(picked, 1)]
    )

    # Rebuild the top-level index exactly once after the batch — saves N writes
    # to the same file and avoids any chance of races between parallel workers.
    try:
        write_index(s.output_dir)
    except Exception as exc:
        warn("index", exc)

    return results


def _sum_new(results: dict[str, dict]) -> int:
    return sum(int(v.get("new_count", 0) or 0) for v in results.values())


async def run_ingest() -> dict[str, Any]:
    """Poll RSS + X/Nitter once. Updates the sqlite DB."""
    await init_db()
    try:
        rss_results = await poll_rss()
    except Exception as exc:
        warn("ingest.rss", exc)
        rss_results = {}
    try:
        x_results = await poll_all_twitter()
    except Exception as exc:
        warn("ingest.twitter", exc)
        x_results = {}
    rss_new = _sum_new(rss_results)
    x_new = _sum_new(x_results)
    step(f"ingest: rss {rss_new} new, x {x_new} new")
    return {
        "ok": True,
        "rss_new": rss_new,
        "x_new": x_new,
        "rss_outlets": len(rss_results),
        "x_outlets": len(x_results),
    }
