"""Twitter / X ingest via rotating Nitter instances + RSSBridge fallback.

This module gives us a feed for every handle in ``twitter_sources.yml``
without paying for the X API. Strategy:

1. For each handle, try Nitter instances from ``nitter_instances.yml`` in
   order, with a small per-instance timeout. First 200 wins.
2. On total failure, try the RSSBridge fallback list (Twitter Bridge → Atom).
3. Parsed entries are normalised into ``Article`` rows and pushed through the
   same SQLite upsert as the RSS poller. So Twitter posts hit the cluster
   pipeline as if they were just another outlet.
4. Status per handle (ok / nitter_failed / rssbridge_used / total_fail) is
   captured per fetch so the health-check workflow can decide whether to
   open an issue.

Nitter rot is famously bad — we expect 1-3 instances to be broken at any
moment. The healthcheck workflow (``p4-monitoring``) pings each instance
hourly and opens an issue if every instance returns non-200.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import feedparser
import httpx
import yaml
from slugify import slugify

from app.core.config import get_settings
from app.core.db import (
    get_article_by_url_hash,
    get_feed_meta,
    set_feed_meta,
    upsert_article,
)
from app.core.models import Article


NITTER_CONFIG = Path(__file__).resolve().parent / "nitter_instances.yml"
TWITTER_SOURCES_FILE_DEFAULT = "twitter_sources.yml"
USER_AGENT = "newspipeline/0.2 (+twitter-nitter-mirror)"


@dataclass
class HandleConfig:
    handle: str
    outlet_id: str
    name: str
    lean: str = "center"
    trust: float = 0.7
    is_wire: bool = False


@dataclass
class NitterRing:
    nitter: list[str]
    rssbridge: list[str]

    @classmethod
    def load(cls, path: Path | None = None) -> "NitterRing":
        p = path or NITTER_CONFIG
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(
            nitter=[u.rstrip("/") for u in data.get("instances", []) if u],
            rssbridge=[u.rstrip("/") for u in data.get("rssbridge_instances", []) if u],
        )

    def shuffled_nitter(self) -> list[str]:
        out = list(self.nitter)
        random.shuffle(out)
        return out


def load_handles(path: str = TWITTER_SOURCES_FILE_DEFAULT) -> list[HandleConfig]:
    p = get_settings().project_root / path
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: list[HandleConfig] = []
    for row in data.get("handles", []):
        handle = (row.get("handle") or "").lstrip("@").strip()
        if not handle:
            continue
        out.append(
            HandleConfig(
                handle=handle,
                outlet_id=row.get("outlet_id", f"x_{handle.lower()}"),
                name=row.get("name", f"{handle} (X)"),
                lean=row.get("lean", "center"),
                trust=float(row.get("trust", 0.7)),
                is_wire=bool(row.get("is_wire", False)),
            )
        )
    return out


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


async def _fetch_feed_url(client: httpx.AsyncClient, url: str) -> tuple[int, bytes | None]:
    try:
        r = await client.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/atom+xml,application/rss+xml,*/*;q=0.5"},
            timeout=10.0,
            follow_redirects=True,
        )
        if r.status_code >= 400:
            return r.status_code, None
        return r.status_code, r.content
    except Exception:
        return 599, None


async def _try_nitter(handle: HandleConfig, ring: NitterRing) -> tuple[str, bytes | None, str]:
    """Try every Nitter instance; first 200 wins. Returns (source_url, body, instance_used)."""
    async with httpx.AsyncClient() as client:
        for inst in ring.shuffled_nitter():
            url = f"{inst}/{handle.handle}/rss"
            status, body = await _fetch_feed_url(client, url)
            if status == 200 and body:
                return url, body, inst
    return "", None, ""


async def _try_rssbridge(handle: HandleConfig, ring: NitterRing) -> tuple[str, bytes | None, str]:
    async with httpx.AsyncClient() as client:
        for inst in ring.rssbridge:
            url = (
                f"{inst}/?action=display&bridge=TwitterBridge&context=By+username"
                f"&u={handle.handle}&format=Atom"
            )
            status, body = await _fetch_feed_url(client, url)
            if status == 200 and body:
                return url, body, inst
    return "", None, ""


_ERROR_STUB_MARKERS = (
    "bridge returned error",
    "rss-bridge",
    "instance is down",
    "nitter is currently",
    "this feed is currently unavailable",
    "could not fetch feed",
)


def _looks_like_error_stub(title: str, summary: str) -> bool:
    """Reject RSSBridge / Nitter placeholder rows that aren't real posts."""
    blob = f"{title}\n{summary}".lower()
    return any(m in blob for m in _ERROR_STUB_MARKERS)


def _parse_handle_to_articles(handle: HandleConfig, body: bytes, source_url: str) -> list[Article]:
    parsed = feedparser.parse(body)
    out: list[Article] = []
    now = datetime.now(timezone.utc)
    for entry in parsed.entries[:30]:
        link = getattr(entry, "link", None)
        title = (getattr(entry, "title", "") or "").strip()
        if not link or not title:
            continue
        summary_preview = (getattr(entry, "summary", "") or "").strip()
        if _looks_like_error_stub(title, summary_preview):
            continue
        # Canonicalize to twitter.com so we dedupe across nitter mirrors
        canonical = link
        for marker in ("nitter.", "rss-bridge", "rssbridge"):
            if marker in canonical:
                tail = canonical.split("/", 3)[-1] if "/" in canonical else ""
                if tail:
                    canonical = f"https://twitter.com/{tail}"
                break
        url_h = _url_hash(canonical)
        summary = (getattr(entry, "summary", "") or "").strip()
        for tag in ("<p>", "</p>", "<br>", "<br/>"):
            summary = summary.replace(tag, " ")
        published_at = None
        if getattr(entry, "published_parsed", None):
            try:
                published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)  # type: ignore[misc]
            except Exception:
                pass
        article_id = f"{handle.outlet_id}:{url_h[:12]}:{slugify(title)[:60]}"
        out.append(
            Article(
                article_id=article_id,
                url=canonical,
                url_hash=url_h,
                outlet_id=handle.outlet_id,
                outlet_lean=handle.lean,
                outlet_trust=handle.trust,
                title=title[:300],
                summary=summary[:1500],
                published_at=published_at,
                fetched_at=now,
            )
        )
    return out


@dataclass
class HandleResult:
    handle: str
    outlet_id: str
    status: str
    new_count: int
    source_url: str
    instance: str


async def fetch_handle(handle: HandleConfig, ring: NitterRing) -> HandleResult:
    source_url, body, instance = await _try_nitter(handle, ring)
    used = "nitter"
    if body is None:
        source_url, body, instance = await _try_rssbridge(handle, ring)
        used = "rssbridge"
    if body is None:
        return HandleResult(handle.handle, handle.outlet_id, "fail", 0, "", "")
    await set_feed_meta(handle.outlet_id, None, None, 200)
    articles = _parse_handle_to_articles(handle, body, source_url)
    new_count = 0
    for a in articles:
        if await get_article_by_url_hash(a.url_hash):
            continue
        await upsert_article(a.model_dump(mode="json"), a.url_hash)
        new_count += 1
    return HandleResult(handle.handle, handle.outlet_id, used, new_count, source_url, instance)


async def poll_all_twitter(
    handles: Iterable[HandleConfig] | None = None,
    *,
    concurrency: int = 4,
) -> dict[str, dict[str, Any]]:
    handles_list = list(handles or load_handles())
    if not handles_list:
        return {}
    ring = NitterRing.load()
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, dict[str, Any]] = {}

    async def _one(h: HandleConfig) -> None:
        async with sem:
            await asyncio.sleep(random.uniform(0, 0.3))
            try:
                res = await fetch_handle(h, ring)
                results[h.outlet_id] = {
                    "status": res.status,
                    "new_count": res.new_count,
                    "handle": h.handle,
                    "instance": res.instance,
                }
            except Exception as ex:
                results[h.outlet_id] = {
                    "status": "fail",
                    "new_count": 0,
                    "handle": h.handle,
                    "error": str(ex)[:200],
                }

    await asyncio.gather(*[_one(h) for h in handles_list])
    return results


async def healthcheck_nitter() -> dict[str, str]:
    """Hit every Nitter instance's homepage and report 'ok' or 'fail'.

    Used by the ``nitter_healthcheck.yml`` workflow to open an issue when
    every instance is down.
    """
    ring = NitterRing.load()
    out: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
        for inst in ring.nitter:
            try:
                r = await client.get(inst, headers={"User-Agent": USER_AGENT})
                out[inst] = "ok" if r.status_code == 200 else f"fail:{r.status_code}"
            except Exception as ex:
                out[inst] = f"fail:{type(ex).__name__}"
    return out
