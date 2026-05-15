"""RSS feed pollers with ETag/Last-Modified support."""
from __future__ import annotations
import asyncio
import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from pathlib import Path

import feedparser
import httpx
import yaml
from slugify import slugify

from app.core.config import get_settings
from app.core.db import get_article_by_url_hash, upsert_article, get_feed_meta, set_feed_meta
from app.core.models import Article


@dataclass
class FeedConfig:
    outlet_id: str
    name: str
    url: str
    lean: str
    trust: float
    is_wire: bool


_DROP_QUERY_KEYS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid", "mc_cid", "mc_eid"}


def canonicalize_url(url: str) -> str:
    """Normalize a URL for dedup: lowercase host, drop tracking params, sort, drop fragment."""
    p = urlparse(url.strip())
    scheme = (p.scheme or "https").lower()
    netloc = p.netloc.lower()
    path = p.path.rstrip("/") or "/"
    qs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if k.lower() not in _DROP_QUERY_KEYS]
    qs.sort()
    query = urlencode(qs)
    return urlunparse((scheme, netloc, path, "", query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(canonicalize_url(url).encode("utf-8")).hexdigest()[:32]


def load_feeds(path: str = "feeds.yml") -> list[FeedConfig]:
    p = get_settings().project_root / path
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text())
    out: list[FeedConfig] = []
    for f in data.get("feeds", []):
        oid = f["outlet_id"]
        out.append(
            FeedConfig(
                outlet_id=oid,
                name=f.get("name", oid),
                url=f["url"],
                lean=f.get("lean", "center"),
                trust=float(f.get("trust", 0.8)),
                is_wire=oid in ("reuters", "apnews", "ap"),
            )
        )
    return out


async def _fetch_feed(client: httpx.AsyncClient, fc: FeedConfig) -> tuple[int, bytes | None, str | None, str | None]:
    """Fetch with conditional GET. Returns (status, body, etag, last_modified)."""
    meta = await get_feed_meta(fc.outlet_id)
    headers = {"User-Agent": "newspipeline/0.1 (+https://example.com/bot)"}
    if meta:
        if meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"):
            headers["If-Modified-Since"] = meta["last_modified"]
    try:
        r = await client.get(fc.url, headers=headers, timeout=15.0, follow_redirects=True)
        if r.status_code == 304:
            return 304, None, meta.get("etag") if meta else None, meta.get("last_modified") if meta else None
        if r.status_code >= 400:
            return r.status_code, None, None, None
        return r.status_code, r.content, r.headers.get("etag"), r.headers.get("last-modified")
    except Exception as ex:
        return 599, str(ex).encode(), None, None


def _parse_to_articles(fc: FeedConfig, body: bytes) -> list[Article]:
    parsed = feedparser.parse(body)
    out: list[Article] = []
    now = datetime.now(timezone.utc)
    for entry in parsed.entries[:30]:
        link = getattr(entry, "link", None)
        title = getattr(entry, "title", "").strip()
        if not link or not title:
            continue
        canonical = canonicalize_url(link)
        h = url_hash(canonical)
        summary = (getattr(entry, "summary", "") or getattr(entry, "description", "") or "").strip()
        # strip basic html
        summary = summary.replace("<p>", " ").replace("</p>", " ").replace("<br>", " ").replace("<br/>", " ")
        published_at = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                pass
        article_id = f"{fc.outlet_id}:{h[:12]}:{slugify(title)[:60]}"
        out.append(
            Article(
                article_id=article_id,
                url=canonical,
                url_hash=h,
                outlet_id=fc.outlet_id,
                outlet_lean=fc.lean,
                outlet_trust=fc.trust,
                title=title,
                summary=summary[:1500],
                published_at=published_at,
                fetched_at=now,
            )
        )
    return out


async def poll_feed(fc: FeedConfig) -> tuple[int, list[Article]]:
    """Poll one feed. Returns (status, new_articles_added)."""
    async with httpx.AsyncClient() as client:
        status, body, etag, last_modified = await _fetch_feed(client, fc)
    await set_feed_meta(fc.outlet_id, etag, last_modified, status)
    if status == 304 or body is None:
        return status, []
    if status >= 400:
        return status, []
    articles = _parse_to_articles(fc, body)
    new_articles: list[Article] = []
    for a in articles:
        existing = await get_article_by_url_hash(a.url_hash)
        if existing:
            continue
        await upsert_article(a.model_dump(mode="json"), a.url_hash)
        new_articles.append(a)
    return status, new_articles


async def poll_all(feeds: list[FeedConfig] | None = None) -> dict[str, dict]:
    feeds = feeds or load_feeds()
    if not feeds:
        return {}
    results: dict[str, dict] = {}
    async def _one(fc: FeedConfig):
        # jittered concurrency to be polite
        await asyncio.sleep(random.uniform(0, 0.5))
        try:
            status, new = await poll_feed(fc)
            results[fc.outlet_id] = {"status": status, "new_count": len(new), "name": fc.name}
        except Exception as ex:
            results[fc.outlet_id] = {"status": -1, "error": str(ex), "name": fc.name}
    await asyncio.gather(*[_one(fc) for fc in feeds])
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    args = parser.parse_args()

    async def main():
        from app.core.db import init_db
        await init_db()
        feeds = load_feeds()
        if not feeds:
            print("No feeds.yml found.")
            return
        print(f"Polling {len(feeds)} feeds...")
        results = await poll_all(feeds)
        for outlet_id, r in sorted(results.items()):
            status = r.get("status", "?")
            new = r.get("new_count", 0)
            name = r.get("name", outlet_id)
            err = r.get("error", "")
            tag = "OK" if status in (200, 304) else "ERR"
            print(f"  [{tag}] {outlet_id:14s} status={status} new={new:3d}  {name}{(' err=' + err) if err else ''}")
    asyncio.run(main())
