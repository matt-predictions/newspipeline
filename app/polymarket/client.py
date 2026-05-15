"""Thin async client over Polymarket's public Gamma API.

We only need three reads:

- ``search_markets(query)`` — keyword/entity search across active events
- ``get_market(slug)`` — full event/market detail by slug
- ``list_active(limit)`` — recent active events for the offline entity index

The public Gamma host is unauthenticated and rate-limited at a friendly cap,
so we keep an httpx.AsyncClient with sane timeouts and cache responses on disk
for the day. All network failures degrade to empty results; the matcher /
proposer below handle the "no live market" branch cleanly.

Endpoint reference (subject to upstream change):

    https://gamma-api.polymarket.com/events?active=true&closed=false&limit=200
    https://gamma-api.polymarket.com/events?slug=<event-slug>
    https://gamma-api.polymarket.com/markets?slug=<market-slug>
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_TIMEOUT_S = 12.0
USER_AGENT = "newspipeline/0.2 (+polymarket-marketing-video)"


@dataclass
class PolyOutcome:
    name: str
    price: float | None  # 0..1; None when unavailable

    @classmethod
    def from_event_token(cls, token: dict[str, Any]) -> "PolyOutcome":
        price = None
        for k in ("price", "lastTradePrice", "outcomePrice"):
            v = token.get(k)
            if v is not None:
                try:
                    price = float(v)
                    break
                except (TypeError, ValueError):
                    continue
        return cls(name=str(token.get("outcome") or token.get("name") or "?"), price=price)


@dataclass
class PolyMarket:
    slug: str
    title: str
    url: str
    end_date_iso: str | None
    volume_usd: float | None
    outcomes: list[PolyOutcome]
    tags: list[str]
    is_active: bool
    raw: dict[str, Any]

    @classmethod
    def from_event_payload(cls, payload: dict[str, Any]) -> "PolyMarket":
        slug = str(payload.get("slug") or "")
        title = str(payload.get("title") or payload.get("question") or slug)
        url = (
            f"https://polymarket.com/event/{slug}" if slug else payload.get("url", "")
        )
        end_iso = payload.get("endDate") or payload.get("end_date") or payload.get("endTime")
        try:
            volume = float(payload.get("volume", 0) or 0)
        except (TypeError, ValueError):
            volume = None
        outcomes_raw = payload.get("outcomes") or payload.get("tokens") or []
        outcomes: list[PolyOutcome] = []
        if isinstance(outcomes_raw, list):
            for o in outcomes_raw:
                if isinstance(o, dict):
                    outcomes.append(PolyOutcome.from_event_token(o))
        tags_raw = payload.get("tags") or []
        tags = []
        if isinstance(tags_raw, list):
            for t in tags_raw:
                if isinstance(t, dict):
                    label = t.get("label") or t.get("slug")
                    if label:
                        tags.append(str(label))
                elif isinstance(t, str):
                    tags.append(t)
        is_active = not bool(payload.get("closed", False)) and bool(
            payload.get("active", True)
        )
        return cls(
            slug=slug,
            title=title,
            url=url,
            end_date_iso=str(end_iso) if end_iso else None,
            volume_usd=volume,
            outcomes=outcomes,
            tags=tags,
            is_active=is_active,
            raw=payload,
        )

    def yes_price_cents(self) -> int | None:
        for o in self.outcomes:
            if o.price is not None and o.name.strip().lower() in ("yes", "true"):
                return int(round(o.price * 100))
        if self.outcomes and self.outcomes[0].price is not None:
            return int(round(self.outcomes[0].price * 100))
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "url": self.url,
            "end_date_iso": self.end_date_iso,
            "volume_usd": self.volume_usd,
            "outcomes": [
                {"name": o.name, "price": o.price} for o in self.outcomes
            ],
            "tags": self.tags,
            "is_active": self.is_active,
            "yes_price_cents": self.yes_price_cents(),
        }


def _cache_dir() -> Path:
    p = get_settings().data_dir / "cache" / "polymarket"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(*parts: str) -> str:
    blob = "|".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _cache_get(key: str, *, ttl_seconds: int = 600) -> Any | None:
    p = _cache_dir() / f"{key}.json"
    if not p.exists():
        return None
    age = (
        datetime.now(timezone.utc)
        - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    ).total_seconds()
    if age > ttl_seconds:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _cache_put(key: str, value: Any) -> None:
    p = _cache_dir() / f"{key}.json"
    p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


async def _http_get_json(
    url: str, params: dict[str, Any] | None = None, *, timeout: float = DEFAULT_TIMEOUT_S
) -> Any | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            r = await client.get(url, params=params)
            if r.status_code >= 400:
                return None
            return r.json()
    except Exception:
        return None


def _dry_run_event(slug: str = "fed-rate-decision-q3-2026") -> dict[str, Any]:
    return {
        "slug": slug,
        "title": "Fed Funds Rate decision in Q3 2026",
        "active": True,
        "closed": False,
        "endDate": "2026-09-30T00:00:00Z",
        "volume": 482_315.0,
        "tags": [{"label": "Economy"}, {"label": "Federal Reserve"}],
        "outcomes": [
            {"outcome": "Yes", "price": 0.62},
            {"outcome": "No", "price": 0.38},
        ],
    }


async def search_markets(query: str, *, limit: int = 8) -> list[PolyMarket]:
    """Keyword search against active Polymarket events."""
    s = get_settings()
    q = (query or "").strip()
    if not q:
        return []
    key = _cache_key("search", q.lower(), str(limit))
    cached = _cache_get(key, ttl_seconds=600)
    if cached is not None:
        return [PolyMarket.from_event_payload(p) for p in cached]
    if s.dry_run:
        payload = [_dry_run_event()]
        _cache_put(key, payload)
        return [PolyMarket.from_event_payload(p) for p in payload]
    data = await _http_get_json(
        f"{GAMMA_BASE_URL}/events",
        params={
            "active": "true",
            "closed": "false",
            "search": q,
            "limit": str(limit),
        },
    )
    if not isinstance(data, list):
        return []
    _cache_put(key, data)
    return [PolyMarket.from_event_payload(p) for p in data]


async def get_market(slug: str) -> PolyMarket | None:
    s = get_settings()
    slug = (slug or "").strip()
    if not slug:
        return None
    key = _cache_key("event", slug)
    cached = _cache_get(key, ttl_seconds=300)
    if cached is not None:
        return PolyMarket.from_event_payload(cached)
    if s.dry_run:
        payload = _dry_run_event(slug=slug)
        _cache_put(key, payload)
        return PolyMarket.from_event_payload(payload)
    data = await _http_get_json(
        f"{GAMMA_BASE_URL}/events",
        params={"slug": slug},
    )
    if isinstance(data, list) and data:
        _cache_put(key, data[0])
        return PolyMarket.from_event_payload(data[0])
    if isinstance(data, dict):
        _cache_put(key, data)
        return PolyMarket.from_event_payload(data)
    return None


async def list_active(limit: int = 200) -> list[PolyMarket]:
    s = get_settings()
    key = _cache_key("active", str(limit))
    cached = _cache_get(key, ttl_seconds=900)
    if cached is not None:
        return [PolyMarket.from_event_payload(p) for p in cached]
    if s.dry_run:
        payload = [_dry_run_event()]
        _cache_put(key, payload)
        return [PolyMarket.from_event_payload(p) for p in payload]
    data = await _http_get_json(
        f"{GAMMA_BASE_URL}/events",
        params={"active": "true", "closed": "false", "limit": str(limit)},
    )
    if not isinstance(data, list):
        return []
    _cache_put(key, data)
    return [PolyMarket.from_event_payload(p) for p in data]


async def warm_active_index(limit: int = 200) -> int:
    """Helper for the matcher to keep a recent local index."""
    return len(await list_active(limit))
