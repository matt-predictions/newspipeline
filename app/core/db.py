from __future__ import annotations

import json
import aiosqlite
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spend (
  day TEXT PRIMARY KEY,
  cents INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS articles (
  url_hash TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feed_meta (
  outlet_id TEXT PRIMARY KEY,
  etag TEXT,
  last_modified TEXT,
  last_status INTEGER
);
"""


async def init_db(path: Path | None = None) -> None:
    s = get_settings()
    dbp = path or s.db_path
    dbp.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(dbp) as db:
        await db.executescript(SCHEMA)
        await db.commit()


async def upsert_event(event_id: str, payload: dict[str, Any]) -> None:
    s = get_settings()
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(s.db_path) as db:
        await db.execute(
            "INSERT INTO events(event_id, payload, updated_at) VALUES(?, ?, ?)"
            " ON CONFLICT(event_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
            (event_id, json.dumps(payload, default=str), now),
        )
        await db.commit()


async def get_event(event_id: str) -> dict[str, Any] | None:
    s = get_settings()
    async with aiosqlite.connect(s.db_path) as db:
        cur = await db.execute(
            "SELECT payload FROM events WHERE event_id = ?", (event_id,)
        )
        row = await cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])


async def list_events(limit: int = 50) -> list[dict[str, Any]]:
    s = get_settings()
    async with aiosqlite.connect(s.db_path) as db:
        cur = await db.execute(
            "SELECT event_id, payload, updated_at FROM events ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
    out = []
    for eid, payload, updated in rows:
        d = json.loads(payload)
        d["_event_id"] = eid
        d["_updated_at"] = updated
        out.append(d)
    return out


async def add_spend_cents(day: str, cents: int) -> None:
    s = get_settings()
    async with aiosqlite.connect(s.db_path) as db:
        await db.execute(
            "INSERT INTO spend(day, cents) VALUES(?, ?)"
            " ON CONFLICT(day) DO UPDATE SET cents = spend.cents + excluded.cents",
            (day, cents),
        )
        await db.commit()


async def get_spend_day(day: str) -> int:
    s = get_settings()
    async with aiosqlite.connect(s.db_path) as db:
        cur = await db.execute("SELECT cents FROM spend WHERE day = ?", (day,))
        row = await cur.fetchone()
        return int(row[0]) if row else 0


async def get_article_by_url_hash(url_hash: str) -> dict[str, Any] | None:
    s = get_settings()
    async with aiosqlite.connect(s.db_path) as db:
        cur = await db.execute(
            "SELECT payload FROM articles WHERE url_hash = ?", (url_hash,)
        )
        row = await cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])


async def upsert_article(payload: dict[str, Any], url_hash: str) -> None:
    s = get_settings()
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(s.db_path) as db:
        await db.execute(
            "INSERT INTO articles(url_hash, payload, updated_at) VALUES(?, ?, ?)"
            " ON CONFLICT(url_hash) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
            (url_hash, json.dumps(payload, default=str), now),
        )
        await db.commit()


async def get_feed_meta(outlet_id: str) -> dict[str, Any] | None:
    s = get_settings()
    async with aiosqlite.connect(s.db_path) as db:
        cur = await db.execute(
            "SELECT etag, last_modified, last_status FROM feed_meta WHERE outlet_id = ?",
            (outlet_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        etag, lm, st = row
        return {"etag": etag, "last_modified": lm, "last_status": st}


async def set_feed_meta(
    outlet_id: str, etag: str | None, last_modified: str | None, status: int
) -> None:
    s = get_settings()
    async with aiosqlite.connect(s.db_path) as db:
        await db.execute(
            "INSERT INTO feed_meta(outlet_id, etag, last_modified, last_status) VALUES(?,?,?,?)"
            " ON CONFLICT(outlet_id) DO UPDATE SET etag=excluded.etag, last_modified=excluded.last_modified, last_status=excluded.last_status",
            (outlet_id, etag, last_modified, status),
        )
        await db.commit()


async def list_articles(limit: int = 200) -> list[dict[str, Any]]:
    s = get_settings()
    async with aiosqlite.connect(s.db_path) as db:
        cur = await db.execute(
            "SELECT payload FROM articles ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
    return [json.loads(r[0]) for r in rows]
