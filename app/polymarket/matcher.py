"""Entity-set → live Polymarket market matcher.

The plan: ``research.entities`` + ``research.headline_ledger`` come in; we
score each candidate market by token-overlap on titles + tags, with a small
boost for shared named-entity tokens. If the best match clears
``MATCH_MIN_SCORE`` we return it; otherwise we return ``None`` and let the
proposer fill in.

This is deliberately a cheap deterministic matcher — no embedding call. We
already paid for the OpenAI embedding when clustering articles; doing it
again per market on every event would dominate the budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.polymarket.client import PolyMarket, list_active, search_markets


MATCH_MIN_SCORE: float = 0.32
MATCH_TOPK: int = 5
TOKEN_MIN_LEN: int = 3

_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "what", "when",
    "where", "after", "before", "into", "over", "amid", "says", "said",
    "will", "would", "could", "should", "yes", "no",
}

_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")


@dataclass
class MarketMatch:
    market: PolyMarket
    score: float
    matched_tokens: list[str]

    def as_dict(self) -> dict:
        return {
            "market": self.market.as_dict(),
            "score": self.score,
            "matched_tokens": self.matched_tokens,
        }


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t) >= TOKEN_MIN_LEN and t.lower() not in _STOP
    }


def _score(query_tokens: set[str], market: PolyMarket) -> tuple[float, list[str]]:
    if not query_tokens:
        return 0.0, []
    title_tokens = _tokens(market.title)
    tag_tokens: set[str] = set()
    for tag in market.tags:
        tag_tokens.update(_tokens(tag))
    matched = sorted(query_tokens & (title_tokens | tag_tokens))
    if not matched:
        return 0.0, []
    title_hits = len(query_tokens & title_tokens)
    tag_hits = len(query_tokens & tag_tokens) - title_hits
    base = title_hits + 0.5 * tag_hits
    score = base / max(len(query_tokens), 1)
    if any(t[0].isupper() for t in matched):
        score *= 1.10
    return min(score, 1.0), matched


def _query_text(
    entities: Iterable[str], headline_ledger: Iterable[str], extra: Iterable[str] = ()
) -> str:
    parts = list(entities) + list(headline_ledger) + list(extra)
    return "\n".join(p for p in parts if isinstance(p, str))


async def match_to_live_market(
    *,
    entities: Iterable[str],
    headline_ledger: Iterable[str],
    extra: Iterable[str] = (),
    use_search: bool = True,
    use_active_index: bool = True,
) -> MarketMatch | None:
    """Return the best live-market match or None.

    Pulls from both the search endpoint (entity-driven keyword query) and the
    active-events index (so we catch markets whose titles don't contain the
    obvious keyword). Best wins.
    """
    query = _query_text(entities, headline_ledger, extra)
    query_tokens = _tokens(query)
    candidates: list[PolyMarket] = []
    if use_search:
        seed = next(iter(entities), None) or " ".join(list(query_tokens)[:3])
        if seed:
            candidates.extend(await search_markets(seed, limit=MATCH_TOPK))
    if use_active_index:
        candidates.extend(await list_active(limit=200))
    seen: set[str] = set()
    best: MarketMatch | None = None
    for m in candidates:
        if m.slug in seen:
            continue
        seen.add(m.slug)
        if not m.is_active:
            continue
        score, matched = _score(query_tokens, m)
        if score < MATCH_MIN_SCORE:
            continue
        if best is None or score > best.score:
            best = MarketMatch(market=m, score=score, matched_tokens=matched)
    return best
