"""Story-candidate enumeration.

POC scoring: a candidate is a cluster of >= 2 articles from >= 2 distinct
outlets passing a real cohesion gate (mean pairwise cosine >= 0.58, cluster
size capped). We rank by ``mean_pairwise_sim * outlet_count`` — cross-outlet
cohesion is the signal that "this is a real wire story".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.cluster.cohesion import (
    CLUSTER_SIZE_CAP,
    PAIR_FALLBACK_MIN_SIM,
    THRESHOLD_WALK,
    cap_cluster_to_centroid,
    cluster_passes_cohesion,
    mean_pairwise_cosine,
    story_dedup_hash,
)
from app.cluster.embedding import cluster_articles, cosine_sim
from app.ingest.rss import load_feeds


def parse_published_iso(a: dict[str, Any]) -> datetime | None:
    v = a.get("published_at")
    if not v:
        return None
    try:
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return None


def sort_articles_newest_first(
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def key(a: dict[str, Any]) -> float:
        p = parse_published_iso(a)
        return p.timestamp() if p else 0.0

    return sorted(articles, key=key, reverse=True)


def enrich_lean_from_feeds(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feeds = {f.outlet_id: f for f in load_feeds()}
    for a in articles:
        oid = a.get("outlet_id", "")
        if oid in feeds:
            a["outlet_lean"] = feeds[oid].lean
            a.setdefault("outlet_trust", feeds[oid].trust)
    return articles


@dataclass(frozen=True)
class StoryCandidate:
    indices: tuple[int, ...]
    score: float
    outlet_count: int
    outlets: tuple[str, ...]
    label: str
    mean_pairwise_sim: float = 0.0
    dedup_hash: str = ""


def _entities_from_cluster(sub: list[dict[str, Any]]) -> list[str]:
    """Cheap entity proxy: capitalized title tokens, used for the dedup hash."""
    out: list[str] = []
    for a in sub:
        for tok in (a.get("title") or "").split():
            t = tok.strip(",.;:!?\"'()[]")
            if len(t) >= 3 and t[0].isupper():
                out.append(t.lower())
    return out


def _make_candidate(
    idxs: list[int],
    articles: list[dict[str, Any]],
    embeddings: list[list[float]],
    label: str,
) -> StoryCandidate | None:
    idxs = sorted(set(idxs))
    if len(idxs) < 2:
        return None
    sub = [articles[i] for i in idxs]
    outlets = sorted({a.get("outlet_id", "") for a in sub if a.get("outlet_id")})
    if len(outlets) < 2:
        return None
    mean_sim = mean_pairwise_cosine(embeddings, idxs)
    return StoryCandidate(
        indices=tuple(idxs),
        score=mean_sim * len(outlets),
        outlet_count=len(outlets),
        outlets=tuple(outlets),
        label=label,
        mean_pairwise_sim=mean_sim,
        dedup_hash=story_dedup_hash(_entities_from_cluster(sub)),
    )


def best_cross_outlet_pair(
    articles: list[dict[str, Any]], embeddings: list[list[float]]
) -> tuple[list[int], float]:
    bi: list[int] = []
    best = -1.0
    by_outlet: dict[str, list[int]] = {}
    for i, a in enumerate(articles):
        oid = a.get("outlet_id") or "unknown"
        by_outlet.setdefault(oid, []).append(i)
    oids = list(by_outlet.keys())
    for oi in range(len(oids)):
        for oj in range(oi + 1, len(oids)):
            for ii in by_outlet[oids[oi]]:
                for jj in by_outlet[oids[oj]]:
                    s = cosine_sim(embeddings[ii], embeddings[jj])
                    if s > best:
                        best = s
                        bi = sorted([ii, jj])
    return bi, best


def discover_story_candidates(
    articles: list[dict[str, Any]],
    embeddings: list[list[float]],
    *,
    poc_mode: bool = False,
) -> list[StoryCandidate]:
    """Walk strict → relaxed thresholds with a real cohesion gate.

    ``poc_mode`` accepts a high-similarity 2-article cross-outlet pair when the
    wider walk produces nothing — useful for tests / first-run smoke checks.
    """
    seen_indices: set[tuple[int, ...]] = set()
    seen_hashes: set[str] = set()
    ranked: list[StoryCandidate] = []

    def push(c: StoryCandidate | None) -> None:
        if c is None or c.indices in seen_indices:
            return
        if c.dedup_hash and c.dedup_hash in seen_hashes:
            return
        seen_indices.add(c.indices)
        if c.dedup_hash:
            seen_hashes.add(c.dedup_hash)
        ranked.append(c)

    for th in THRESHOLD_WALK:
        for cl in cluster_articles(articles, embeddings, threshold=th):
            if len(cl) < 2:
                continue
            capped = cap_cluster_to_centroid(
                embeddings, list(cl), max_size=CLUSTER_SIZE_CAP
            )
            outlets = [articles[i].get("outlet_id", "") for i in capped]
            verdict = cluster_passes_cohesion(embeddings, capped, outlets)
            if not verdict.passes:
                continue
            push(
                _make_candidate(
                    capped,
                    articles,
                    embeddings,
                    label=f"cohesion_th{th:.2f}_mean{verdict.mean_sim:.2f}",
                )
            )

    if not ranked or poc_mode:
        pair, sim = best_cross_outlet_pair(articles, embeddings)
        if pair and sim >= PAIR_FALLBACK_MIN_SIM:
            push(
                _make_candidate(
                    pair,
                    articles,
                    embeddings,
                    label=f"cross_pair_sim{sim:.3f}",
                )
            )

    return sorted(ranked, key=lambda c: c.score, reverse=True)
