"""Cohesion gate for story clusters.

We were producing 60-article "mash" clusters because the old threshold sweep
dropped to 0.24 with no quality floor. This module enforces:

- mean pairwise cosine similarity >= 0.58 across an accepted cluster
- cluster size capped at 8 (closest-to-centroid wins)
- threshold walk floor of 0.48 (no 0.30 / 0.24 dust-buckets)
- slug derived from research.news_hook_line, not the first article title
- a day-stable hash over the sorted entity set so we don't publish dupes
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from app.cluster.embedding import cluster_articles, cosine_sim


COHESION_MIN_MEAN_SIM: float = 0.58
COHESION_MIN_OUTLETS: int = 2
CLUSTER_SIZE_CAP: int = 8
THRESHOLD_WALK: tuple[float, ...] = (0.72, 0.62, 0.54, 0.48)
THRESHOLD_FLOOR: float = 0.48
PAIR_FALLBACK_MIN_SIM: float = 0.46


def mean_pairwise_cosine(
    embeddings: list[list[float]], indices: Iterable[int]
) -> float:
    """Mean of all unordered pairs (i, j) with i < j in `indices`."""
    idx = list(indices)
    if len(idx) < 2:
        return 0.0
    total = 0.0
    count = 0
    for a in range(len(idx)):
        ea = embeddings[idx[a]]
        for b in range(a + 1, len(idx)):
            total += cosine_sim(ea, embeddings[idx[b]])
            count += 1
    return total / max(count, 1) if count else 0.0


def _centroid(embeddings: list[list[float]], indices: list[int]) -> np.ndarray:
    mat = np.array([embeddings[i] for i in indices], dtype=np.float64)
    return mat.mean(axis=0)


def cap_cluster_to_centroid(
    embeddings: list[list[float]],
    indices: list[int],
    max_size: int = CLUSTER_SIZE_CAP,
) -> list[int]:
    """Trim a cluster to `max_size` by keeping members closest to the centroid."""
    if len(indices) <= max_size:
        return list(indices)
    cent = _centroid(embeddings, indices)
    norms = np.linalg.norm(cent) or 1.0

    def sim_to_centroid(i: int) -> float:
        v = np.array(embeddings[i], dtype=np.float64)
        denom = (np.linalg.norm(v) * norms) or 1.0
        return float(np.dot(v, cent) / denom)

    ranked = sorted(indices, key=sim_to_centroid, reverse=True)
    return ranked[:max_size]


@dataclass(frozen=True)
class CohesionVerdict:
    passes: bool
    mean_sim: float
    size: int
    outlet_count: int
    reason: str


def cluster_passes_cohesion(
    embeddings: list[list[float]],
    indices: list[int],
    outlets: list[str],
    *,
    min_mean_sim: float = COHESION_MIN_MEAN_SIM,
    min_outlets: int = COHESION_MIN_OUTLETS,
) -> CohesionVerdict:
    distinct = sorted({o for o in outlets if o})
    mean_sim = mean_pairwise_cosine(embeddings, indices)
    if len(indices) < 2:
        return CohesionVerdict(False, mean_sim, len(indices), len(distinct), "size_lt_2")
    if len(distinct) < min_outlets:
        return CohesionVerdict(
            False, mean_sim, len(indices), len(distinct), "outlet_lt_min"
        )
    if mean_sim < min_mean_sim:
        return CohesionVerdict(
            False, mean_sim, len(indices), len(distinct), "mean_sim_below_floor"
        )
    return CohesionVerdict(True, mean_sim, len(indices), len(distinct), "passes")


def walk_thresholds_for_clusters(
    embeddings: list[list[float]],
    *,
    n_items: int,
    articles: list[dict] | None = None,
    thresholds: Iterable[float] = THRESHOLD_WALK,
) -> list[tuple[float, list[list[int]]]]:
    """Re-cluster at each threshold from coarse to relaxed. Returns (th, clusters)."""
    if articles is None:
        articles = [{} for _ in range(n_items)]
    out: list[tuple[float, list[list[int]]]] = []
    for th in thresholds:
        out.append((th, cluster_articles(articles, embeddings, threshold=th)))
    return out


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug_from_hook(hook_line: str, *, day_prefix: bool = True, max_len: int = 80) -> str:
    """Day-prefixed slug from research.news_hook_line (or any short phrase).

    Falls back to a date-only slug if the hook is empty.
    """
    raw = (hook_line or "").strip().lower()
    cleaned = _SLUG_RE.sub("-", raw).strip("-")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not cleaned:
        return f"{today}-story" if day_prefix else "story"
    base = f"{today}-{cleaned}" if day_prefix else cleaned
    return base[:max_len].rstrip("-") or (today if day_prefix else "story")


def story_dedup_hash(entities: Iterable[str], *, day_stable: bool = True) -> str:
    """Day-stable hash over the sorted, lower-cased entity set.

    Two clusters about the same set of named entities on the same day produce
    the same hash, which we use to skip duplicates.
    """
    norm = sorted({(e or "").strip().lower() for e in entities if (e or "").strip()})
    seed = "|".join(norm)
    if day_stable:
        seed = datetime.now(timezone.utc).strftime("%Y-%m-%d") + "|" + seed
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
