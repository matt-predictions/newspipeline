"""Embedding + simple event clustering."""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from app.core.config import get_settings


def cosine_sim(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
        return 0.0
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


async def embed_texts(texts: list[str]) -> list[list[float]]:
    s = get_settings()
    if s.dry_run:
        # Same vector => single cluster for fixture tests
        v = [0.001 * (i % 100) for i in range(1536)]
        return [list(v) for _ in texts]
    client = AsyncOpenAI(api_key=s.openai_api_key)
    out: list[list[float]] = []
    batch_size = int(getattr(s, "embedding_batch_size", 96))
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        if not chunk:
            continue
        r = await client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk,
        )
        # Responses align with chunk order when input is array
        for d in sorted(r.data, key=lambda x: x.index):
            out.append(d.embedding)
    return out


def cluster_articles(
    articles: list[dict[str, Any]],
    embeddings: list[list[float]],
    threshold: float = 0.72,
) -> list[list[int]]:
    """Greedy cluster by max similarity to any member (POC)."""
    n = len(articles)
    if n == 0:
        return []
    clusters: list[list[int]] = []
    for i in range(n):
        placed = False
        for cl in clusters:
            best = max(cosine_sim(embeddings[i], embeddings[k]) for k in cl)
            if best >= threshold:
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters
