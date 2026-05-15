"""Hero image generation.

One function: ``render_hero(prompt, out_path)`` produces a single PNG with the
inline Polymarket-ish brand prefix appended. Used as the Higgsfield reference
image.

Caching is keyed on the rendered prompt hash so identical prompts are free.
On API failure we walk size/quality combos and then fall back across models
(``OPENAI_IMAGE_MODEL`` → ``gpt-image-1`` → ``dall-e-3``).
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, BadRequestError

from app.core.config import get_settings


_BRAND_PREFIX = (
    "EDITORIAL / NEWS ILLUSTRATION — vertical 9:16 reference still, cinematic "
    "lighting. Background MUST be off-black (#0F1115) edge-to-edge — never "
    "white, never light gray. Single cyan accent (#1AB4E0) on the focal "
    "element. Warm orange (#E07A2F) only on risk / alert beats. "
    "HARD RULE — NEVER depict any named living public figure in ANY style "
    "(no photoreal, no illustration, no cartoon, no caricature). Use SYMBOLIC "
    "STAND-INS ONLY: solid black silhouettes with a cyan rim-light, empty "
    "podiums with name plates, flags + text, or news-graphic abstractions. "
    "Keep on-frame text to ≤ 8 words. No neon-grid / hologram / floating-particle slop."
)


_IMAGE_TRY_GPT_IMAGE_1: list[dict[str, Any]] = [
    {"size": "1024x1536", "quality": "high"},
    {"size": "1024x1536", "quality": "medium"},
    {"size": "1024x1024", "quality": "high"},
    {"size": "auto", "quality": "auto"},
]

_IMAGE_TRY_DALLE3: list[dict[str, Any]] = [
    {"size": "1024x1792", "quality": "standard"},
    {"size": "1024x1024", "quality": "standard"},
]


_DRY_RUN_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:24]


def _try_order(model: str) -> list[dict[str, Any]]:
    return _IMAGE_TRY_DALLE3 if model.startswith("dall-e") else _IMAGE_TRY_GPT_IMAGE_1


async def _attempt(
    client: AsyncOpenAI, *, model: str, prompt: str, kw: dict[str, Any]
) -> bytes | None:
    try:
        r = await client.images.generate(model=model, prompt=prompt[:4000], n=1, **kw)
    except (BadRequestError, Exception):
        return None
    img = r.data[0]
    if hasattr(img, "b64_json") and img.b64_json:
        return base64.b64decode(img.b64_json)
    if hasattr(img, "url") and img.url:
        import httpx

        async with httpx.AsyncClient() as h:
            resp = await h.get(img.url)
            resp.raise_for_status()
            return resp.content
    return None


async def render_hero(prompt: str, out_path: Path) -> Path:
    """Render one hero PNG. ``prompt`` is wrapped in the brand prefix."""
    s = get_settings()
    full = f"{_BRAND_PREFIX}\n\nSCENE:\n{prompt.strip()}"
    cache_dir = s.data_dir / "cache" / "images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{_cache_key(full)}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        out_path.write_bytes(cached.read_bytes())
        return out_path
    if s.dry_run:
        out_path.write_bytes(base64.b64decode(_DRY_RUN_PNG_B64))
        return out_path
    client = AsyncOpenAI(api_key=s.openai_api_key)
    models_to_try = [s.openai_image_model]
    if s.openai_image_model != "gpt-image-1":
        models_to_try.append("gpt-image-1")
    if "dall-e-3" not in models_to_try:
        models_to_try.append("dall-e-3")
    raw: bytes | None = None
    for m in models_to_try:
        for kw in _try_order(m):
            raw = await _attempt(client, model=m, prompt=full, kw=kw)
            if raw:
                break
        if raw:
            break
    if not raw:
        raise RuntimeError("image generation failed across all models / sizes")
    cached.write_bytes(raw)
    out_path.write_bytes(raw)
    return out_path
