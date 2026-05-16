"""Sora 2 video rendering — optional, off by default.

Uses OpenAI's ``videos.create`` endpoint with an image reference. The hero PNG
is passed as a base64 data URL on ``input_reference``. Polls until ``completed``
(or ``failed``), then downloads the MP4 to ``out_path``.

This module is intentionally self-contained:
- Returns ``None`` on any failure (gated by ``ENABLE_SORA_RENDER`` upstream).
- Never raises out — the brief should ship even when Sora is down.
- Caller is responsible for deciding whether to run at all.

Reference: https://developers.openai.com/api/reference/python/resources/videos
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.log import step, warn


_ASPECT_TO_SIZE: dict[str, str] = {
    # Sora 2 supports 720p in either orientation; sora-2-pro adds 1080p.
    "9:16": "720x1280",
    "16:9": "1280x720",
    "1:1": "720x720",
}

_ALLOWED_SECONDS = {"4", "8", "12"}


def _image_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"
    blob = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{blob}"


def _snap_seconds(duration_s: Any) -> str:
    """Snap a free-form duration into one of Sora's allowed values."""
    try:
        v = int(duration_s)
    except (TypeError, ValueError):
        v = 8
    # Nearest allowed: 4 / 8 / 12.
    if v <= 5:
        return "4"
    if v <= 10:
        return "8"
    return "12"


def _size_for(aspect_ratio: str) -> str:
    return _ASPECT_TO_SIZE.get((aspect_ratio or "").strip(), "720x1280")


async def render_video(
    *,
    prompt: str,
    hero_path: Path,
    out_path: Path,
    aspect_ratio: str = "9:16",
    duration_s: int = 8,
) -> Path | None:
    """Submit one Sora job, poll to completion, write MP4. Return ``None`` on any failure.

    Caller MUST already have checked ``settings.enable_sora_render`` and that
    ``hero_path`` exists on disk.
    """
    s = get_settings()
    if not s.has_openai:
        step("sora: skipped (no OPENAI_API_KEY)", indent=1)
        return None
    if not hero_path.exists():
        warn("sora", f"hero image missing at {hero_path}")
        return None

    client = AsyncOpenAI(api_key=s.openai_api_key)
    size = _size_for(aspect_ratio)
    seconds = _snap_seconds(duration_s)
    if seconds not in _ALLOWED_SECONDS:
        seconds = "8"

    try:
        data_url = _image_data_url(hero_path)
    except Exception as exc:
        warn("sora", exc)
        return None

    try:
        job = await client.videos.create(
            prompt=prompt[:2000],
            model=s.sora_model_id,
            input_reference={"image_url": data_url},
            seconds=seconds,
            size=size,
        )
    except Exception as exc:
        warn("sora.create", exc)
        return None

    video_id = getattr(job, "id", None)
    if not video_id:
        warn("sora", "create returned no video id")
        return None

    step(f"sora: queued {video_id} ({size}, {seconds}s)", indent=1)

    deadline_iters = max(1, int(s.sora_max_wait_s / max(s.sora_poll_interval_s, 1.0)))
    status = getattr(job, "status", "queued")
    for _ in range(deadline_iters):
        if status == "completed":
            break
        if status == "failed":
            warn("sora", f"job {video_id} failed")
            return None
        await asyncio.sleep(s.sora_poll_interval_s)
        try:
            job = await client.videos.retrieve(video_id)
        except Exception as exc:
            warn("sora.retrieve", exc)
            return None
        status = getattr(job, "status", status)
    if status != "completed":
        warn("sora", f"job {video_id} did not complete (status={status})")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        content = await client.videos.download_content(video_id=video_id)
    except Exception as exc:
        warn("sora.download", exc)
        return None

    # The OpenAI SDK returns an ``HttpxBinaryResponseContent`` with a
    # ``write_to_file`` method for the streaming case, plus ``.content`` for
    # the in-memory bytes. Try the file helper first, then fall back.
    try:
        write_helper = getattr(content, "write_to_file", None)
        if callable(write_helper):
            await asyncio.to_thread(write_helper, str(out_path))
        else:
            blob = getattr(content, "content", None)
            if blob is None and hasattr(content, "read"):
                blob = await content.read()  # type: ignore[func-returns-value]
            if not isinstance(blob, (bytes, bytearray)):
                warn("sora.download", "no bytes returned from download_content")
                return None
            out_path.write_bytes(bytes(blob))
    except Exception as exc:
        warn("sora.write", exc)
        return None

    step(f"sora: wrote {out_path.name}", indent=1)
    return out_path
