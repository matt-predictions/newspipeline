"""Higgsfield image-to-video rendering — optional, off by default.

Implements the documented `platform.higgsfield.ai` queue API:

  POST /{model_id}
       body: { image_url, prompt, duration, aspect_ratio }
       → { status: "queued", request_id, status_url, ... }
  GET  /requests/{request_id}/status
       → { status, video: { url } } once "completed"

Auth is the Higgsfield `KEY_ID:KEY_SECRET` pair, passed verbatim in the
``Authorization: Key ...`` header. Users paste their whole key string into
``HIGGSFIELD_API_KEY``.

Behavior when ``HIGGSFIELD_API_KEY`` is empty (Max's current state): prints
``higgsfield: skipped (no HIGGSFIELD_API_KEY)`` and returns ``None``. The
brief still ships the hero PNG + prompt + camera move so you can drop it
into Higgsfield by hand until the key lands.

Docs: https://docs.higgsfield.ai/docs/guides/video.md
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.log import step, warn


_BASE_URL = "https://platform.higgsfield.ai"
_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_TERMINAL_OK = {"completed"}
_TERMINAL_FAIL = {"failed", "nsfw", "canceled", "cancelled"}


def _image_data_url(path: Path) -> str:
    """Encode a local image as a ``data:`` URL.

    If the deployed endpoint rejects ``data:`` URLs in ``image_url`` (some do),
    swap this for a presigned upload — but every API call we tested with image
    references accepts data URLs, so we start here for simplicity.
    """
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"
    blob = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{blob}"


def _compose_prompt(prompt: str, camera_move: str) -> str:
    """Bake the camera move into the natural-language motion prompt.

    Higgsfield doesn't expose ``camera_move`` as a structured field — their
    docs say to describe the camera action inline. We prepend a single
    "Camera: <move>" hint and let the model handle the rest.
    """
    cam = (camera_move or "").strip()
    if not cam:
        return prompt.strip()
    cam_human = cam.replace("_", " ")
    return f"Camera: {cam_human}. {prompt.strip()}"


async def render_video(
    prompt: str,
    hero_path: Path,
    out_path: Path,
    *,
    camera_move: str = "static",
    aspect_ratio: str = "16:9",
    duration_s: int = 8,
) -> Path | None:
    """Submit one Higgsfield job, poll to completion, write MP4. Returns ``None`` on any failure.

    Caller is responsible for the ``enable_higgsfield_render`` gate; this
    function additionally short-circuits when ``HIGGSFIELD_API_KEY`` is empty
    so it's safe to call regardless.
    """
    s = get_settings()
    api_key = s.higgsfield_api_key.strip()
    if not api_key:
        step("higgsfield: skipped (no HIGGSFIELD_API_KEY)", indent=1)
        return None
    if not hero_path.exists():
        warn("higgsfield", f"hero image missing at {hero_path}")
        return None

    image_url = _image_data_url(hero_path)
    body: dict[str, Any] = {
        "image_url": image_url,
        "prompt": _compose_prompt(prompt, camera_move)[:2000],
        "duration": int(duration_s),
        "aspect_ratio": aspect_ratio or "16:9",
    }
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    model_id = (s.higgsfield_model_id or "").strip("/")
    submit_url = f"{_BASE_URL}/{model_id}"

    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT, follow_redirects=True
        ) as client:
            r = await client.post(submit_url, json=body, headers=headers)
            r.raise_for_status()
            queued = r.json() if r.content else {}
    except Exception as exc:
        warn("higgsfield.submit", exc)
        return None

    request_id = queued.get("request_id") if isinstance(queued, dict) else None
    if not request_id:
        warn("higgsfield", f"submit returned no request_id: {str(queued)[:200]}")
        return None
    step(
        f"higgsfield: queued {request_id} ({s.higgsfield_model_id}, {duration_s}s)",
        indent=1,
    )

    status_url = (
        queued.get("status_url") or f"{_BASE_URL}/requests/{request_id}/status"
    )
    poll = max(2.0, float(s.higgsfield_poll_interval_s))
    iters = max(1, int(s.higgsfield_max_wait_s / poll))
    final_payload: dict[str, Any] = {}
    status = str(queued.get("status") or "queued").lower()
    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT, follow_redirects=True
        ) as client:
            for _ in range(iters):
                if status in _TERMINAL_OK:
                    break
                if status in _TERMINAL_FAIL:
                    warn("higgsfield", f"job {request_id} terminal status={status}")
                    return None
                await asyncio.sleep(poll)
                try:
                    pr = await client.get(status_url, headers=headers)
                    pr.raise_for_status()
                    final_payload = pr.json() if pr.content else {}
                except Exception as exc:
                    warn("higgsfield.poll", exc)
                    return None
                status = str(final_payload.get("status") or status).lower()
    except Exception as exc:
        warn("higgsfield.poll", exc)
        return None

    if status not in _TERMINAL_OK:
        warn("higgsfield", f"job {request_id} did not complete (status={status})")
        return None

    video_url = ""
    video_field = final_payload.get("video") if isinstance(final_payload, dict) else None
    if isinstance(video_field, dict):
        video_url = str(video_field.get("url") or "")
    elif isinstance(video_field, str):
        video_url = video_field
    if not video_url:
        warn("higgsfield", f"job {request_id} completed without video.url")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT, follow_redirects=True
        ) as client:
            async with client.stream("GET", video_url) as resp:
                resp.raise_for_status()
                with out_path.open("wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        fh.write(chunk)
    except Exception as exc:
        warn("higgsfield.download", exc)
        return None

    step(f"higgsfield: wrote {out_path.name}", indent=1)
    return out_path
