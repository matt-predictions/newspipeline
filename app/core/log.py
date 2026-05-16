"""Greybeard progress logger.

Single-line, timestamped, stdout-only. No JSON dumps, no stack traces.

    [HH:MM:SS] cluster: 884 articles -> 12 candidates, picking top 5
    [HH:MM:SS] WARNING: hero render failed: BadRequestError: ...
"""

from __future__ import annotations

from datetime import datetime


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def step(msg: str, *, indent: int = 0) -> None:
    pad = "  " * indent
    print(f"[{_ts()}] {pad}{msg}", flush=True)


def warn(where: str, exc: BaseException | str) -> None:
    msg = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    # Squash multi-line errors to one line.
    msg = " ".join(str(msg).splitlines())[:200]
    print(f"[{_ts()}] WARNING: {where} failed: {msg}", flush=True)
