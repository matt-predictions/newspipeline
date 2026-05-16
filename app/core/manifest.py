"""Append-only run manifest.

Every story-generation run gets one manifest at::

    runs/<YYYY-MM-DD>/<cluster_hash[:12]>/manifest.json

The manifest is the replayable audit trail: every step (ingest, brief,
panel, jjj, hero, higgsfield, write) writes one event with timestamp,
model, latency, token counts, dollar/cent cost, and status. The
top-level summary at the bottom gives total cost + wall time + per-step
roll-up so the pipeline can print "this run cost 7c in 41s" at the end.

The manifest is JSON (not JSONL) — at the end of the run we write the
final object once. During the run, the ``RunManifest`` object holds the
event list in memory; ``flush()`` writes the JSON atomically.

Schema (top level)::

    {
      "schema_version": "1.0",
      "run_id": "<YYYY-MM-DDTHH:MM:SSZ>--<cluster_hash[:12]>",
      "cluster_hash": "...",
      "event_id": "2026-05-16-supreme-court-...",
      "started_at_iso": "...",
      "ended_at_iso": "...",
      "wall_seconds": 41.7,
      "totals": {
        "spend_cents": 7,
        "tokens_in": 12453,
        "tokens_out": 1842,
        "events": 8
      },
      "events": [
        {
          "ts_iso": "...",
          "elapsed_s_from_start": 0.7,
          "step": "brief",
          "model": "gpt-4o",
          "provider": "openai",
          "latency_ms": 4_320,
          "tokens_in": 1842,
          "tokens_out": 612,
          "spend_cents": 4,
          "status": "ok",
          "extras": { ... }
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.core.config import get_settings


SCHEMA_VERSION = "1.0"


@dataclass
class ManifestEvent:
    ts_iso: str
    elapsed_s_from_start: float
    step: str
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    spend_cents: int = 0
    status: str = "ok"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    cluster_hash: str
    event_id: str
    run_id: str
    started_at_iso: str
    started_monotonic: float
    folder: Path
    events: list[ManifestEvent] = field(default_factory=list)

    @classmethod
    def begin(cls, *, cluster_hash: str, event_id: str = "") -> "RunManifest":
        s = get_settings()
        now = datetime.now(timezone.utc)
        short_hash = (cluster_hash or "????????")[:12]
        run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}--{short_hash}"
        day = now.strftime("%Y-%m-%d")
        folder = s.runs_dir / day / short_hash
        folder.mkdir(parents=True, exist_ok=True)
        return cls(
            cluster_hash=cluster_hash,
            event_id=event_id,
            run_id=run_id,
            started_at_iso=now.isoformat(),
            started_monotonic=time.monotonic(),
            folder=folder,
        )

    def set_event_id(self, event_id: str) -> None:
        self.event_id = event_id

    def record(
        self,
        step: str,
        *,
        model: str = "",
        provider: str = "",
        latency_ms: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        spend_cents: int = 0,
        status: str = "ok",
        extras: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        elapsed = round(time.monotonic() - self.started_monotonic, 3)
        self.events.append(
            ManifestEvent(
                ts_iso=now.isoformat(),
                elapsed_s_from_start=elapsed,
                step=step,
                model=model,
                provider=provider,
                latency_ms=int(latency_ms),
                tokens_in=int(tokens_in),
                tokens_out=int(tokens_out),
                spend_cents=int(spend_cents),
                status=status,
                extras=extras or {},
            )
        )

    @contextmanager
    def time_step(
        self, step: str, **kw: Any
    ) -> Iterator[dict[str, Any]]:
        """Context manager that records a step + auto-fills ``latency_ms``.

        Mutate the yielded dict to attach ``model``, ``provider``,
        ``tokens_in`` etc. captured during the call.
        """
        start = time.monotonic()
        payload: dict[str, Any] = {
            "model": "",
            "provider": "",
            "tokens_in": 0,
            "tokens_out": 0,
            "spend_cents": 0,
            "status": "ok",
            "extras": {},
        }
        payload.update(kw)
        try:
            yield payload
        except Exception as exc:
            payload["status"] = f"error: {type(exc).__name__}"
            payload["extras"] = {**(payload.get("extras") or {}), "exc": str(exc)[:240]}
            self.record(
                step,
                latency_ms=int((time.monotonic() - start) * 1000),
                **payload,
            )
            raise
        self.record(
            step,
            latency_ms=int((time.monotonic() - start) * 1000),
            **payload,
        )

    def totals(self) -> dict[str, int]:
        spend = sum(e.spend_cents for e in self.events)
        tin = sum(e.tokens_in for e in self.events)
        tout = sum(e.tokens_out for e in self.events)
        return {
            "spend_cents": int(spend),
            "tokens_in": int(tin),
            "tokens_out": int(tout),
            "events": len(self.events),
        }

    def wall_seconds(self) -> float:
        return round(time.monotonic() - self.started_monotonic, 3)

    def flush(self) -> Path:
        """Write the manifest to disk. Idempotent — safe to call multiple times."""
        now = datetime.now(timezone.utc)
        body = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "cluster_hash": self.cluster_hash,
            "event_id": self.event_id,
            "started_at_iso": self.started_at_iso,
            "ended_at_iso": now.isoformat(),
            "wall_seconds": self.wall_seconds(),
            "totals": self.totals(),
            "events": [asdict(e) for e in self.events],
        }
        path = self.folder / "manifest.json"
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def update_latest_symlink(self) -> None:
        """Maintain ``runs/latest`` → most recent run folder.

        Best-effort: skips silently on filesystems without symlink
        support. The symlink is convenience-only; the manifest under
        ``runs/<date>/<hash>/`` is the source of truth.
        """
        s = get_settings()
        latest = s.runs_dir / "latest"
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            # Use a relative target so the symlink survives folder moves.
            latest.symlink_to(self.folder.relative_to(s.runs_dir))
        except (OSError, NotImplementedError):
            pass


__all__ = ["ManifestEvent", "RunManifest", "SCHEMA_VERSION"]
