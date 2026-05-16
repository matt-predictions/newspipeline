"""``./eval`` entrypoint: run both judges, print a compact table.

Usage::

    ./eval                 # runs against the latest output/<event>/ folder
    ./eval output/<event>  # explicit target

Prints a small markdown-style table with overall + subscores per judge,
and the one-paragraph rationales. Exits non-zero only on actual error
(judges returning low scores is informational, not fatal).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from evals import consensus_quality, style


def _latest_event_folder() -> Path | None:
    out = get_settings().output_dir
    if not out.exists():
        return None
    candidates = sorted(
        (
            p
            for p in out.iterdir()
            if p.is_dir() and (p / "README.md").exists()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


async def _amain(target: Path) -> int:
    if not target.exists():
        print(f"target folder not found: {target}", file=sys.stderr)
        return 2

    print(f"== evals: {target.name} ==\n")

    judges = await asyncio.gather(
        consensus_quality.score_event_folder(target),
        style.score_event_folder(target),
        return_exceptions=True,
    )

    rows: list[tuple[str, str, str]] = []
    for j in judges:
        if isinstance(j, BaseException):
            rows.append(("error", f"{type(j).__name__}: {j}", ""))
            continue
        sub = " · ".join(f"{k}={v:.1f}" for k, v in j.subscores.items())
        rows.append((j.name, f"{j.overall:.2f}", sub))

    print("| judge | overall | subscores |")
    print("|---|---:|---|")
    for name, overall, sub in rows:
        print(f"| {name} | {overall} | {sub} |")
    print()

    for j in judges:
        if isinstance(j, BaseException):
            continue
        print(f"### {j.name}")
        print(f"_overall {j.overall:.2f}_")
        if j.rationale:
            print(f"\n{j.rationale}\n")

    # Also dump the raw score JSON to a sidecar next to the event so
    # eval scores can be diff'd over time.
    summary = {
        j.name: j.as_dict()
        for j in judges
        if not isinstance(j, BaseException)
    }
    (target / "evals.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return 0


def main() -> None:
    target_arg = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
    target = Path(target_arg) if target_arg else _latest_event_folder()
    if target is None:
        print("no event folder under output/ — run ./run -n 1 first", file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(_amain(target)))


if __name__ == "__main__":
    main()
