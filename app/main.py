"""CLI: ``python -m app.main {ingest,run}``.

Two commands. The whole POC:

- ``ingest``: poll RSS + X feeds, persist to sqlite.
- ``run``:    pick a cluster, generate the brief, write the event folder.
              ``--dedup-hash`` targets a specific story; otherwise top candidate.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from app.core.config import get_settings
from app.story.pipeline import run_ingest, run_story


async def _main_async(args: argparse.Namespace) -> None:
    s = get_settings()
    if args.cmd == "ingest":
        print(json.dumps(await run_ingest()))
        return
    if args.cmd == "run":
        s.validate_keys()
        print(json.dumps(await run_story(args.dedup_hash), default=str, indent=2))
        return


def main() -> None:
    p = argparse.ArgumentParser(prog="newspipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Poll RSS + X/Nitter once")
    run = sub.add_parser("run", help="Pick a cluster and write the event folder")
    run.add_argument(
        "--dedup-hash",
        default=None,
        help="Target a specific cluster (default: top candidate)",
    )
    asyncio.run(_main_async(p.parse_args()))


if __name__ == "__main__":
    main()
