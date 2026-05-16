"""CLI: ``python -m app.main {ingest,run}``.

Two commands. The whole POC:

- ``ingest``: poll RSS + X feeds, persist to sqlite.
- ``run``:    by default generates 5 distinct stories (``-n`` to override).
              Pass ``--dedup-hash`` to target one specific cluster.

Progress prints to stdout while it runs; final JSON summary at the end.
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
        if args.n and args.n > 1 and not args.dedup_hash:
            from app.story.pipeline import run_multiple

            results = await run_multiple(n=args.n)
            print(json.dumps(results, default=str, indent=2))
            return
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
    run.add_argument(
        "-n",
        type=int,
        default=5,
        help="Number of distinct stories to generate (default: 5)",
    )
    asyncio.run(_main_async(p.parse_args()))


if __name__ == "__main__":
    main()
