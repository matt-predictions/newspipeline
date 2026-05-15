"""Apply API keys from a local text file into .env (never prints secrets)."""
from __future__ import annotations

import os
import re
from pathlib import Path


def parse_keys(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip().upper()] = v.strip().strip('"').strip("'")
            continue
        if line.startswith("sk-ant-"):
            out.setdefault("ANTHROPIC_API_KEY", line)
        elif line.startswith("sk-"):
            out.setdefault("OPENAI_API_KEY", line)
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    candidates = [
        Path(os.path.expanduser("~/Desktop/keys")),
        Path(os.path.expanduser("~/Desktop/keys.txt")),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if not src:
        print("No keys file found at ~/Desktop/keys or ~/Desktop/keys.txt")
        raise SystemExit(1)
    data = parse_keys(src.read_text(encoding="utf-8"))
    ex = root / ".env.example"
    template = ex.read_text(encoding="utf-8") if ex.exists() else ""
    lines_out: list[str] = []
    if template:
        for line in template.splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                lines_out.append(f"ANTHROPIC_API_KEY={data.get('ANTHROPIC_API_KEY', '')}")
            elif line.startswith("OPENAI_API_KEY="):
                lines_out.append(f"OPENAI_API_KEY={data.get('OPENAI_API_KEY', '')}")
            else:
                lines_out.append(line)
    else:
        lines_out = [
            f"ANTHROPIC_API_KEY={data.get('ANTHROPIC_API_KEY','')}",
            f"OPENAI_API_KEY={data.get('OPENAI_API_KEY','')}",
        ]
    (root / ".env").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"Wrote {root / '.env'} from {src.name} (keys not shown).")


if __name__ == "__main__":
    main()
