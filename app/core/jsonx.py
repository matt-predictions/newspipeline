from __future__ import annotations

import json
import re


def extract_json_object(text: str) -> dict:
    """Parse first JSON object from model output (handles markdown fences)."""
    t = text.strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in t:
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1]
    t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            raise
        return json.loads(m.group(0))
