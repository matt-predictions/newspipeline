"""Per-event output writer + top-level index.

Each event folder contains EXACTLY four files (plus the hero PNG):

- ``README.md``         — human-readable wire summary, cross-outlet read,
                          panel transcript, Polymarket angle, Higgsfield prompt.
- ``hero.png``          — reference still (passed to Higgsfield as ref image).
- ``higgsfield.json``   — {prompt, camera_move, aspect_ratio, duration_s,
                          ref_image_path}: drop straight into Higgsfield API.
- ``conversation.json`` — persona panel transcript (the AI debate).
- ``sources.json``      — raw cluster article URLs + outlet metadata.

Plus a top-level ``output/README.md`` index of every event.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _lean_chip(lean: str) -> str:
    l = (lean or "").lower()
    if l in {"left", "center-left"}:
        return "🟦"
    if l in {"right", "center-right"}:
        return "🟥"
    if l in {"center", "centrist"}:
        return "⬜"
    if l in {"wire", "newswire"}:
        return "📰"
    return "•"


def _format_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""
    by_outlet: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for s in sources:
        oid = s.get("outlet_id") or "?"
        if oid not in by_outlet:
            order.append(oid)
            by_outlet[oid] = []
        by_outlet[oid].append(s)
    out = ["## Sources on the wire", ""]
    for oid in order:
        rows = by_outlet[oid]
        lean = rows[0].get("outlet_lean") or "unknown"
        chip = _lean_chip(lean)
        out.append(f"### {chip} `{oid}` — {lean} ({len(rows)} post{'s' if len(rows) != 1 else ''})")
        for r in rows:
            title = (r.get("title") or "").strip().replace("\n", " ")[:180]
            url = (r.get("url") or "").strip()
            ts = (r.get("published_at") or "")[:16]
            if url:
                out.append(f"- [{title}]({url}) — _{ts}_")
            else:
                out.append(f"- {title} — _{ts}_")
        out.append("")
    return "\n".join(out)


def _format_cross_outlet(brief: dict[str, Any]) -> str:
    angles = brief.get("per_outlet_angle") or {}
    divergence = brief.get("divergent_framings") or []
    sway = brief.get("public_opinion_sway") or {}
    grade = brief.get("story_grade")
    watch = brief.get("what_to_watch_next") or ""
    lines: list[str] = []
    if grade:
        lines.append(f"**Story grade:** {grade}/10")
    if angles:
        lines.append("")
        lines.append("**Per-outlet framing:**")
        for oid, line in angles.items():
            lines.append(f"- `{oid}` — {line}")
    if divergence:
        lines.append("")
        lines.append("**Framing divergence:**")
        for d in divergence:
            outlets = ", ".join(d.get("outlets", []))
            lines.append(f"- [{outlets}] {d.get('frame', '')}")
    if sway:
        direction = sway.get("shift_direction", "?")
        mag = sway.get("magnitude_pp", 0)
        dur = sway.get("duration_days", 0)
        lines.append("")
        lines.append(f"**Public-opinion sway:** {direction} · ~{mag}pp over ~{dur}d")
        if sway.get("cohorts_moved_positive"):
            lines.append(f"- ↗ {', '.join(sway['cohorts_moved_positive'])}")
        if sway.get("cohorts_moved_negative"):
            lines.append(f"- ↘ {', '.join(sway['cohorts_moved_negative'])}")
        if sway.get("rationale"):
            lines.append(f"- _{sway['rationale']}_")
    if watch:
        lines.append("")
        lines.append(f"**Next inflection:** {watch}")
    return "\n".join(lines)


_ENDED_REASON_PHRASE: dict[str, str] = {
    "stalemate": "Stalemate — no convergence inside the turn cap",
    "consensus": "Panel converged via explicit agreement round",
    "repetition": "Debate stalled — repeated arguments",
    "no_panel": "No panel available",
}


def _format_conversation(conversation: dict[str, Any] | None) -> str:
    if not conversation:
        return "_No persona conversation recorded._"
    turns = conversation.get("turns") or []
    if not turns:
        return "_No persona conversation recorded._"
    moderator = conversation.get("moderator_take") or ""
    consensus = conversation.get("consensus_probability_pct")
    panel = conversation.get("panel") or []
    da_id = conversation.get("devil_advocate_id")
    ended = str(conversation.get("ended_reason") or "")
    convergence_note = (conversation.get("convergence_note") or "").strip()
    debate_turns = [t for t in turns if not t.get("is_agreement_turn")]
    agreement_turns = [t for t in turns if t.get("is_agreement_turn")]

    head: list[str] = []
    if panel:
        # Mark the chosen panelist inline so readers see "walter 🔥" rather than just a name.
        annotated = [(f"{p} 🔥" if p == da_id else p) for p in panel]
        head.append(f"**Panel:** {', '.join(annotated)}")
        if da_id:
            head.append(f"🔥 _Devil's advocate: `{da_id}`_")
    phrase = _ENDED_REASON_PHRASE.get(ended, ended or "unknown")
    if ended == "stalemate":
        phrase = f"Stalemate — no convergence after {len(turns)} turns"
    if consensus is not None:
        head.append(f"**Outcome:** {phrase} — final read **{consensus}c YES**")
    else:
        head.append(f"**Outcome:** {phrase}")
    head.append(
        f"**Turns:** {len(debate_turns)} debate + {len(agreement_turns)} agreement = {len(turns)} total"
    )
    if convergence_note:
        head.append(f"**Trajectory:** _{convergence_note}_")
    if moderator:
        head.append(f"**Take:** {moderator}")

    def _line(t: dict[str, Any]) -> str:
        pid = t.get("persona_id", "?")
        pct = t.get("probability_pct")
        pct_str = f" ({pct}c)" if pct is not None else ""
        badge = " 🔥" if t.get("is_devil_advocate") else ""
        return f"- **{pid}**{badge}{pct_str}: {t.get('text', '').strip()}"

    body_lines: list[str] = []
    if debate_turns:
        body_lines.append("### Debate")
        body_lines.extend(_line(t) for t in debate_turns)
    if agreement_turns:
        body_lines.append("")
        body_lines.append("### Agreement round")
        body_lines.extend(_line(t) for t in agreement_turns)
    return "\n".join(head) + "\n\n" + "\n".join(body_lines)


def _format_market(market: dict[str, Any] | None) -> str:
    if not market:
        return "_No Polymarket angle this run — story is the deliverable._"
    url = market.get("market_url")
    slug = market.get("proposed_market_slug")
    proposed = market.get("proposed_market") or {}
    if url:
        line = f"**Live Polymarket market:** [{url}]({url})"
        if market.get("price_cents") is not None:
            line += f"  \nYES ≈ **{market['price_cents']}c**"
        return line
    if slug:
        title = proposed.get("title") or slug
        return (
            f"**Polymarket angle (proposed market):** _{title}_  \n"
            f"slug: `{slug}` — full spec in `higgsfield.json#polymarket`"
        )
    return "_No Polymarket angle this run — story is the deliverable._"


def write_event(
    folder: Path,
    *,
    event_id: str,
    brief: dict[str, Any],
    sources: list[dict[str, Any]],
    conversation: dict[str, Any] | None,
    market: dict[str, Any] | None,
    hero_path: Path | None,
    video_path: Path | None = None,
    hero_skip_reason: str | None = None,
) -> None:
    """Emit the four output files for one event (plus ``video.mp4`` when Higgsfield ran)."""
    folder.mkdir(parents=True, exist_ok=True)

    higgs = dict(brief.get("higgsfield") or {})
    if hero_path and hero_path.exists():
        higgs["ref_image_path"] = f"./{hero_path.name}"
    if video_path and video_path.exists():
        higgs["video_path"] = f"./{video_path.name}"
    if market:
        higgs["polymarket"] = market
    (folder / "higgsfield.json").write_text(
        json.dumps(higgs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    (folder / "sources.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if conversation is not None:
        (folder / "conversation.json").write_text(
            json.dumps(conversation, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    outlet_counts: dict[str, int] = {}
    lean_counts: dict[str, int] = {}
    for s in sources:
        outlet_counts[s.get("outlet_id", "?")] = outlet_counts.get(s.get("outlet_id", "?"), 0) + 1
        lean_counts[s.get("outlet_lean", "unknown")] = lean_counts.get(s.get("outlet_lean", "unknown"), 0) + 1
    outlets_line = ", ".join(f"{k} ({v})" for k, v in sorted(outlet_counts.items(), key=lambda x: -x[1]))
    lean_line = ", ".join(f"{k}={v}" for k, v in sorted(lean_counts.items(), key=lambda x: -x[1]))

    sources_section = _format_sources(sources)
    cross_section = _format_cross_outlet(brief)
    panel_section = _format_conversation(conversation)
    market_section = _format_market(market)

    hook = brief.get("hook") or event_id
    higgs_summary = brief.get("higgsfield") or {}
    higgs_prompt = higgs_summary.get("prompt") or ""
    higgs_camera = higgs_summary.get("camera_move") or "static"
    higgs_ratio = higgs_summary.get("aspect_ratio") or "9:16"
    higgs_dur = higgs_summary.get("duration_s") or 5

    if hero_path and hero_path.exists():
        hero_block = f"![hero](./{hero_path.name})"
    elif hero_skip_reason:
        hero_block = f"_hero skipped — {hero_skip_reason}_"
    else:
        hero_block = "_(no hero rendered)_"

    video_block = ""
    if video_path and video_path.exists():
        video_block = (
            "\n## Video\n\n"
            f"Rendered by Higgsfield ({higgs_camera}, {higgs_ratio}, {higgs_dur}s). "
            f"File: [`{video_path.name}`](./{video_path.name}).\n\n"
            f"![preview](./{video_path.name})\n"
        )

    jjj = brief.get("_jjj") or {}
    jjj_line = ""
    jjj_notes = (jjj.get("notes") or "").strip() if isinstance(jjj, dict) else ""
    if jjj_notes:
        jjj_line = f"\n_Edited by JJJ: {jjj_notes}_\n"

    body = f"""# {hook}

**Who's posting:** {outlets_line}  
**Lean mix:** {lean_line}

{sources_section}
## Cross-outlet read

{cross_section}

## Persona panel

{panel_section}

## Polymarket angle

{market_section}

## Hero (reference image for Higgsfield)

{hero_block}

## Higgsfield video prompt

```
{higgs_prompt}
```

- camera: `{higgs_camera}`
- aspect: `{higgs_ratio}`
- duration: `{higgs_dur}s`
- ref image: {f"`./{hero_path.name}`" if hero_path and hero_path.exists() else "_none — Higgsfield will need an external reference still_"}

Drop `higgsfield.json` into the Higgsfield API to render.
{video_block}{jjj_line}
_Event ID: `{event_id}`._
"""
    (folder / "README.md").write_text(body, encoding="utf-8")


def write_index(output_dir: Path) -> Path:
    """Rebuild ``output/README.md`` — a one-glance table of every event."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for f in sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        readme = f / "README.md"
        higgs = f / "higgsfield.json"
        if not readme.exists():
            continue
        hook = readme.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        higgs_data: dict[str, Any] = {}
        if higgs.exists():
            try:
                higgs_data = json.loads(higgs.read_text(encoding="utf-8"))
            except Exception:
                pass
        rows.append({
            "folder": f.name,
            "hook": hook,
            "camera": higgs_data.get("camera_move", "?"),
            "ratio": higgs_data.get("aspect_ratio", "?"),
        })
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Output",
        "",
        f"_Generated {now} — {len(rows)} event{'s' if len(rows) != 1 else ''}._",
        "",
        "Each event folder is a complete brief: sources, cross-outlet read, persona panel, hero PNG, and a Higgsfield-ready video prompt. Open the folder's `README.md` for the human-readable summary.",
        "",
    ]
    if not rows:
        lines.append("_No events yet — run `python -m app.main run` to populate._")
    else:
        lines += ["| # | Story | Camera | Aspect |", "|---:|---|---|:-:|"]
        for i, r in enumerate(rows, 1):
            link = f"[`{r['folder']}`](./{r['folder']}/README.md)"
            lines.append(f"| {i} | **{r['hook'][:90]}**<br/>{link} | `{r['camera']}` | `{r['ratio']}` |")
    p = output_dir / "README.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
