CURRENT HOOK ({current_hook_len} chars):
{current_hook}

CURRENT HIGGSFIELD PROMPT:
{current_prompt}

PANEL OUTCOME:
- ended_reason: {ended}
- final consensus: {consensus_str}
- devil's advocate: {da_id}

MOST COMPELLING TURNS:
{quotes}

Now rewrite. Constraints:
- "hook": ≤ 100 chars, declarative (no "may", "could", "might"), no clickbait. Lead with the action. Preserve entities (people, places, numbers) verbatim.
- "higgsfield_prompt": 2-4 sentences, leans into the framing that WON the debate. If ended_reason was "da_swayed", the DA's framing won — favor it. If "room_swayed", the room's framing won — favor it. If "stalemate" or "max_turns", favor the higher-stakes framing of the two. NEVER name living public figures (silhouettes, empty podiums, name plates, flags only). No rendered text overlays. No on-camera quotes.
- "edit_notes": ≤ 240 chars explaining what you changed and why (which framing you leaned into, what hedging you cut).

Return STRICT JSON only:
{{"hook": "...", "higgsfield_prompt": "...", "edit_notes": "..."}}
