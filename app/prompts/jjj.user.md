---
model: openai:gpt-4o
temperature: 0.35
purpose: JJJ rewrites hook + Higgsfield video prompt after the panel concludes
owner: newspipeline-core
version: 1.2.0
inputs:
  current_hook_len: Length of the current hook (chars) — soft budget signal
  current_hook: The brief's current hook line
  current_prompt: The brief's current Higgsfield video prompt
  ended: ended_reason from the conversation transcript ("consensus" | "stalemate" | "repetition" | "no_panel")
  consensus_str: Final consensus probability (e.g. "62c YES" or "no consensus")
  da_id: The persona slug assigned as devil's advocate for this debate
  quotes: Bullet list of the most compelling transcript turns
output_format: json_object
output_schema: app.agents.schemas:JJJEdit
---
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
- "higgsfield_prompt": 2-4 sentences, lean into the framing that WON the debate. Decide which framing won using `ended_reason` + the final consensus + the panel turns:
    - "consensus" — the panel locked in. Lean into the framing the room converged on; if the devil's advocate moved toward the room, the room's framing won; if the room moved toward the DA, the DA's framing won. The compelling-turn excerpts above show which way.
    - "stalemate" — no lock-in. Favor the higher-stakes framing of the two camps.
    - "repetition" — debate stalled in groupthink. Favor whichever framing the panel kept circling on.
    - "no_panel" — no debate signal. Keep the original framing; only sharpen voice.
  NEVER name living public figures (silhouettes, empty podiums, name plates, flags only). No rendered text overlays. No on-camera quotes.
- "edit_notes": ≤ 240 chars explaining what you changed and why (which framing you leaned into, what hedging you cut).

Return STRICT JSON only (no prose, no fences):
{{"hook": "...", "higgsfield_prompt": "...", "edit_notes": "..."}}
