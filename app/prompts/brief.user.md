---
model: openai:gpt-4o
temperature: 0.4
purpose: One-shot user prompt that turns a clustered news story into a brief
owner: newspipeline-core
version: 1.2.0
inputs:
  today_iso: ISO date string (YYYY-MM-DD) the run is pinned to
  cluster: Multi-line per-outlet headline + summary ledger
  market_block: Optional live/proposed Polymarket context (may be empty)
  moves: Pipe-separated list of allowed Higgsfield camera moves
output_format: json_object
output_schema: app.agents.schemas:BriefOut
---
TODAY IS {today_iso}. Use this date for any forward-looking horizon. NEVER carry years over from your training data; if a fact happened earlier (e.g. "in 2023"), state it verbatim — but every horizon, deadline, or "next inflection" date must be on or after {today_iso}.

CLUSTER (one story, multiple outlets, verbatim posts):
{cluster}
{market_block}
Return STRICT JSON with EXACTLY these keys (no extras, no missing fields, no prose outside the object):

- "hook": one-line headline summarizing the story in the news-desk's voice (max 110 chars). Declarative — no hedging.
- "per_outlet_angle": object mapping outlet_id -> one-sentence summary of what THIS outlet uniquely emphasizes (cite their wording). Mark wires (AFP/AP/Reuters) as "wire repeat" when they're not adding distinctive framing.
- "convergent_facts": array of 3-6 facts every outlet agrees on (verbatim-grounded).
- "divergent_framings": array of objects {{"outlets": [...], "frame": "..."}} showing real framing splits.
- "public_opinion_sway": {{"shift_direction": "positive" | "negative" | "polarizing" | "muted", "magnitude_pp": integer 1-25, "duration_days": integer, "cohorts_moved_positive": [1-4 audience labels], "cohorts_moved_negative": [1-4 audience labels], "rationale": 1-2 sentences quoting verbatim cluster phrasing}}.
- "story_grade": integer 1-10 — wire weight (10 = top of every front page; 3 = filler).
- "what_to_watch_next": 1-2 sentences on the next inflection point.
- "higgsfield": {{
    "prompt": 2-4 sentence cinematic prompt for Higgsfield image-to-video. NEVER name living public figures — use symbolic stand-ins (silhouettes, empty podiums, name plates, flags). Include lighting and composition cues. No rendered text overlays. No on-camera quotes.
    "camera_move": one of [{moves}]
    "aspect_ratio": "16:9" (default — matches the LANDSCAPE hero image) | "9:16" (only for explicitly mobile-vertical stories) | "1:1"
    "duration_s": 5 | 8 | 12 (integer)
  }}
- "hero_image_prompt": one paragraph for the LANDSCAPE 16:9 reference still that Higgsfield uses as the first frame. Off-black background (#0F1115) edge-to-edge, single cyan accent (#1AB4E0) on the focal element, warm orange (#E07A2F) only on risk/alert beats, solid black silhouettes with cyan rim-light if any figures appear (NEVER named public figures, NEVER faces). ≤8 words of text in the frame. Compose for 16:9 — the still and the video share the same aspect ratio.

Hard rules:
- All horizons / forward-looking dates must be on or after {today_iso}.
- Never invent quotes — only echo phrasing that appears verbatim in the cluster.
- The video prompt and the hero prompt must be visually different (the video moves; the still freezes a beat) but share the same aspect ratio.

No fields may be missing. No prose outside the JSON.
