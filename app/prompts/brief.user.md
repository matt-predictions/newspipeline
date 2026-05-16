TODAY IS {today_iso}. Use this date for any horizon — never carry years over from training data.

CLUSTER (one story, multiple outlets, verbatim posts):
{cluster}
{market_block}
Return STRICT JSON with EXACTLY these keys:

- "hook": one-line headline summarizing the story in the news-desk's voice (max 110 chars).
- "per_outlet_angle": object mapping outlet_id -> one-sentence summary of what THIS outlet uniquely emphasizes (cite their wording). Mark wires (AFP/AP/Reuters) as "wire repeat" when they're not adding distinctive framing.
- "convergent_facts": array of 3-6 facts every outlet agrees on (verbatim-grounded).
- "divergent_framings": array of objects {{outlets:[...], frame:"..."}} showing real framing splits.
- "public_opinion_sway": {{shift_direction: "positive"|"negative"|"polarizing"|"muted", magnitude_pp: 1-25 integer, duration_days: integer, cohorts_moved_positive: [1-4 audience labels], cohorts_moved_negative: [1-4 audience labels], rationale: 1-2 sentences quoting verbatim cluster phrasing}}.
- "story_grade": integer 1-10 — wire weight (10=top of every front page; 3=filler).
- "what_to_watch_next": 1-2 sentences on the next inflection point.
- "higgsfield": {{
    prompt: 2-4 sentence cinematic prompt for Higgsfield video API. NEVER name living public figures — use symbolic stand-ins (silhouettes, empty podiums, name plates, flags). Include lighting and composition cues, but no rendered text overlays.
    camera_move: one of [{moves}]
    aspect_ratio: "9:16" | "16:9" | "1:1"
    duration_s: 5 | 8 | 12 (integer)
  }}
- "hero_image_prompt": one paragraph for the reference still. Off-black background (#0F1115) edge-to-edge, single cyan accent (#1AB4E0) on the focal element, warm orange (#E07A2F) only on risk/alert beats, solid black silhouettes with cyan rim-light if any figures appear (NEVER named public figures, NEVER faces). ≤8 words of text in the frame.

Hard rules:
- All horizons / dates referenced anywhere must be on or after {today_iso}.
- Never invent quotes — only echo phrasing that appears verbatim in the cluster.
- The video prompt and hero prompt must be visually different (the video moves; the still freezes a beat).

No fields may be missing. No prose outside the JSON.
