TODAY IS {today_iso}. The current year is {year_now}. Any horizon you propose MUST be on or after {min_horizon}. Past years are an automatic fail and the pipeline will reject and re-prompt.

Generate a proposed Polymarket market spec from the research below.
{stale_block}
REQUIRED JSON KEYS:
- title: short, plain English. If the title references a year, that year MUST be {year_now} or later (e.g. "Will the Fed cut rates by 50bp before end of Q3 {year_now}?"). NEVER use a past year.
- slug: kebab-case ASCII, ≤ 60 chars. If the slug references a year, that year MUST be {year_now} or later.
- outcome_type: binary | multi | scalar
- outcomes: list of strings (for binary: ["Yes", "No"])
- resolution_criteria: ONE paragraph quoting the headline ledger and naming a source-of-truth (e.g. "FOMC statement on …", "AP confirmed …"). No speculation.
- resolution_source_hint: where a resolver would look (e.g. "FOMC press release", "OFAC sanctions list", "AP/Reuters wire confirmation")
- horizon: human-readable, in the future ("within 30 days", "by end of Q3 {year_now}", "by {next_year}-03-31"). NEVER reference a year before {year_now}.
- horizon_iso: ISO 8601 deadline strictly AFTER {today_iso} (must be ≥ {min_horizon}T00:00:00Z). If you cannot pin one, return null.
- confidence_market_attracts_volume: 0..1 float (lower = niche / unlikely to clear)
- rationale: 2-3 sentences calling out why this market is grounded and what the brief should highlight

HARD RULES:
- You MAY NOT propose markets the cluster cannot resolve (no "will X resign" if cluster doesn't mention resignation).
- You MAY NOT invent statistics. If the cluster only says "tensions rise", confidence stays ≤ 0.2 and outcome_type is "multi".
- Title may NOT mention named individuals' likeness — use roles ("the chair", "the CEO") if unavoidable.
- DO NOT carry over any year from your training data. The clock you trust is the TODAY pin above.

{brand_kit_block}

RESEARCH:
{research_blob}
