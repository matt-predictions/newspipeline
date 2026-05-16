# newspipeline

News wire in, Higgsfield-ready video brief out. One LLM-judged
five-persona panel debate per story, polished by a post-panel editor,
shipped as a content-addressed brief folder.

```
RSS + X feeds  ─►  cluster (cohesion-gated)  ─►  brief writer
                                                    │
              ┌─────────────────────┐               ▼
              │  Polymarket angle   │           5-persona panel
              │  (live or proposed) │             (debate → agreement)
              └──────────┬──────────┘               │
                         │                          ▼
                         └────────►  JJJ editor (post-panel rewrite)
                                              │
                                              ▼
                                      hero image (16:9)
                                              │
                                              ▼
                                  Higgsfield video prompt
                                              │
                                              ▼
                          output/<event>/  +  runs/<date>/<hash>/manifest.json
```

## What you need

- Python 3.11+
- At least one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`. Both is best.

| Keys set                  | Brief     | Panel     | Proposer  | Hero image |
|---------------------------|-----------|-----------|-----------|------------|
| both                      | OpenAI    | Anthropic | Anthropic | yes        |
| only `OPENAI_API_KEY`     | OpenAI    | OpenAI    | OpenAI    | yes        |
| only `ANTHROPIC_API_KEY`  | Anthropic | Anthropic | Anthropic | **skipped**|

Anthropic-only runs surface a `_hero skipped — no OPENAI_API_KEY_` note
in the event README; everything else still lands.

## Setup & run

```bash
git clone <this repo> && cd newspipeline
cp .env.example .env       # paste your key(s) into .env
./run                      # creates .venv, installs deps, ingests, writes 5 briefs
```

CLI surface:

```bash
./run -n 1                              # one story
./run --dedup-hash <hash>               # target a specific cluster
./run --help                            # full CLI
./eval                                  # judge the latest event folder
```

Every step prints `[N/M] step (Xs)` with elapsed seconds. The panel
emits one line per turn so a 20-30 turn debate is fully observable as
it runs — no silent waits.

## What you get

Two layouts. The CI workflow continues to commit `output/`; the
content-addressed `runs/` folder is the replayable audit trail.

```
output/<event>/                           # human-readable
    README.md           wire summary + sources + panel + video prompt
    hero.png            16:9 reference still
    higgsfield.json     wire-droppable API body — drop straight in
    meta.json           camera_move + polymarket angle + JJJ notes
    conversation.json   five-persona Polymarket-style debate
    sources.json        raw article URLs + outlet metadata
    manifest.json       per-step latency + token cost (mirror of runs/)
    video.mp4           optional — only when ENABLE_HIGGSFIELD_RENDER=true

runs/<YYYY-MM-DD>/<cluster_hash[:12]>/    # content-addressed audit trail
    manifest.json       canonical append-only event log
```

Plus `output/README.md` — a one-glance table of every committed event.

### `higgsfield.json` shape

Matches the Higgsfield queue API verbatim — drop into `POST /{model_id}`
without renaming a single field:

```json
{
  "prompt": "Camera: dolly in. ...",
  "image_url": "./hero.png",
  "duration": 8,
  "aspect_ratio": "16:9"
}
```

`image_url` is a path relative to the event folder; the caller swaps in
a real URL or data-URL at submit time. Non-API fields (camera_move,
polymarket angle, JJJ edit notes) live in `meta.json` so they're still
inspectable without polluting the wire body.

## Architecture

```
app/
  main.py                       CLI: ingest | run
  story/
    pipeline.py                 orchestrator (linear, manifested)
    write.py                    emits the per-event output folder
  cluster/                      embedding + cohesion gate
  ingest/                       RSS + Nitter pollers
  agents/
    brief.py                    one-shot cross-outlet brief writer
    base.py                     retry + cost + AgentResult
    schemas.py                  Pydantic models at every LLM boundary
    _loader.py                  reads persona folders below
    aisha/ persona.md skill.md      ── panelists (5 + 3 bench)
    devon/  …
    james/  …
    maria/  …
    robert/ …
    sam/    …                  bench (not in default panel)
    tom/    …
    walter/ …
    jjj/   persona.md skill.md
           agent.py             post-panel editor (rewrites hook +
                                higgsfield prompt to lean into the
                                framing that WON the debate)
  personas/
    conversation.py             two-phase panel runner (debate →
                                agreement → consensus | stalemate |
                                repetition)
    panel.py                    shim around app.agents._loader
  polymarket/
    client.py                   Gamma API client
    matcher.py                  live-market matcher
    proposer.py                 propose-a-market fallback
  render/
    higgsfield.py               image-to-video API client
  imagegen.py                   hero render (gpt-image-1, 16:9)
  prompts/                      flat .md files with YAML frontmatter
  core/
    config.py                   pydantic-settings .env loader
    db.py                       sqlite wrapper (articles, spend, events)
    manifest.py                 append-only run manifest
    log.py                      single-line timestamped progress
    jsonx.py                    extract first JSON object from a string
    models.py                   Article + ResearchOut

evals/
  consensus_quality.py          LLM-judge: convergence, calibration,
                                persona fidelity
  style.py                      LLM-judge: hook punch, higgsfield
                                voice, README freshness
  run.py                        ./eval entrypoint
```

### Prompt frontmatter

Every prompt under `app/prompts/*.md` carries YAML frontmatter:

```
---
model: openai:gpt-4o
temperature: 0.4
purpose: One-shot user prompt for the cross-outlet brief
inputs:
  today_iso: "ISO date string the run is pinned to"
  cluster: "Per-outlet headline ledger"
  ...
output_format: json_object
output_schema: app.agents.schemas:BriefOut
owner: newspipeline-core
version: 1.2.0
---
<body>
```

Load with `from app.prompts import load_prompt, prompt_meta`. Body is
returned by `load_prompt(name, **subs)`; the parsed `PromptMeta` is
returned by `prompt_meta(name)`.

### Structured outputs

Every LLM boundary parses against a Pydantic schema in
`app.agents.schemas`:

| Boundary | Schema |
|---|---|
| `write_brief` | `BriefOut` |
| `edit_brief` (JJJ) | `JJJEdit` |
| `_one_turn` (panel) | `PanelTurnOut` |
| `propose_market` | `ProposedMarket` (in `app.polymarket.proposer`) |

`parse_structured(text, model)` does JSON extraction + validation; on
failure it raises `StructuredOutputError` which callers handle with a
safe fallback (the run survives one bad LLM response).

## Video (optional)

Higgsfield image-to-video is wired and ready. Add
`HIGGSFIELD_API_KEY=KEY_ID:KEY_SECRET` to `.env` and set
`ENABLE_HIGGSFIELD_RENDER=true`. Without those, the pipeline ships a
complete brief (hero image + wire-droppable `higgsfield.json`) you can
paste into Higgsfield by hand.

## Evals

```bash
./eval                          # latest event folder
./eval output/<event>           # explicit target
```

Two judges run in parallel; a `evals.json` sidecar lands next to the
event so scores can be diff'd over time.

## CI

`.github/workflows/run.yml` runs the same thing on a 30-min cron and
commits `output/` back to `main`. Add the API keys as repo secrets.

## Roadmap

See `ROADMAP.md`. Highlights of what's deferred:

- Vector embeddings persisted across runs (so the same story doesn't
  resplit into two clusters)
- Persona long-term memory (Aisha remembers her last N takes across
  runs)
- DAG / Prefect-style orchestration with step-level retries + parallel
  poly+brief
- Langfuse trace integration
- MCP server exposing each persona as a tool
- Multi-scene Higgsfield render
