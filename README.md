# newspipeline

POC: news wire → AI breakdown → Higgsfield-ready video brief.

Polls RSS feeds and X/Twitter (via Nitter), clusters the same story across
outlets, lets a small persona panel debate the public-opinion impact, and
emits a per-event folder with everything you need to hand to Higgsfield's
image-to-video API: a hero reference image, a video prompt, a camera move,
and the persona conversation transcript.

## What you get per story

```
output/<event>/
  README.md           wire summary + sources + cross-outlet read + panel + video prompt
  hero.png            reference image (off-black bg, silhouettes only — no faces)
  higgsfield.json     {prompt, camera_move, aspect_ratio, duration_s, ref_image_path}
  conversation.json   persona panel transcript
  sources.json        raw cluster article URLs
```

Plus an auto-generated `output/README.md` index of every story.

## Quickstart

```bash
git clone <this repo>
cd newspipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: ANTHROPIC_API_KEY=..., OPENAI_API_KEY=...

python -m app.main ingest      # poll RSS + X into sqlite
python -m app.main run         # pick the top cluster, write the event folder
```

To target a specific cluster:

```bash
python -m app.main run --dedup-hash <hash>
```

## How it works

1. **`ingest`** polls RSS feeds in `feeds.yml` and X accounts in
   `twitter_sources.yml` (via rotating Nitter instances). Writes to
   `data/pipeline.db`.
2. **`run`** loads recent articles, embeds them, walks cohesion thresholds
   (mean pairwise cosine ≥ 0.58, ≥ 2 distinct outlets) to find candidate
   clusters, and picks the top one.
3. One LLM call (`app/agents/brief.py`) reads the cluster and returns
   cross-outlet framing, public-opinion sway, a Higgsfield video prompt,
   and a hero image prompt.
4. 4 personas (`app/personas/cards/*.yml`) debate the story's probability
   in a round-robin conversation.
5. Polymarket: look up a live market for the entities; otherwise propose
   a clean spec with a real future horizon.
6. Render the hero image and write the 4-file output folder.

## Architecture

```
app/
  main.py              CLI: ingest, run
  imagegen.py          render_hero(prompt, out_path)
  core/                config, db, models, jsonx, urlutil
  ingest/              rss + twitter_nitter
  cluster/             embedding + cohesion + candidates
  agents/
    base.py            LLM helpers (retry, cost, model fallback)
    brief.py           single LLM call → cross-outlet brief + Higgsfield prompt
  personas/            panel + round-robin conversation
  polymarket/          live matcher + market proposer (date-pinned, never
                       hallucinates years)
  story/
    pipeline.py        end-to-end driver (~210 LOC)
    write.py           emits the 5 event files + output/README.md index
```

~3,600 LOC, no tests, no debate loops, no scaffolding. POC-scale.

## GitHub Actions

`.github/workflows/run.yml` runs every 30 minutes on cron: ingests, runs
the top candidate, commits `output/` back to main. Add `ANTHROPIC_API_KEY`
and `OPENAI_API_KEY` as repo secrets.

## Env

See `.env.example`. Only `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are
required — everything else has sensible defaults.
