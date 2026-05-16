# newspipeline

News wire in, Higgsfield-ready video brief out. Five briefs per run.

## What you need

- Python 3.11+
- At least one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`. Both is best.

| Keys set                | Brief    | Panel     | Proposer  | Hero image | Sora |
|-------------------------|----------|-----------|-----------|------------|------|
| both                    | OpenAI   | Anthropic | Anthropic | yes        | opt  |
| only `OPENAI_API_KEY`   | OpenAI   | OpenAI    | OpenAI    | yes        | opt  |
| only `ANTHROPIC_API_KEY`| Anthropic| Anthropic | Anthropic | **skipped**| skip |

Anthropic-only runs surface a `_hero skipped — no OPENAI_API_KEY_` note in
the event README; everything else still lands.

## Setup & run

```bash
git clone <this repo> && cd newspipeline
cp .env.example .env       # paste your key(s) into .env
./run                      # creates .venv, installs deps, ingests, writes 5 briefs
```

`./run -n 1` for a single brief, `./run --dedup-hash <hash>` to retry one
specific cluster, `./run --help` for the CLI. Every step prints a
timestamped line; final JSON summary at the end so you can pipe it.

## What you get

`output/<event>/` for each brief:

```
README.md           wire summary + sources + cross-outlet read + panel + video prompt
hero.png            reference still (off-black bg, silhouettes only, no faces)
higgsfield.json     {prompt, camera_move, aspect_ratio, duration_s, ref_image_path}
conversation.json   four-persona Polymarket-style probability debate
sources.json        raw article URLs + outlet metadata
video.mp4           optional — only when ENABLE_SORA_RENDER=true
```

Plus `output/README.md` — a one-glance table of every event written so far.

## Sora video (optional)

Set `ENABLE_SORA_RENDER=true` in `.env` (and optionally `SORA_MODEL_ID=sora-2-pro`).
Each story will additionally render a `video.mp4` using the hero PNG as the
input reference. Sora burns money — leave it off unless you mean it.

## How it works

`ingest` polls feeds in `feeds.yml` (RSS) and `twitter_sources.yml` (X via
rotating Nitter), writes to `data/pipeline.db`. `run` embeds recent
articles, walks cohesion thresholds to find cross-outlet clusters, then for
each cluster: one OpenAI brief call, a four-persona Anthropic debate, a
Polymarket live-match-or-propose pass, and a `gpt-image-1` hero render.
Sibling clusters of the same story (high index overlap) are skipped so
`-n 5` returns five actually-different stories.

`.github/workflows/run.yml` runs the same thing on a 30-min cron and
commits `output/` back to `main`. Add the two API keys as repo secrets.
