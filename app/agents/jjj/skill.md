# What JJJ does

A single LLM call that runs AFTER the persona panel resolves. Two
outputs:

- A new `hook` line (≤ 100 chars, declarative, no "may/could/might")
- A new `higgsfield_prompt` (2-4 sentences, leaning into the framing
  that won the debate)

Plus a one-line `edit_notes` explaining the call.

## Decision rule

`ended_reason` from the conversation transcript drives the framing
choice:

| ended_reason | What JJJ leans into |
| --- | --- |
| `consensus` | Whichever side absorbed the other (room vs DA). The compelling-turn excerpts show direction. |
| `stalemate` | The higher-stakes framing of the two camps. |
| `repetition` | Whichever framing the panel kept circling on. |
| `no_panel` | Don't change the framing — only sharpen voice. |

JJJ is forbidden from inventing facts, changing entity names, or
rewriting `hero_image_prompt` (the hero has already been rendered or
committed to; we don't re-roll the visual brief).

## Costs

One Sonnet (Anthropic) or `gpt-4o` (OpenAI) call. Output is small (≤ 1k
tokens). Failure is non-fatal — the brief survives untouched with a
warning under `_jjj.error`.
