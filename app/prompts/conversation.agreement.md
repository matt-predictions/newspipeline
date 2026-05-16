---
purpose: Agreement-round brief — final lock-in or explicit break
owner: newspipeline-core
version: 1.1.0
inputs:
  median: Converged room median (cents) the panelist should lock against
  turn_count: How many debate turns preceded this agreement round
output_format: text
---
AGREEMENT ROUND (read FIRST, before persona card):
The panel has converged near {median}c after {turn_count} turns of real debate. This is your FINAL turn — state a final probability AND a one-sentence agreement statement.

If you genuinely cannot agree, you MAY break consensus here — but ONLY if you have a NEW objection you haven't already raised. Don't break consensus just to be different. If you accept the converged read, lock in a number within 5pp of {median} and explicitly endorse the final.

Format your "text" field as a declarative final statement: "I'll lock in at Xc because…" or "I'm breaking consensus — Yc — because [genuinely new objection]". Cite at least one specific persona by name in your reasoning. Your "probability_pct" field is your final number.

