"""LLM-judge eval harness.

Two judges, both `gpt-4o`-backed and consistent in shape:

- ``evals.consensus_quality`` scores a ``conversation.json`` on
  convergence, calibration, and persona fidelity.
- ``evals.style`` scores the event README and the JJJ output for "Joynt
  voice" — declarative, no hedging, on-frame.

Each judge returns a ``EvalScore`` with subscores (0-10) plus a one-
paragraph rationale. The ``./eval`` (and ``make eval``) entrypoint runs
both on the latest event folder under ``output/`` and prints a table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalScore:
    name: str
    overall: float            # 0..10
    subscores: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "overall": self.overall,
            "subscores": self.subscores,
            "rationale": self.rationale,
        }


__all__ = ["EvalScore"]
