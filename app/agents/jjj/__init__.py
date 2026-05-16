"""JJJ — post-panel editor agent.

Module re-export so callers can keep using ``from app.agents.jjj import
edit_brief`` after the migration to folder-per-agent.
"""

from app.agents.jjj.agent import edit_brief

__all__ = ["edit_brief"]
