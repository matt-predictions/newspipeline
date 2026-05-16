"""LLM agents — one folder per persona/editor.

Layout::

    app/agents/
      aisha/        persona.md + skill.md  (panelist)
      devon/        persona.md + skill.md  (panelist)
      james/        persona.md + skill.md  (panelist)
      maria/        persona.md + skill.md  (panelist)
      robert/       persona.md + skill.md  (panelist)
      sam/          persona.md + skill.md  (panelist, bench)
      tom/          persona.md + skill.md  (panelist, bench)
      walter/       persona.md + skill.md  (panelist, bench)
      jjj/          persona.md + skill.md + agent.py  (post-panel editor)
      brief.py                                  (system: one-shot brief writer)
      base.py                                   (system: retry / cost / AgentResult)
      schemas.py                                (system: Pydantic boundary schemas)
      _loader.py                                (loader for the persona folders)

Top-level entry points::

    from app.agents import load_panelists, load_agents, get_agent
    from app.agents.brief import write_brief
    from app.agents.jjj import edit_brief
"""

from app.agents._loader import (
    AgentSpec,
    get_agent,
    load_agents,
    load_panelists,
)

__all__ = [
    "AgentSpec",
    "load_agents",
    "load_panelists",
    "get_agent",
]
