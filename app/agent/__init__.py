"""The covered-call management agent.

    tools.py       a shared registry of capabilities (data, math, reporting) as
                   typed tools — one source of truth.
    agent.py       a multi-step (ReAct) agent that drives the local LLM to chain
                   those tools — a comprehensive covered-call management expert.
    mcp_server.py  exposes the same tools over MCP (usable by any MCP client).
"""

from app.agent.tools import build_tools, Tool, tool_catalog_text
from app.agent.agent import CoveredCallAgent

__all__ = ["build_tools", "Tool", "tool_catalog_text", "CoveredCallAgent"]
