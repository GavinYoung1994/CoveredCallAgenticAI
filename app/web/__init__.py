"""Local web UI: the agent chat + workflow buttons + live log console.

``server.create_app`` is a Flask app factory with injectable dependencies (agent,
workflow runners, log buffer) so it is testable without the LLM or live APIs.
"""

from app.web.server import create_app

__all__ = ["create_app"]
