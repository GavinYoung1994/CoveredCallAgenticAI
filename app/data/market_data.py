"""Market-data provider selection.

One place that decides whether the screener/defense graphs, the MCP server, and
the agent tools talk to Massive (Polygon-compatible, default) or the legacy
Schwab OAuth client. Both expose the SAME interface, so callers are agnostic.

Switch with the ``MARKET_DATA_PROVIDER`` env var ("massive" | "schwab").
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger("market-data")


def get_market_data_client(provider: str | None = None, **kwargs) -> Any:
    """Return the configured market-data client (``MassiveClient`` by default,
    ``SchwabClient`` when the provider is 'schwab'). ``kwargs`` pass through to
    the client constructor (e.g. an injected ``http_client`` in tests)."""
    name = (provider or settings.market_data_provider or "massive").strip().lower()
    if name == "schwab":
        from app.data.schwab_client import SchwabClient
        logger.info("Market-data provider: Schwab.")
        return SchwabClient(**kwargs)
    if name != "massive":
        logger.warning("Unknown MARKET_DATA_PROVIDER %r; defaulting to Massive.", name)
    from app.data.massive_client import MassiveClient
    logger.info("Market-data provider: Massive.")
    return MassiveClient(**kwargs)
