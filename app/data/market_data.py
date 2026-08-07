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


def get_earnings_client(llm: Any = None, provider: str | None = None) -> Any:
    """Return the composite earnings-date client with the configured PRIMARY
    provider tried first (``EARNINGS_PROVIDER`` = 'finnhub' | 'massive'), the
    other kept as a fallback, then the Google-search engine if enabled. Each
    provider degrades to None (unknown → flagged) so the chain never crashes."""
    from app.data.earnings_client import EarningsClient
    from app.data.massive_earnings import MassiveEarningsClient
    from app.data.earnings_search import EarningsSearchClient, CompositeEarningsClient

    name = (provider or settings.earnings_provider or "finnhub").strip().lower()
    finnhub, massive = EarningsClient(), MassiveEarningsClient()
    providers = [massive, finnhub] if name == "massive" else [finnhub, massive]
    if name not in ("finnhub", "massive"):
        logger.warning("Unknown EARNINGS_PROVIDER %r; defaulting to Finnhub first.", name)
    logger.info("Earnings provider order: %s",
                ", ".join(type(p).__name__ for p in providers))
    if settings.earnings_search_enabled:
        providers.append(EarningsSearchClient(llm=llm))
    return CompositeEarningsClient(providers)
