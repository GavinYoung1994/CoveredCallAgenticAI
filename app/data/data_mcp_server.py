"""FastMCP server exposing EVERY data-API method as an MCP tool.

Wraps the real client classes (single source of truth):
  * SchwabClient        — quotes, fundamentals, price history, option chains,
                          expirations, instruments, optionable check.
  * NewsClient          — massive.com headlines (+ full-article fetch), raw feed.
  * EarningsClient      — Finnhub earnings calendar / next date.
  * EarningsSearchClient— Google-search earnings date (+ raw search).

Clients are built lazily (first tool call) so importing this module touches no
network/credentials. Run:  ./venv/bin/python -m app.data.data_mcp_server
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server.fastmcp import FastMCP

from app.data.schwab_client import SchwabClient
from app.data.news_client import NewsClient
from app.data.earnings_client import EarningsClient
from app.data.earnings_search import EarningsSearchClient

logging.basicConfig(level=logging.INFO)
mcp = FastMCP("Covered Call Data APIs")

# Lazy singletons — constructed on first use (avoids httpx/SSL + credential
# needs at import time).
_clients: Dict[str, Any] = {}


def _schwab() -> SchwabClient:
    return _clients.setdefault("schwab", SchwabClient())


def _news() -> NewsClient:
    return _clients.setdefault("news", NewsClient())


def _earnings() -> EarningsClient:
    return _clients.setdefault("earnings", EarningsClient())


def _earnings_search() -> EarningsSearchClient:
    return _clients.setdefault("earnings_search", EarningsSearchClient())


# ── Charles Schwab market data ─────────────────────────────────────────
@mcp.tool()
def schwab_get_quote(symbol: str, fields: str = "quote,fundamental,reference") -> Dict[str, Any]:
    """Single-symbol Schwab quote (quote + fundamental + reference blocks)."""
    return _schwab().get_quote(symbol, fields=fields)


@mcp.tool()
def schwab_get_quotes(symbols: List[str], fields: str = "quote,fundamental,reference") -> Dict[str, Any]:
    """Batch Schwab quotes for many symbols (chunked + rate-limited)."""
    return _schwab().get_quotes_chunked(symbols, fields=fields)


@mcp.tool()
def schwab_get_fundamentals(symbol: str) -> Dict[str, Any]:
    """Normalized fundamentals for one symbol: price, avg volume, dividend yield."""
    client = _schwab()
    return client.extract_fundamentals(client.get_quotes([symbol]), symbol)


@mcp.tool()
def schwab_get_price_history(
    symbol: str, period_type: str = "month", period: int = 6,
    frequency_type: str = "daily", frequency: int = 1,
) -> Dict[str, Any]:
    """Historical candlestick (OHLCV) data for technical analysis."""
    return _schwab().get_price_history(
        symbol, period_type=period_type, period=period,
        frequency_type=frequency_type, frequency=frequency)


@mcp.tool()
def schwab_get_option_chain(
    symbol: str, contract_type: str = "CALL", strike_count: int = 15,
    range_filter: str = "OTM", from_date: Optional[str] = None, to_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Option chain (defaults to OTM calls) for covered-call selection."""
    return _schwab().get_option_chain(
        symbol, contract_type=contract_type, strike_count=strike_count,
        range_filter=range_filter, from_date=from_date, to_date=to_date)


@mcp.tool()
def schwab_get_option_expirations(symbol: str) -> Dict[str, Any]:
    """Expiration timeline for an optionable equity."""
    return _schwab().get_option_expirations(symbol)


@mcp.tool()
def schwab_lookup_instrument(symbol: str, projection: str = "symbol-search") -> Dict[str, Any]:
    """Instrument/fundamental lookup by symbol pattern."""
    return _schwab().lookup_instrument(symbol, projection=projection)


@mcp.tool()
def schwab_is_optionable(symbol: str) -> Dict[str, Any]:
    """Whether a symbol has listed options (probes the expiration chain)."""
    return {"symbol": symbol.upper(), "is_optionable": _schwab().is_optionable(symbol)}


# ── massive.com news ───────────────────────────────────────────────────
@mcp.tool()
def news_get_headlines(ticker: str, limit: int = 10, fetch_content: bool = False) -> Dict[str, Any]:
    """Recent news headlines for a ticker (optionally with full article bodies)."""
    return {"ticker": ticker.upper(),
            "headlines": _news().get_headlines(ticker, limit=limit, fetch_content=fetch_content)}


@mcp.tool()
def news_get_raw(ticker: str, limit: int = 10) -> Dict[str, Any]:
    """Raw massive.com news API response for a ticker."""
    return _news().get_news_raw(ticker, limit=limit)


@mcp.tool()
def news_fetch_article_text(url: str) -> Dict[str, Any]:
    """Best-effort fetch of an article's body text from its URL."""
    return {"url": url, "text": _news().fetch_article_text(url)}


# ── earnings (Finnhub) ─────────────────────────────────────────────────
@mcp.tool()
def earnings_finnhub_next_date(symbol: str, from_date: str, to_date: str) -> Dict[str, Any]:
    """Next earnings date (YYYY-MM-DD) from Finnhub within a window, or null."""
    return {"symbol": symbol.upper(),
            "next_earnings_date": _earnings().get_next_earnings_date(symbol, from_date, to_date)}


@mcp.tool()
def earnings_finnhub_calendar(symbol: str, from_date: str, to_date: str) -> Dict[str, Any]:
    """Raw Finnhub earnings-calendar rows for a symbol in a date window."""
    return {"symbol": symbol.upper(),
            "calendar": _earnings().get_earnings_calendar(symbol, from_date, to_date)}


# ── earnings (Google-search engine, LLM-assisted) ──────────────────────
@mcp.tool()
def earnings_search_next_date(
    symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None
) -> Dict[str, Any]:
    """Next earnings date via web search (found or inferred quarterly), or null."""
    return {"symbol": symbol.upper(),
            "next_earnings_date": _earnings_search().get_next_earnings_date(symbol, from_date, to_date)}


if __name__ == "__main__":
    mcp.run()
