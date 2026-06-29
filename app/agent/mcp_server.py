"""FastMCP server exposing the management/analysis tools over MCP.

This makes the whole system's data + quant capabilities available to ANY MCP
client (Claude Desktop, IDE agents, etc.), alongside the existing math_mcp and
schwab_mcp servers. It registers the SAME tools the in-process agent uses
(``app.agent.tools``), so there is a single source of truth.

Run:  ./venv/bin/python -m app.agent.mcp_server
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Make the project importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server.fastmcp import FastMCP

from app.agent.tools import build_tools
from app.manage import ManagementService

logging.basicConfig(level=logging.INFO)
mcp = FastMCP("Covered Call Manager")

# Build the shared registry bound to real services. Memory is loaded lazily so
# importing this module doesn't require chromadb.
_service = ManagementService()


def _memory():
    from app.memory.vector_db import TradeMemory
    return TradeMemory()


_service._memory = None  # set on first learnings call (lazy)
_tools = build_tools(_service)


# Register each registry tool as an MCP tool. We wrap with explicit signatures
# where helpful; for simple ones a generic kwargs passthrough keeps it DRY.
@mcp.tool()
def get_cash() -> Dict[str, Any]:
    """Current account cash balance (USD)."""
    return _tools["get_cash"].run()


@mcp.tool()
def set_cash(amount: float) -> Dict[str, Any]:
    """Set the account cash balance."""
    return _tools["set_cash"].run(amount=amount)


@mcp.tool()
def list_holdings(status: Optional[str] = None) -> Dict[str, Any]:
    """List positions, optionally filtered by status (OPEN/ASSIGNED/LIQUIDATED/EXPIRED)."""
    return _tools["list_holdings"].run(status=status)


@mcp.tool()
def update_holding_status(
    status: str, symbol: Optional[str] = None, position_id: Optional[str] = None,
    stock_sale_price: Optional[float] = None, call_buyback_price: Optional[float] = None,
    contracts: Optional[int] = None,
) -> Dict[str, Any]:
    """Record an executed/closed trade for a holding (by symbol OR position_id).
    ASSIGNED defaults the sale price to the strike; updates cash + realized P&L."""
    return _tools["update_holding_status"].run(
        status=status, symbol=symbol, position_id=position_id, stock_sale_price=stock_sale_price,
        call_buyback_price=call_buyback_price, contracts=contracts)


@mcp.tool()
def performance_report() -> Dict[str, Any]:
    """Portfolio performance: realized P&L, win rate, premium, annualized return on cash."""
    return _tools["performance_report"].run()


@mcp.tool()
def analyze_covered_call(
    underlying_price: float, strike: float, premium: float, days_to_expiration: int,
    volatility: Optional[float] = None, sentiment_score: int = 3,
) -> Dict[str, Any]:
    """Exact quant read on a covered call (yield/Greeks/POP/expected move/score)."""
    return _tools["analyze_covered_call"].run(
        underlying_price=underlying_price, strike=strike, premium=premium,
        days_to_expiration=days_to_expiration, volatility=volatility, sentiment_score=sentiment_score)


@mcp.tool()
def defense_branches(
    entry_stock_price: float, current_stock_price: float, original_premium: float,
    current_call_ask: float, roll_down_premium: float = 0.0,
) -> Dict[str, Any]:
    """Exact P&L for the 3 downside-defense branches (Eject / Roll / Hold)."""
    return _tools["defense_branches"].run(
        entry_stock_price=entry_stock_price, current_stock_price=current_stock_price,
        original_premium=original_premium, current_call_ask=current_call_ask,
        roll_down_premium=roll_down_premium)


@mcp.tool()
def search_learnings(query: str, n: int = 5) -> Dict[str, Any]:
    """Semantic search of past trade lessons / performance reports."""
    if _service._memory is None:
        _service._memory = _memory()
    return _tools["search_learnings"].run(query=query, n=n)


@mcp.tool()
def run_stock_screener() -> Dict[str, Any]:
    """Run the full entry-screener pipeline; returns graded recommendations (long-running)."""
    return _tools["run_stock_screener"].run()


@mcp.tool()
def run_defense_scan() -> Dict[str, Any]:
    """Run downside-defense over all open holdings; returns which breached (long-running)."""
    return _tools["run_defense_scan"].run()


@mcp.tool()
def run_performance_review(period: str = "weekly") -> Dict[str, Any]:
    """Generate a performance report with an LLM narrative (weekly/monthly)."""
    return _tools["run_performance_review"].run(period=period)


if __name__ == "__main__":
    mcp.run()
