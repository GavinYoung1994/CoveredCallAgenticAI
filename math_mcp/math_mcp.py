"""FastMCP wrapper around the deterministic math engine.

This server exposes the SAME pure functions defined in ``app/engine/math_engine``
as MCP tools, so a tool-calling LLM can reach them. The LangGraph nodes import
those functions directly instead — either path runs identical, deterministic
math. Keeping one implementation means there is no risk of the MCP tools and the
in-process calls drifting apart.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make the project root importable so we can reuse app.engine.math_engine.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from app.engine import math_engine as engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("math-engine-mcp")
mcp = FastMCP("Quantitative Math Engine")


# Each tool is a one-line delegation to the pure engine function. The
# docstrings are what the LLM reads when deciding which tool to call.
@mcp.tool()
def basic_calculator(operation: str, num1: float, num2: float) -> Dict[str, Any]:
    """Anti-hallucination calculator. operation ∈ add/subtract/multiply/divide/percent_change."""
    return engine.basic_calculator(operation, num1, num2)


@mcp.tool()
def calculate_position_size(
    available_cash_balance: float, stock_price: float, max_allocation_percent: float = 100.0
) -> Dict[str, Any]:
    """How many covered-call contracts the account can open (1 contract = 100 shares)."""
    return engine.calculate_position_size(available_cash_balance, stock_price, max_allocation_percent)


@mcp.tool()
def calculate_adjusted_cost_basis(
    original_stock_purchase_price: float,
    historical_premiums_collected: float,
    new_premium_offered: float = 0.0,
) -> Dict[str, Any]:
    """True breakeven of a position after all option premiums collected."""
    return engine.calculate_adjusted_cost_basis(
        original_stock_purchase_price, historical_premiums_collected, new_premium_offered
    )


@mcp.tool()
def calculate_technical_indicators(
    candles: List[Dict[str, Any]], trend_lookback_days: int = 20
) -> Dict[str, Any]:
    """RSI / SMA(20,50,200) / Bollinger Bands + heuristic trend classification."""
    return engine.calculate_technical_indicators(candles, trend_lookback_days)


@mcp.tool()
def calculate_historical_volatility(candles: List[Dict[str, Any]], window: int = 252) -> Dict[str, Any]:
    """Annualized realized volatility from daily closes (IV-Rank benchmark)."""
    return engine.calculate_historical_volatility(candles, window)


@mcp.tool()
def calculate_iv_rank(current_iv: float, iv_52wk_high: float, iv_52wk_low: float) -> Dict[str, Any]:
    """IV Rank (0–100): where current IV sits in its 52-week range."""
    return engine.calculate_iv_rank(current_iv, iv_52wk_high, iv_52wk_low)


@mcp.tool()
def black_scholes_call_greeks(
    spot: float,
    strike: float,
    days_to_expiration: int,
    volatility: float,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> Dict[str, Any]:
    """Black-Scholes Greeks (delta/gamma/theta/vega/rho) + probability of assignment for a call."""
    return engine.black_scholes_call_greeks(
        spot, strike, days_to_expiration, volatility, risk_free_rate, dividend_yield
    )


@mcp.tool()
def calculate_probability_of_profit(
    spot: float,
    breakeven_price: float,
    volatility: float,
    days_to_expiration: int,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> Dict[str, Any]:
    """Risk-neutral probability a covered call finishes above breakeven (profitable)."""
    return engine.calculate_probability_of_profit(
        spot, breakeven_price, volatility, days_to_expiration, risk_free_rate, dividend_yield
    )


@mcp.tool()
def calculate_expected_move(
    spot: float, volatility: float, days_to_expiration: int, num_std: float = 1.0
) -> Dict[str, Any]:
    """Expected ±num_std move of the underlying over the holding period (from IV)."""
    return engine.calculate_expected_move(spot, volatility, days_to_expiration, num_std)


@mcp.tool()
def calculate_moneyness(spot: float, strike: float) -> Dict[str, Any]:
    """Upside cushion (% OTM) to the strike before assignment."""
    return engine.calculate_moneyness(spot, strike)


@mcp.tool()
def calculate_dividend_yield(annual_dividend_per_share: float, stock_price: float) -> Dict[str, Any]:
    """Annual dividend yield % for the Scout's >2% dividend filter."""
    return engine.calculate_dividend_yield(annual_dividend_per_share, stock_price)


@mcp.tool()
def find_optimal_covered_call(
    option_chain: Dict[str, Any],
    target_delta: float = 0.35,
    min_dte: int = 30,
    max_dte: int = 45,
) -> Dict[str, Any]:
    """Return the single best covered-call strike (closest to target delta in the DTE window)."""
    return engine.find_optimal_covered_call(
        option_chain, target_delta=target_delta, min_dte=min_dte, max_dte=max_dte
    )


@mcp.tool()
def score_covered_call_candidate(
    annualized_yield_pct: float,
    iv_to_hv_ratio: float,
    sentiment_score: int,
    downside_buffer_pct: float,
    prob_keep_premium_pct: float,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """Composite 0–100 score blending yield/IV/sentiment/buffer/probability."""
    return engine.score_covered_call_candidate(
        annualized_yield_pct, iv_to_hv_ratio, sentiment_score,
        downside_buffer_pct, prob_keep_premium_pct, weights,
    )


@mcp.tool()
def grade_from_score(score: float, thresholds: Dict[str, float]) -> str:
    """Map a 0–100 composite score to an A/B/C/D letter grade."""
    return engine.grade_from_score(score, thresholds)


@mcp.tool()
def calculate_yield_metrics(
    current_stock_price: float, strike_price: float, premium_collected: float, days_to_expiration: int
) -> Dict[str, Any]:
    """Annualized Return on Capital (AROC) + breakevens."""
    return engine.calculate_yield_metrics(
        current_stock_price, strike_price, premium_collected, days_to_expiration
    )


@mcp.tool()
def calculate_premium_composition(
    current_stock_price: float, strike_price: float, option_premium: float
) -> Dict[str, Any]:
    """Split a call premium into intrinsic (equity) and extrinsic (time/vol) value."""
    return engine.calculate_premium_composition(current_stock_price, strike_price, option_premium)


@mcp.tool()
def calculate_liquidity_slippage(
    bid: float, ask: float, max_acceptable_spread_percent: float = 10.0
) -> Dict[str, Any]:
    """Bid-ask spread liquidity guard for option chains."""
    return engine.calculate_liquidity_slippage(bid, ask, max_acceptable_spread_percent)


@mcp.tool()
def is_earnings_within_cycle(earnings_date: str, expiration_date: str) -> Dict[str, Any]:
    """Earnings guardrail: disqualify if earnings falls on/before the option expiration."""
    return engine.is_earnings_within_cycle(earnings_date, expiration_date)


@mcp.tool()
def calculate_macro_divergence(
    asset_open: float, asset_close: float, benchmark_open: float, benchmark_close: float
) -> Dict[str, Any]:
    """Relative weakness vs a benchmark → macro-vs-micro loss classification."""
    return engine.calculate_macro_divergence(asset_open, asset_close, benchmark_open, benchmark_close)


@mcp.tool()
def generate_tot_defense_branches(
    entry_stock_price: float,
    current_stock_price: float,
    original_premium: float,
    current_call_ask: float,
    roll_down_premium: float = 0.0,
) -> Dict[str, Any]:
    """ToT generator: exact P&L for Branch A (Liquidate) / B (Roll Down) / C (Hold)."""
    return engine.generate_tot_defense_branches(
        entry_stock_price, current_stock_price, original_premium, current_call_ask, roll_down_premium
    )


if __name__ == "__main__":
    mcp.run()
