"""Deterministic quantitative math engine (pure functions only).

This module is the single source of truth for every number the system
produces. The LangGraph nodes import these functions directly; the FastMCP
wrapper in ``math_mcp/math_mcp.py`` re-exports the same functions as tools so a
tool-calling LLM can also reach them. Either way, the math is identical and
deterministic.

Design notes
------------
* ``pandas``/``numpy`` are imported LAZILY, inside the only two functions that
  need them. That keeps the arithmetic helpers (yields, spreads, P&L) usable —
  and testable — without the heavy data-science stack installed.
* We compute RSI / SMA / Bollinger Bands by hand rather than via ``pandas_ta``
  (an abandoned package incompatible with NumPy 2 / Python 3.13).
"""

from __future__ import annotations

import math
from datetime import date, datetime
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

# Standard normal CDF / PDF — stdlib only, no scipy needed.
_NORM = NormalDist()


def _norm_cdf(x: float) -> float:
    return _NORM.cdf(x)


def _norm_pdf(x: float) -> float:
    return _NORM.pdf(x)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — Basic arithmetic guard (anti-hallucination calculator)
# ══════════════════════════════════════════════════════════════════════
def basic_calculator(operation: str, num1: float, num2: float) -> Dict[str, Any]:
    """Foundational calculator to prevent LLM arithmetic hallucinations.

    operation ∈ {add, subtract, multiply, divide, percent_change}
    For ``percent_change`` num1 is the OLD value and num2 the NEW value.
    """
    try:
        if operation == "add":
            result = num1 + num2
        elif operation == "subtract":
            result = num1 - num2
        elif operation == "multiply":
            result = num1 * num2
        elif operation == "divide":
            if num2 == 0:
                return {"error": "Division by zero is not allowed."}
            result = num1 / num2
        elif operation == "percent_change":
            if num1 == 0:
                return {"error": "Cannot calculate percent change from zero."}
            result = ((num2 - num1) / num1) * 100
        else:
            return {"error": f"Unknown operation: {operation}"}
        return {"operation": operation, "num1": num1, "num2": num2, "result": round(result, 4)}
    except Exception as e:  # pragma: no cover - defensive
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — Position sizing & cost basis
# ══════════════════════════════════════════════════════════════════════
def calculate_position_size(
    available_cash_balance: float,
    stock_price: float,
    max_allocation_percent: float = 100.0,
) -> Dict[str, Any]:
    """How many covered-call contracts the account can open (1 contract = 100 shares)."""
    usable_cash = available_cash_balance * (max_allocation_percent / 100.0)
    cost_per_contract = stock_price * 100

    if cost_per_contract > usable_cash:
        return {
            "can_afford_trade": False,
            "max_contracts": 0,
            "reason": f"1 contract costs ${cost_per_contract:.2f}, but only ${usable_cash:.2f} is allocated.",
        }

    max_contracts = int(usable_cash // cost_per_contract)
    total_capital_required = max_contracts * cost_per_contract
    cash_remaining = available_cash_balance - total_capital_required
    return {
        "can_afford_trade": True,
        "max_contracts": max_contracts,
        "total_shares_to_buy": max_contracts * 100,
        "total_capital_required": round(total_capital_required, 2),
        "cash_remaining_after_trade": round(cash_remaining, 2),
    }


def calculate_adjusted_cost_basis(
    original_stock_purchase_price: float,
    historical_premiums_collected: float,
    new_premium_offered: float = 0.0,
) -> Dict[str, Any]:
    """True breakeven of a position after all option premiums collected."""
    total_premiums = historical_premiums_collected + new_premium_offered
    adjusted_basis = original_stock_purchase_price - total_premiums
    return {
        "original_cost_basis": original_stock_purchase_price,
        "total_premiums_harvested": round(total_premiums, 2),
        "new_adjusted_cost_basis": round(adjusted_basis, 2),
        "downside_protection_dollars": round(total_premiums, 2),
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — Technical indicators & trend (needs pandas/numpy, lazy)
# ══════════════════════════════════════════════════════════════════════
def _closes_to_series(candles: List[Dict[str, Any]]):
    """Convert Schwab candle dicts → a numeric pandas Series of closes."""
    import pandas as pd  # lazy import

    df = pd.DataFrame(candles)
    if "close" not in df.columns:
        raise ValueError("Candle data must contain a 'close' field.")
    return pd.to_numeric(df["close"])


def _sma(series, length: int):
    return series.rolling(window=length).mean()


def _rsi(series, length: int = 14):
    """Wilder's RSI (the standard smoothing used by most charting platforms)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/length, no bias correction.
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _bollinger(series, length: int = 20, num_std: float = 2.0):
    mid = series.rolling(window=length).mean()
    std = series.rolling(window=length).std(ddof=0)  # population std, matches convention
    return mid + num_std * std, mid, mid - num_std * std


def calculate_technical_indicators(
    candles: List[Dict[str, Any]], trend_lookback_days: int = 20
) -> Dict[str, Any]:
    """RSI, SMA(20/50, plus 200 if available), Bollinger Bands + trend label.

    Trend detection: slope of the 50-SMA over the lookback window classifies
    Upward / Sideways / Downward — serving the design's "detect sideways or
    upward trends" requirement.

    Adaptive history: the binding indicator is the 50-SMA, so we only need
    ``50 + trend_lookback_days`` candles (≈70 for the default 20-day lookback).
    The 200-SMA is a long-term-investor lens that's unnecessary for a 30–45 day
    covered-call horizon, so it is computed ONLY when ≥200 candles are supplied
    (otherwise reported as ``None``). This lets us fetch ~90 days instead of a
    full year and avoid hammering the data API.
    """
    required = 50 + trend_lookback_days
    n = len(candles) if candles else 0
    if n < required:
        return {"error": f"Insufficient candle data. Need >= {required} closes for "
                         f"the 50-SMA + {trend_lookback_days}-day trend window (got {n})."}

    import pandas as pd  # lazy import

    close = _closes_to_series(candles)
    df = pd.DataFrame({"close": close})
    df["SMA_20"] = _sma(close, 20)
    df["SMA_50"] = _sma(close, 50)
    df["RSI_14"] = _rsi(close, 14)
    df["BB_UPPER"], df["BB_MID"], df["BB_LOWER"] = _bollinger(close, 20, 2.0)

    has_200 = n >= 200
    if has_200:
        df["SMA_200"] = _sma(close, 200)

    # Drop only on the columns we strictly require (NOT SMA_200, which would
    # wipe every row when fewer than 200 candles are present).
    required_cols = ["SMA_50", "RSI_14", "BB_UPPER", "BB_MID", "BB_LOWER"]
    df = df.dropna(subset=required_cols)
    if len(df) < trend_lookback_days:
        return {"error": "Not enough data remaining after indicator ramp-up."}

    window = df.tail(trend_lookback_days)
    sma50_start = float(window.iloc[0]["SMA_50"])
    sma50_end = float(window.iloc[-1]["SMA_50"])
    sma50_pct_change = ((sma50_end - sma50_start) / sma50_start) * 100

    bb_width_start = (window.iloc[0]["BB_UPPER"] - window.iloc[0]["BB_LOWER"]) / window.iloc[0]["BB_MID"]
    bb_width_end = (window.iloc[-1]["BB_UPPER"] - window.iloc[-1]["BB_LOWER"]) / window.iloc[-1]["BB_MID"]
    is_bb_squeezing = bool(bb_width_end < bb_width_start)

    if abs(sma50_pct_change) < 1.5:
        trend = "Sideways (Consolidating)"
    elif sma50_pct_change >= 1.5:
        trend = "Upward (Bullish)"
    else:
        trend = "Downward (Bearish)"

    latest = df.iloc[-1]
    sma_200_val = (
        round(float(latest["SMA_200"]), 2)
        if has_200 and not pd.isna(latest.get("SMA_200"))
        else None
    )
    return {
        "current_snapshot": {
            "price": float(latest["close"]),
            "RSI_14": round(float(latest["RSI_14"]), 2),
            "SMA_20": round(float(latest["SMA_20"]), 2),
            "SMA_50": round(float(latest["SMA_50"]), 2),
            "SMA_200": sma_200_val,  # None when < 200 candles supplied
            "Bollinger_Upper": round(float(latest["BB_UPPER"]), 2),
            "Bollinger_Lower": round(float(latest["BB_LOWER"]), 2),
        },
        "trend_analysis": {
            "lookback_period_days": trend_lookback_days,
            "candles_used": int(n),
            "sma_50_slope_percent": round(sma50_pct_change, 2),
            "is_volatility_contracting": is_bb_squeezing,
            "detected_trend": trend,
        },
    }


def calculate_historical_volatility(
    candles: List[Dict[str, Any]], window: int = 252
) -> Dict[str, Any]:
    """Annualized historical (realized) volatility from daily closes.

    Used as the "historical average" benchmark for the IV-Rank guardrail when a
    dedicated 52-week IV history is unavailable.
    """
    import numpy as np  # lazy import

    close = _closes_to_series(candles).to_numpy(dtype=float)
    if len(close) < 30:
        return {"error": "Need at least ~30 closes to estimate volatility."}
    sample = close[-(window + 1):] if len(close) > window + 1 else close
    log_returns = np.diff(np.log(sample))
    daily_std = float(np.std(log_returns, ddof=1))
    annualized = daily_std * math.sqrt(252)
    return {
        "samples_used": int(len(log_returns)),
        "daily_volatility": round(daily_std, 5),
        "annualized_hv_percent": round(annualized * 100, 2),
    }


def calculate_iv_rank(
    current_iv: float, iv_52wk_high: float, iv_52wk_low: float
) -> Dict[str, Any]:
    """IV Rank = where current IV sits in its 52-week high/low range (0–100).

    Design §2 wants IV Rank above its historical average (>50 signals premiums
    are rich and advantageous for a seller).
    """
    if iv_52wk_high <= iv_52wk_low:
        return {"error": "52-week IV high must exceed the low."}
    iv_rank = (current_iv - iv_52wk_low) / (iv_52wk_high - iv_52wk_low) * 100
    iv_rank = max(0.0, min(100.0, iv_rank))  # clamp into [0, 100]
    return {
        "current_iv": round(current_iv, 4),
        "iv_52wk_high": round(iv_52wk_high, 4),
        "iv_52wk_low": round(iv_52wk_low, 4),
        "iv_rank": round(iv_rank, 2),
        "premiums_are_rich": iv_rank >= 50.0,
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3b — Option pricing, Greeks & probability (Black-Scholes)
# ══════════════════════════════════════════════════════════════════════
def black_scholes_call_greeks(
    spot: float,
    strike: float,
    days_to_expiration: int,
    volatility: float,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> Dict[str, Any]:
    """Black-Scholes Greeks + probabilities for a European CALL.

    Schwab usually supplies delta directly, but computing the full Greek set
    ourselves (a) fulfils the mission's "compute option Greeks" mandate, (b)
    gives us a deterministic cross-check on the broker's numbers, and (c) yields
    the **probability of assignment** N(d2), which delta only approximates.

    :param volatility: annualized IV as a decimal (e.g. 0.30 for 30%).
    :param risk_free_rate: annualized risk-free rate as a decimal.
    :param dividend_yield: annualized continuous dividend yield as a decimal.
    """
    if spot <= 0 or strike <= 0 or volatility <= 0 or days_to_expiration <= 0:
        return {"error": "spot, strike, volatility and days_to_expiration must all be positive."}

    T = days_to_expiration / 365.0
    sig_rt = volatility * math.sqrt(T)
    d1 = (math.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * T) / sig_rt
    d2 = d1 - sig_rt

    disc_q = math.exp(-dividend_yield * T)
    disc_r = math.exp(-risk_free_rate * T)

    delta = disc_q * _norm_cdf(d1)
    gamma = disc_q * _norm_pdf(d1) / (spot * sig_rt)
    vega = spot * disc_q * _norm_pdf(d1) * math.sqrt(T)           # per 1.00 vol
    theta_annual = (
        -(spot * disc_q * _norm_pdf(d1) * volatility) / (2 * math.sqrt(T))
        - risk_free_rate * strike * disc_r * _norm_cdf(d2)
        + dividend_yield * spot * disc_q * _norm_cdf(d1)
    )
    rho = strike * T * disc_r * _norm_cdf(d2)                     # per 1.00 rate

    prob_itm = _norm_cdf(d2)          # risk-neutral P(S_T > strike) → assigned
    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta_per_day": round(theta_annual / 365.0, 4),         # decay per calendar day
        "vega_per_1pct_vol": round(vega / 100.0, 4),
        "rho_per_1pct_rate": round(rho / 100.0, 4),
        "prob_assignment_percent": round(prob_itm * 100, 2),     # P(call finishes ITM)
        "prob_expire_otm_percent": round((1 - prob_itm) * 100, 2),  # keep premium + shares
    }


def calculate_probability_of_profit(
    spot: float,
    breakeven_price: float,
    volatility: float,
    days_to_expiration: int,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> Dict[str, Any]:
    """Risk-neutral probability a covered call is profitable at expiration.

    Profit occurs when the underlying finishes above the breakeven
    (cost basis − premium). Uses the lognormal P(S_T > breakeven).
    """
    if spot <= 0 or breakeven_price <= 0 or volatility <= 0 or days_to_expiration <= 0:
        return {"error": "All inputs must be positive."}
    T = days_to_expiration / 365.0
    d2 = (
        math.log(spot / breakeven_price)
        + (risk_free_rate - dividend_yield - 0.5 * volatility ** 2) * T
    ) / (volatility * math.sqrt(T))
    pop = _norm_cdf(d2)
    return {
        "breakeven_price": round(breakeven_price, 2),
        "prob_of_profit_percent": round(pop * 100, 2),
    }


def calculate_expected_move(
    spot: float, volatility: float, days_to_expiration: int, num_std: float = 1.0
) -> Dict[str, Any]:
    """Expected (±num_std) move of the underlying over the holding period.

    A good covered-call strike usually sits at or beyond the +1σ expected move,
    so the stock is statistically unlikely to be called away.
    """
    if spot <= 0 or volatility <= 0 or days_to_expiration <= 0:
        return {"error": "spot, volatility and days_to_expiration must be positive."}
    T = days_to_expiration / 365.0
    move = spot * volatility * math.sqrt(T) * num_std
    return {
        "num_std": num_std,
        "expected_move_dollars": round(move, 2),
        "expected_move_percent": round((move / spot) * 100, 2),
        "upper_expected_price": round(spot + move, 2),
        "lower_expected_price": round(spot - move, 2),
    }


def calculate_moneyness(spot: float, strike: float) -> Dict[str, Any]:
    """Upside room to the strike (distinct from the premium downside buffer).

    Positive cushion = out-of-the-money call (room for the stock to rise before
    assignment).
    """
    if spot <= 0:
        return {"error": "spot must be positive."}
    cushion_pct = (strike - spot) / spot * 100
    return {
        "spot": spot,
        "strike": strike,
        "otm_cushion_percent": round(cushion_pct, 2),
        "is_otm": strike > spot,
    }


def calculate_dividend_yield(
    annual_dividend_per_share: float, stock_price: float
) -> Dict[str, Any]:
    """Annual dividend yield % — feeds the Scout's >2% dividend filter."""
    if stock_price <= 0:
        return {"error": "stock_price must be positive."}
    yield_pct = annual_dividend_per_share / stock_price * 100
    return {
        "annual_dividend_per_share": annual_dividend_per_share,
        "stock_price": stock_price,
        "dividend_yield_percent": round(yield_pct, 2),
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — Option-chain selection
# ══════════════════════════════════════════════════════════════════════
def find_optimal_covered_call(
    option_chain: Dict[str, Any],
    target_delta: float = 0.35,
    delta_band: Tuple[float, float] = (0.30, 0.40),
    min_dte: int = 30,
    max_dte: int = 45,
) -> Dict[str, Any]:
    """Parse a Schwab option chain and return the single best covered-call strike.

    Selection: among CALL contracts whose days-to-expiration fall in
    [min_dte, max_dte] AND whose |delta| falls in ``delta_band``, pick the one
    closest to ``target_delta``. Returns only one contract to protect the LLM
    context window. If nothing matches the band, returns the closest-to-target
    contract within the DTE window and flags ``in_delta_band=False``.
    """
    call_map = option_chain.get("callExpDateMap", {})
    if not call_map:
        return {"error": "No call option chain data found."}

    lo, hi = delta_band
    best_in_band: Optional[Dict[str, Any]] = None
    best_in_band_diff = float("inf")
    best_overall: Optional[Dict[str, Any]] = None
    best_overall_diff = float("inf")

    for exp_date_key, strikes in call_map.items():
        # Schwab keys look like "2026-07-17:32" → the part after ':' is DTE.
        try:
            days_to_expiration = int(exp_date_key.split(":")[1])
        except (IndexError, ValueError):
            continue
        if not (min_dte <= days_to_expiration <= max_dte):
            continue

        for strike, contracts in strikes.items():
            if not contracts:
                continue
            contract = contracts[0]
            delta = contract.get("delta", None)
            if delta in (None, "NaN") or (isinstance(delta, float) and math.isnan(delta)):
                continue
            try:
                delta_val = abs(float(delta))
            except (TypeError, ValueError):
                continue

            candidate = {
                "symbol": contract.get("symbol"),
                "strike": float(strike),
                "expiration_key": exp_date_key,
                "days_to_expiration": days_to_expiration,
                "delta": round(delta_val, 4),
                "bid": float(contract.get("bid", 0.0) or 0.0),
                "ask": float(contract.get("ask", 0.0) or 0.0),
                "mark": float(contract.get("mark", 0.0) or 0.0),
                "volume": int(contract.get("totalVolume", 0) or 0),
                "open_interest": int(contract.get("openInterest", 0) or 0),
                # Schwab reports implied volatility as a percent (e.g. 22.5).
                "volatility": float(contract.get("volatility", 0.0) or 0.0),
            }
            diff = abs(delta_val - target_delta)

            if diff < best_overall_diff:
                best_overall_diff = diff
                best_overall = candidate
            if lo <= delta_val <= hi and diff < best_in_band_diff:
                best_in_band_diff = diff
                best_in_band = candidate

    if best_in_band:
        best_in_band["in_delta_band"] = True
        return best_in_band
    if best_overall:
        best_overall["in_delta_band"] = False
        best_overall["warning"] = f"No contract within delta band {delta_band}; returning closest match."
        return best_overall
    return {"error": "Could not find any valid contract within the DTE window."}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_covered_call_candidate(
    annualized_yield_pct: float,
    iv_to_hv_ratio: float,
    sentiment_score: int,
    downside_buffer_pct: float,
    prob_keep_premium_pct: float,
    weights: Dict[str, float],
    *,
    yield_cap: float = 30.0,
    iv_ratio_cap: float = 2.5,
    buffer_cap: float = 10.0,
) -> Dict[str, Any]:
    """Blend the candidate's signals into a single 0–100 composite score.

    Each input is normalized to 0..1 against a sensible cap, then combined using
    the (normalized) weights. Deterministic and pure — the Risk Manager calls
    this; the §5 feedback loop tunes ``weights``.
    """
    comps = {
        "yield": _clamp01(annualized_yield_pct / yield_cap),
        "iv": _clamp01((iv_to_hv_ratio - 1.0) / (iv_ratio_cap - 1.0)),
        "sentiment": _clamp01((sentiment_score - 1) / 4.0),   # 1..5 → 0..1
        "buffer": _clamp01(downside_buffer_pct / buffer_cap),
        "prob": _clamp01(prob_keep_premium_pct / 100.0),
    }
    total_w = sum(weights.get(k, 0.0) for k in comps) or 1.0
    score = sum(weights.get(k, 0.0) * v for k, v in comps.items()) / total_w * 100.0
    return {"score": round(score, 2), "components": {k: round(v, 3) for k, v in comps.items()}}


def grade_from_score(score: float, thresholds: Dict[str, float]) -> str:
    """Map a 0–100 score to a letter grade using A/B/C cutoffs (else 'D')."""
    if score >= thresholds.get("A", 75.0):
        return "A"
    if score >= thresholds.get("B", 60.0):
        return "B"
    if score >= thresholds.get("C", 45.0):
        return "C"
    return "D"


# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 — Yield, premium composition, liquidity
# ══════════════════════════════════════════════════════════════════════
def calculate_yield_metrics(
    current_stock_price: float,
    strike_price: float,
    premium_collected: float,
    days_to_expiration: int,
) -> Dict[str, Any]:
    """Annualized Return on Capital (AROC) + breakevens for the entry screener."""
    downside_breakeven = current_stock_price - premium_collected
    max_profit = (strike_price - current_stock_price) + premium_collected

    dte = max(days_to_expiration, 1)
    return_if_assigned = max_profit / current_stock_price
    aroc_assigned = return_if_assigned * (365 / dte)
    return_if_flat = premium_collected / current_stock_price
    aroc_flat = return_if_flat * (365 / dte)

    return {
        "downside_breakeven_price": round(downside_breakeven, 2),
        "downside_buffer_percent": round((premium_collected / current_stock_price) * 100, 2),
        "max_profit_dollars": round(max_profit * 100, 2),  # per 100 shares
        "aroc_if_assigned_percent": round(aroc_assigned * 100, 2),
        "aroc_if_flat_percent": round(aroc_flat * 100, 2),
    }


def calculate_premium_composition(
    current_stock_price: float, strike_price: float, option_premium: float
) -> Dict[str, Any]:
    """Split a call premium into intrinsic (equity) and extrinsic (time/vol) value."""
    intrinsic_value = max(0.0, current_stock_price - strike_price)
    extrinsic_value = max(0.0, option_premium - intrinsic_value)
    return {
        "current_stock_price": current_stock_price,
        "strike_price": strike_price,
        "total_premium": option_premium,
        "intrinsic_value": round(intrinsic_value, 2),
        "extrinsic_value": round(extrinsic_value, 2),
        "is_itm": current_stock_price > strike_price,
        "warning": "Rolling to this strike cannibalizes equity!"
        if intrinsic_value > 0 and extrinsic_value < 0.10
        else "Valid extrinsic harvest.",
    }


def calculate_liquidity_slippage(
    bid: float, ask: float, max_acceptable_spread_percent: float = 10.0
) -> Dict[str, Any]:
    """Bid-ask spread guard — the design's 'Liquidity Guard' for option chains."""
    if ask == 0:
        return {"error": "Ask price is zero. Invalid option quote."}
    spread_dollars = ask - bid
    spread_percent = (spread_dollars / ask) * 100
    return {
        "bid": bid,
        "ask": ask,
        "spread_dollars": round(spread_dollars, 2),
        "spread_percent": round(spread_percent, 2),
        "is_tradable": spread_percent <= max_acceptable_spread_percent,
        "rejection_reason": (
            f"Spread is {round(spread_percent, 2)}%, exceeding the "
            f"{max_acceptable_spread_percent}% limit."
            if spread_percent > max_acceptable_spread_percent
            else "None"
        ),
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 6 — Earnings guardrail (pure date math)
# ══════════════════════════════════════════════════════════════════════
def _coerce_date(value: Any) -> Optional[date]:
    """Accept a date, datetime, or 'YYYY-MM-DD' string → date (or None)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"):
            try:
                return datetime.strptime(value[: len(fmt) + 2], fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def is_earnings_within_cycle(
    earnings_date: Any, expiration_date: Any
) -> Dict[str, Any]:
    """Design's hard 'Earnings Guardrail': never sell a call that expires AFTER
    (or on) an upcoming earnings report. Returns disqualify=True if unsafe.
    """
    ed = _coerce_date(earnings_date)
    xd = _coerce_date(expiration_date)
    if ed is None:
        # No known earnings date → cannot prove safety, but don't hard-block.
        return {"disqualify": False, "reason": "No earnings date available; treat as unknown.", "earnings_known": False}
    if xd is None:
        return {"disqualify": True, "reason": "Invalid expiration date.", "earnings_known": True}

    unsafe = ed <= xd
    return {
        "earnings_known": True,
        "earnings_date": ed.isoformat(),
        "expiration_date": xd.isoformat(),
        "days_between": (xd - ed).days,
        "disqualify": unsafe,
        "reason": (
            f"Earnings on {ed.isoformat()} falls on/before expiration {xd.isoformat()} — earnings gap risk."
            if unsafe
            else f"Earnings on {ed.isoformat()} is after expiration {xd.isoformat()} — safe."
        ),
    }


# ══════════════════════════════════════════════════════════════════════
#  SECTION 7 — Downside management (Tree of Thoughts P&L generator)
# ══════════════════════════════════════════════════════════════════════
def calculate_macro_divergence(
    asset_open: float, asset_close: float, benchmark_open: float, benchmark_close: float
) -> Dict[str, Any]:
    """Relative weakness vs a benchmark (e.g. SPY) → macro-vs-micro loss lean."""
    if asset_open == 0 or benchmark_open == 0:
        return {"error": "Opening prices cannot be zero."}
    asset_pct = ((asset_close - asset_open) / asset_open) * 100
    bench_pct = ((benchmark_close - benchmark_open) / benchmark_open) * 100
    divergence = asset_pct - bench_pct

    if asset_pct < -3.0 and bench_pct < -2.0 and abs(divergence) < 1.5:
        lean = "Macro Sector Drag"
    elif asset_pct < -5.0 and bench_pct > -1.0:
        lean = "Micro Company Failure"
    else:
        lean = "Mixed / Uncorrelated"

    return {
        "asset_percent_change": round(asset_pct, 2),
        "benchmark_percent_change": round(bench_pct, 2),
        "divergence_spread": round(divergence, 2),
        "heuristic_classification": lean,
    }


def generate_tot_defense_branches(
    entry_stock_price: float,
    current_stock_price: float,
    original_premium: float,
    current_call_ask: float,
    roll_down_premium: float = 0.0,
) -> Dict[str, Any]:
    """ToT thought generator: exact P&L for the 3 downside-defense branches.

    Branch A (Hard Eject) | Branch B (Roll Down) | Branch C (Hold & Wait).
    """
    shares = 100
    stock_loss = (current_stock_price - entry_stock_price) * shares
    call_pnl = (original_premium - current_call_ask) * shares

    branch_a_net_pnl = stock_loss + call_pnl
    roll_net_credit = (roll_down_premium - current_call_ask) * shares
    branch_c_unrealized_net = stock_loss + call_pnl

    return {
        "Branch_A_Liquidate": {
            "action": "Buy-to-close the call and sell the 100 shares.",
            "realized_cash_loss": round(branch_a_net_pnl, 2),
            "capital_freed_up": round(current_stock_price * shares - (current_call_ask * shares), 2),
        },
        "Branch_B_Roll_Down": {
            "action": "Buy-to-close current call, sell a new lower-strike call.",
            "net_credit_received": round(roll_net_credit, 2),
            "unrealized_stock_loss": round(stock_loss, 2),
            "is_valid": roll_net_credit > 0,  # invalid if it requires a net debit
        },
        "Branch_C_Hold": {
            "action": "Do nothing. Wait for recovery.",
            "unrealized_net_pnl": round(branch_c_unrealized_net, 2),
        },
    }
