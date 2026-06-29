"""Node 2 — The Quant Agent (Greeks & Options Analytics).

Job (design §2): evaluate the mathematical feasibility of each Scout survivor
and emit a "mathematical shortlist of optimized option strikes." This node does
NO LLM work — it orchestrates the deterministic engine and applies hard,
configurable filters. Every dropped candidate is recorded with a precise reason.

The per-candidate pipeline is decomposed into small, named stage helpers (each
can reject by raising ``_Reject``); ``_analyze_candidate`` chains them, and
``quant_node`` just loops + logs:
  1. _ensure_affordable   — can the account buy 100 shares?
  2. _load_trend          — daily candles → indicators; reject downtrends
  3. _select_contract     — best OTM call in the delta + DTE band
  4. _ensure_liquid       — bid-ask spread guard + OI/volume minimums
  5. _assess_iv           — Black-Scholes greeks + IV-richness vs realized vol
  6. yield (AROC)         — stored for the Risk Manager to grade
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import rules as default_rules
from app.config import settings
from app.engine import math_engine as eng
from app.data.schwab_client import SchwabClient
from app.state import QuantCandidate, ScreenerState, reject

logger = logging.getLogger("node.quant")


class _Reject(Exception):
    """Internal control-flow signal: a candidate failed a Quant stage.

    ``reason`` is the audit-trail message; ``error`` (optional) is a separate
    run-level error string for API/exception failures (vs. plain filter misses).
    """

    def __init__(self, reason: str, error: Optional[str] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.error = error


# ── per-stage helpers (module-level, pure orchestration over the engine) ──
def _underlying_price(cand: Dict[str, Any]) -> float:
    return float(cand.get("fundamentals", {}).get("last_price", 0.0) or 0.0)


def _premium_per_share(contract: Dict[str, Any]) -> float:
    """Use the mark if present, else the bid-ask midpoint."""
    mark = contract.get("mark", 0.0)
    if mark and mark > 0:
        return float(mark)
    bid, ask = contract.get("bid", 0.0), contract.get("ask", 0.0)
    return float((bid + ask) / 2.0) if ask else 0.0


def _ensure_affordable(account_cash: float, price: float, rules) -> None:
    sizing = eng.calculate_position_size(account_cash, price, rules.max_allocation_per_trade_pct)
    if not sizing.get("can_afford_trade"):
        raise _Reject(f"Unaffordable: 100 shares cost ${price * 100:,.0f}, cash ${account_cash:,.0f}.")


def _load_trend(client: SchwabClient, sym: str, rules) -> Tuple[Dict[str, Any], list]:
    """Fetch daily candles, compute indicators, and reject downtrends.

    Returns (indicators, candles). Raises _Reject on fetch error, insufficient
    data, or (when configured) a downward trend.
    """
    try:
        hist = client.get_price_history(
            sym, period_type="month", period=rules.price_history_months,
            frequency_type="daily", frequency=1)
    except Exception as exc:  # noqa: BLE001
        raise _Reject(f"Price history error: {exc}", error=f"Quant price-history failed for {sym}: {exc}")
    candles = hist.get("candles", []) or []
    indicators = eng.calculate_technical_indicators(candles, rules.trend_lookback_days)
    if "error" in indicators:
        raise _Reject(f"Indicators: {indicators['error']}")
    trend = indicators["trend_analysis"]["detected_trend"]
    if rules.reject_downtrend and trend.startswith("Downward"):
        raise _Reject(f"Downtrend ({trend}); covered calls prefer flat/up.")
    return indicators, candles


def _select_contract(client: SchwabClient, sym: str, from_date: str, to_date: str, rules) -> Dict[str, Any]:
    """Pick the best OTM call within the delta + DTE band. Raises _Reject if none."""
    try:
        chain = client.get_option_chain(
            sym, contract_type="CALL", range_filter="OTM", from_date=from_date, to_date=to_date)
    except Exception as exc:  # noqa: BLE001
        raise _Reject(f"Option chain error: {exc}", error=f"Quant option-chain failed for {sym}: {exc}")
    contract = eng.find_optimal_covered_call(
        chain, target_delta=rules.target_delta, delta_band=rules.delta_band,
        min_dte=rules.min_days_to_expiration, max_dte=rules.max_days_to_expiration)
    if "error" in contract:
        raise _Reject(f"No contract: {contract['error']}")
    if not contract.get("in_delta_band", False):
        raise _Reject(f"Best call delta {contract['delta']} outside band {rules.delta_band}.")
    return contract


def _ensure_liquid(contract: Dict[str, Any], rules) -> Dict[str, Any]:
    """Bid-ask spread + open-interest + volume guards. Returns the liquidity dict."""
    liq = eng.calculate_liquidity_slippage(contract["bid"], contract["ask"], rules.max_bid_ask_spread_pct)
    if not liq.get("is_tradable"):
        raise _Reject(f"Illiquid option: {liq.get('rejection_reason')}")
    if contract["open_interest"] < rules.min_option_open_interest:
        raise _Reject(f"Open interest {contract['open_interest']} < {rules.min_option_open_interest}.")
    if contract["volume"] < rules.min_option_volume:
        raise _Reject(f"Option volume {contract['volume']} < {rules.min_option_volume}.")
    return liq


def _assess_iv(
    contract: Dict[str, Any], candles: list, price: float, rules
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """Compute Greeks, the IV-richness check (vs realized vol), and expected move.

    Returns (greeks, iv_info, expected_move). Raises _Reject if IV is not rich
    (when required).
    """
    dte = contract["days_to_expiration"]
    iv_pct = contract.get("volatility", 0.0)

    greeks: Dict[str, Any] = {}
    if iv_pct > 0:
        greeks = eng.black_scholes_call_greeks(price, contract["strike"], dte, iv_pct / 100.0)

    hv = eng.calculate_historical_volatility(candles)
    iv_info = {"current_iv_percent": iv_pct, "realized_vol_percent": hv.get("annualized_hv_percent")}
    if rules.require_rich_iv:
        hv_pct = hv.get("annualized_hv_percent")
        if not iv_pct or not hv_pct:
            raise _Reject("Cannot assess IV richness (missing IV/HV).")
        ratio = iv_pct / hv_pct
        iv_info["iv_to_hv_ratio"] = round(ratio, 2)
        if ratio < rules.iv_richness_min_ratio:
            raise _Reject(f"IV not rich: IV {iv_pct:.1f}% / HV {hv_pct:.1f}% = {ratio:.2f} "
                          f"< {rules.iv_richness_min_ratio}.")

    exp_move = eng.calculate_expected_move(price, iv_pct / 100.0, dte) if iv_pct > 0 else None
    return greeks, iv_info, exp_move


def _analyze_candidate(
    client: SchwabClient, cand: Dict[str, Any], account_cash: float,
    from_date: str, to_date: str, rules,
) -> QuantCandidate:
    """Run one Scout survivor through the full Quant pipeline. Raises _Reject at
    the first failing stage; returns a fully-analyzed QuantCandidate otherwise."""
    sym = cand["symbol"]
    price = _underlying_price(cand)
    if price <= 0:
        raise _Reject("Missing underlying price.")

    _ensure_affordable(account_cash, price, rules)
    indicators, candles = _load_trend(client, sym, rules)
    contract = _select_contract(client, sym, from_date, to_date, rules)
    liq = _ensure_liquid(contract, rules)

    premium = _premium_per_share(contract)
    if premium <= 0:
        raise _Reject("No usable premium (bid/ask/mark all zero).")

    greeks, iv_info, exp_move = _assess_iv(contract, candles, price, rules)
    yield_m = eng.calculate_yield_metrics(price, contract["strike"], premium, contract["days_to_expiration"])

    return {
        "symbol": sym,
        "underlying_price": price,
        "snapshot": indicators["current_snapshot"],
        "trend": indicators["trend_analysis"],
        "contract": contract,
        "greeks": greeks,
        "yield_metrics": yield_m,
        "iv_rank": iv_info,
        "expected_move": exp_move,
        "liquidity": liq,
        "passes_quant": True,
    }


def _apply_candidate_cap(scout_candidates: list, rules) -> Tuple[list, list]:
    """Split the Scout list into (to_analyze, deferred_rejections) per the cap."""
    cap = getattr(rules, "max_quant_candidates", 0)
    if not (cap and cap > 0 and len(scout_candidates) > cap):
        return scout_candidates, []
    deferred = [
        reject(c["symbol"], "QUANT", f"Not analyzed: beyond max_quant_candidates={cap} this run.")
        for c in scout_candidates[cap:]
    ]
    logger.info("Quant: capping analysis to %d of %d candidates (rest deferred).",
                cap, len(scout_candidates))
    return scout_candidates[:cap], deferred


def build_quant_node(
    client: SchwabClient,
    rules=default_rules,
    today: Optional[date] = None,
) -> Callable[[ScreenerState], dict]:
    """Return a Quant node bound to a Schwab client + strategy rules.

    ``today`` is injectable for deterministic tests of the expiration window.
    """

    def quant_node(state: ScreenerState) -> dict:
        scout_candidates = state.get("scout_candidates") or []
        account_cash = float(state.get("account_cash", 0.0) or 0.0)
        run_today = today or date.today()
        from_date = (run_today + timedelta(days=rules.min_days_to_expiration)).isoformat()
        to_date = (run_today + timedelta(days=rules.max_days_to_expiration)).isoformat()

        to_analyze, rejections = _apply_candidate_cap(scout_candidates, rules)
        results: List[QuantCandidate] = []
        errors: List[str] = []

        total = len(to_analyze)
        est_min = (total * 2) / max(settings.schwab_rate_limit_calls, 1)
        logger.info("Quant analyzing %d candidates (cash=$%.0f, ~%d Schwab calls, est ~%.1f min).",
                    total, account_cash, total * 2, est_min)

        for idx, cand in enumerate(to_analyze, 1):
            sym = cand["symbol"]
            logger.info("Quant [%d/%d] %s ...", idx, total, sym)
            try:
                qc = _analyze_candidate(client, cand, account_cash, from_date, to_date, rules)
            except _Reject as r:
                if r.error:
                    errors.append(r.error)
                logger.info("Quant [%d/%d] REJECT %s — %s", idx, total, sym, r.reason)
                rejections.append(reject(sym, "QUANT", r.reason))
                continue
            logger.info("Quant PASS %s: strike %.1f Δ%.2f %dDTE AROC(assigned) %.1f%%",
                        sym, qc["contract"]["strike"], qc["contract"]["delta"],
                        qc["contract"]["days_to_expiration"], qc["yield_metrics"]["aroc_if_assigned_percent"])
            results.append(qc)

        logger.info("Quant produced %d/%d shortlist candidates", len(results), len(scout_candidates))
        out = {"quant_candidates": results, "rejected": rejections}
        if errors:
            out["errors"] = errors
        return out

    return quant_node
