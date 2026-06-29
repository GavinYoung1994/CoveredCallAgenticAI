"""Tool registry — the agent's (and MCP server's) capabilities, one source of truth.

Each ``Tool`` bundles a name, a human/LLM-readable description, a JSON-schema for
its arguments, and a handler. The registry spans four groups:
  * data management  — cash/holdings/learnings/decisions (SQL),
  * quantitative analysis — the full deterministic math engine (yields, Greeks,
    probabilities, scoring, defense branches, …),
  * live market data — Schwab quotes/fundamentals/history/chains, news, earnings,
  * composites — fetch live data AND run analysis in one call (technical read,
    best covered-call finder).

Live-data clients are constructed LAZILY (first call) so building the registry —
and importing this module — needs no network or credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import rules as _rules
from app.engine import math_engine as eng


def _sane_option_window(from_date: Optional[str], to_date: Optional[str]) -> Tuple[str, str]:
    """Coerce a requested option-expiration window into a sane FUTURE range.

    LLMs sometimes emit past dates (from training data), which Schwab rejects with
    a 400. Missing, unparseable, or past dates are replaced with the strategy's
    30–45-day-out window; a backwards range is fixed up.
    """
    today = date.today()

    def _parse(s: Optional[str]) -> Optional[date]:
        try:
            return date.fromisoformat(s) if s else None
        except (TypeError, ValueError):
            return None

    fd, td = _parse(from_date), _parse(to_date)
    if fd is None or fd < today:
        fd = today + timedelta(days=_rules.min_days_to_expiration)
    if td is None or td <= fd:
        td = today + timedelta(days=_rules.max_days_to_expiration)
    return fd.isoformat(), td.isoformat()


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]   # JSON-schema (type=object) for the arguments
    handler: Callable[..., Any]

    def run(self, **kwargs: Any) -> Any:
        return self.handler(**kwargs)


def _obj(properties: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


_NUM = {"type": "number"}
_INT = {"type": "integer"}
_STR = {"type": "string"}
_BOOL = {"type": "boolean"}


# ── lazy live-data clients (built on first use, cached) ────────────────
_CLIENTS: Dict[str, Any] = {}


def _schwab() -> Any:
    if "schwab" not in _CLIENTS:
        from app.data.schwab_client import SchwabClient
        _CLIENTS["schwab"] = SchwabClient()
    return _CLIENTS["schwab"]


def _news() -> Any:
    if "news" not in _CLIENTS:
        from app.data.news_client import NewsClient
        _CLIENTS["news"] = NewsClient()
    return _CLIENTS["news"]


def _earnings() -> Any:
    if "earnings" not in _CLIENTS:
        from app.data.earnings_client import EarningsClient
        from app.data.earnings_search import CompositeEarningsClient, EarningsSearchClient
        _CLIENTS["earnings"] = CompositeEarningsClient([EarningsClient(), EarningsSearchClient()])
    return _CLIENTS["earnings"]


# ── composite handlers (data + math) ──────────────────────────────────
def _analyze_covered_call(
    underlying_price: float, strike: float, premium: float, days_to_expiration: int,
    volatility: Optional[float] = None, sentiment_score: int = 3,
) -> Dict[str, Any]:
    ym = eng.calculate_yield_metrics(underlying_price, strike, premium, days_to_expiration)
    out: Dict[str, Any] = {"yield_metrics": ym, "moneyness": eng.calculate_moneyness(underlying_price, strike)}
    if volatility:
        vol = volatility / 100.0 if volatility > 1.5 else volatility
        out["greeks"] = eng.black_scholes_call_greeks(underlying_price, strike, days_to_expiration, vol)
        out["expected_move"] = eng.calculate_expected_move(underlying_price, vol, days_to_expiration)
        out["prob_of_profit"] = eng.calculate_probability_of_profit(
            underlying_price, ym["downside_breakeven_price"], vol, days_to_expiration)
        prob_keep = out["greeks"].get("prob_expire_otm_percent", 70.0)
        out["composite_score"] = eng.score_covered_call_candidate(
            ym["aroc_if_flat_percent"], 1.3, sentiment_score, ym["downside_buffer_percent"], prob_keep,
            {"yield": 0.35, "iv": 0.2, "sentiment": 0.2, "buffer": 0.15, "prob": 0.1})
    return out


def _technical_analysis(symbol: str, months: int = 6) -> Dict[str, Any]:
    hist = _schwab().get_price_history(symbol, period_type="month", period=months,
                                       frequency_type="daily", frequency=1)
    candles = hist.get("candles", []) or []
    return {
        "symbol": symbol.upper(),
        "indicators": eng.calculate_technical_indicators(candles, 20),
        "historical_volatility": eng.calculate_historical_volatility(candles),
    }


def _find_best_covered_call(symbol: str, target_delta: float = 0.35,
                            min_dte: int = 30, max_dte: int = 45) -> Dict[str, Any]:
    today = date.today()
    chain = _schwab().get_option_chain(
        symbol, contract_type="CALL", range_filter="OTM",
        from_date=(today + timedelta(days=min_dte)).isoformat(),
        to_date=(today + timedelta(days=max_dte)).isoformat())
    band = (max(0.05, target_delta - 0.05), target_delta + 0.05)
    return eng.find_optimal_covered_call(chain, target_delta=target_delta, delta_band=band,
                                         min_dte=min_dte, max_dte=max_dte)


def _get_option_chain(symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None) -> Dict[str, Any]:
    fd, td = _sane_option_window(from_date, to_date)
    return _schwab().get_option_chain(symbol, contract_type="CALL", range_filter="OTM",
                                      from_date=fd, to_date=td)


def _next_earnings(symbol: str) -> Dict[str, Any]:
    today = date.today()
    d = _earnings().get_next_earnings_date(symbol, today.isoformat(),
                                           (today + timedelta(days=90)).isoformat())
    return {"symbol": symbol.upper(), "next_earnings_date": d}


# ── workflow handlers (heavy: live data + LLM; run the full LangGraphs) ──
def _run_stock_screener() -> Dict[str, Any]:
    from app.graphs import run_entry_screener
    final = run_entry_screener()
    recs = final.get("recommendations", [])
    return {
        "run_id": final.get("run_id"),
        "recommendation_count": len(recs),
        "rejected_count": len(final.get("rejected", [])),
        "notified": final.get("notified"),
        "recommendations": [
            {"symbol": r.get("symbol"), "grade": r.get("grade"), "score": r.get("score"),
             "annualized_yield_percent": r.get("annualized_yield_percent")}
            for r in recs
        ],
    }


def _run_defense_scan() -> Dict[str, Any]:
    from app.graphs import run_defense_scan
    res = run_defense_scan()
    return {"scanned": res.get("scanned", 0), "breached": res.get("breached", [])}


def _run_performance_review(period: str = "weekly") -> Dict[str, Any]:
    from app.reporting import generate_report
    from app.llm import get_llm
    out = generate_report(llm=get_llm(), period_label=period)
    return {"stats": out["stats"], "narrative": out["narrative"]}


# ── registry builders ──────────────────────────────────────────────────
def _management_tools(service: Any) -> List[Tool]:
    return [
        Tool("get_cash", "Get the current account cash balance (USD).", _obj({}),
             lambda: service.get_cash()),
        Tool("set_cash", "Set the account cash balance.", _obj({"amount": _NUM}, ["amount"]),
             lambda amount: service.set_cash(amount)),
        Tool("list_holdings", "List positions, optionally filtered by status "
             "(OPEN/ASSIGNED/LIQUIDATED/EXPIRED).", _obj({"status": _STR}),
             lambda status=None: service.list_holdings(status)),
        Tool("get_position", "Get one position by id.", _obj({"position_id": _STR}, ["position_id"]),
             lambda position_id: service.get_position(position_id)),
        Tool("update_holding_status",
             "RECORD an executed/closed trade for a holding. Use this whenever the user says a "
             "position was executed, assigned, called away, sold, liquidated, or expired, or to "
             "close a position. Identify the holding by `symbol` OR `position_id`. status is one of "
             "ASSIGNED (called away — sale price defaults to the strike), LIQUIDATED (needs "
             "stock_sale_price), EXPIRED. Updates cash + realized P&L. Call once PER holding.",
             _obj({"status": _STR, "symbol": _STR, "position_id": _STR, "stock_sale_price": _NUM,
                   "call_buyback_price": _NUM, "contracts": _INT}, ["status"]),
             lambda **kw: service.update_holding_status(**kw)),
        Tool("recent_decisions", "List recent approve/deny decisions.", _obj({"n": _INT}),
             lambda n=10: service.recent_decisions(n)),
        Tool("search_learnings", "Semantic search of past trade lessons / performance reports.",
             _obj({"query": _STR, "n": _INT}, ["query"]),
             lambda query, n=5: service.search_learnings(query, n)),
        Tool("performance_report",
             "Portfolio performance: realized P&L, win rate, premium harvested, annualized return on cash.",
             _obj({}), lambda: _performance_report(service.db_path)),
    ]


def _performance_report(db_path: Any) -> Dict[str, Any]:
    from app.reporting import gather_performance
    return gather_performance(db_path)


def _analysis_tools() -> List[Tool]:
    return [
        Tool("analyze_covered_call",
             "One-shot quant read on a covered call: yield/AROC, breakevens, moneyness, and (if "
             "volatility given as a decimal like 0.25) Greeks, expected move, probability of profit, "
             "and a composite score.",
             _obj({"underlying_price": _NUM, "strike": _NUM, "premium": _NUM,
                   "days_to_expiration": _INT, "volatility": _NUM, "sentiment_score": _INT},
                  ["underlying_price", "strike", "premium", "days_to_expiration"]),
             _analyze_covered_call),
        Tool("calculate_yield_metrics", "Annualized return on capital (AROC) + breakevens.",
             _obj({"current_stock_price": _NUM, "strike_price": _NUM, "premium_collected": _NUM,
                   "days_to_expiration": _INT},
                  ["current_stock_price", "strike_price", "premium_collected", "days_to_expiration"]),
             lambda **kw: eng.calculate_yield_metrics(**kw)),
        Tool("black_scholes_greeks",
             "Black-Scholes Greeks + probability of assignment for a call. volatility is a decimal (0.25).",
             _obj({"spot": _NUM, "strike": _NUM, "days_to_expiration": _INT, "volatility": _NUM,
                   "risk_free_rate": _NUM, "dividend_yield": _NUM},
                  ["spot", "strike", "days_to_expiration", "volatility"]),
             lambda **kw: eng.black_scholes_call_greeks(**kw)),
        Tool("probability_of_profit", "Risk-neutral probability a covered call finishes above breakeven.",
             _obj({"spot": _NUM, "breakeven_price": _NUM, "volatility": _NUM, "days_to_expiration": _INT},
                  ["spot", "breakeven_price", "volatility", "days_to_expiration"]),
             lambda **kw: eng.calculate_probability_of_profit(**kw)),
        Tool("expected_move", "Expected ±move of the underlying over a horizon (from IV decimal).",
             _obj({"spot": _NUM, "volatility": _NUM, "days_to_expiration": _INT, "num_std": _NUM},
                  ["spot", "volatility", "days_to_expiration"]),
             lambda **kw: eng.calculate_expected_move(**kw)),
        Tool("moneyness", "Upside cushion (% OTM) of a strike vs spot.",
             _obj({"spot": _NUM, "strike": _NUM}, ["spot", "strike"]),
             lambda **kw: eng.calculate_moneyness(**kw)),
        Tool("dividend_yield", "Annual dividend yield %.",
             _obj({"annual_dividend_per_share": _NUM, "stock_price": _NUM},
                  ["annual_dividend_per_share", "stock_price"]),
             lambda **kw: eng.calculate_dividend_yield(**kw)),
        Tool("iv_rank", "IV Rank (0–100): where current IV sits in its 52-week range.",
             _obj({"current_iv": _NUM, "iv_52wk_high": _NUM, "iv_52wk_low": _NUM},
                  ["current_iv", "iv_52wk_high", "iv_52wk_low"]),
             lambda **kw: eng.calculate_iv_rank(**kw)),
        Tool("liquidity_slippage", "Bid-ask spread liquidity guard for an option.",
             _obj({"bid": _NUM, "ask": _NUM, "max_acceptable_spread_percent": _NUM}, ["bid", "ask"]),
             lambda **kw: eng.calculate_liquidity_slippage(**kw)),
        Tool("premium_composition", "Split a premium into intrinsic + extrinsic value.",
             _obj({"current_stock_price": _NUM, "strike_price": _NUM, "option_premium": _NUM},
                  ["current_stock_price", "strike_price", "option_premium"]),
             lambda **kw: eng.calculate_premium_composition(**kw)),
        Tool("position_size", "How many covered-call contracts the cash can open (1 = 100 shares).",
             _obj({"available_cash_balance": _NUM, "stock_price": _NUM, "max_allocation_percent": _NUM},
                  ["available_cash_balance", "stock_price"]),
             lambda **kw: eng.calculate_position_size(**kw)),
        Tool("adjusted_cost_basis", "True breakeven after premiums collected.",
             _obj({"original_stock_purchase_price": _NUM, "historical_premiums_collected": _NUM,
                   "new_premium_offered": _NUM},
                  ["original_stock_purchase_price", "historical_premiums_collected"]),
             lambda **kw: eng.calculate_adjusted_cost_basis(**kw)),
        Tool("is_earnings_within_cycle", "Earnings guardrail: does earnings fall on/before expiration?",
             _obj({"earnings_date": _STR, "expiration_date": _STR},
                  ["earnings_date", "expiration_date"]),
             lambda **kw: eng.is_earnings_within_cycle(**kw)),
        Tool("macro_divergence", "Relative weakness vs a benchmark (macro vs micro loss).",
             _obj({"asset_open": _NUM, "asset_close": _NUM, "benchmark_open": _NUM,
                   "benchmark_close": _NUM},
                  ["asset_open", "asset_close", "benchmark_open", "benchmark_close"]),
             lambda **kw: eng.calculate_macro_divergence(**kw)),
        Tool("defense_branches",
             "Exact P&L for the 3 downside-defense branches (Hard Eject / Roll Down / Hold).",
             _obj({"entry_stock_price": _NUM, "current_stock_price": _NUM, "original_premium": _NUM,
                   "current_call_ask": _NUM, "roll_down_premium": _NUM},
                  ["entry_stock_price", "current_stock_price", "original_premium", "current_call_ask"]),
             lambda **kw: eng.generate_tot_defense_branches(**kw)),
        Tool("basic_calculator", "Anti-hallucination calculator (add/subtract/multiply/divide/percent_change).",
             _obj({"operation": _STR, "num1": _NUM, "num2": _NUM}, ["operation", "num1", "num2"]),
             lambda **kw: eng.basic_calculator(**kw)),
    ]


def _data_tools() -> List[Tool]:
    return [
        Tool("get_quote", "Live Schwab quote for one symbol (price, fundamentals, reference).",
             _obj({"symbol": _STR}, ["symbol"]), lambda symbol: _schwab().get_quote(symbol)),
        Tool("get_fundamentals", "Normalized fundamentals (price, avg volume, dividend yield).",
             _obj({"symbol": _STR}, ["symbol"]),
             lambda symbol: _schwab().extract_fundamentals(_schwab().get_quotes([symbol]), symbol)),
        Tool("get_price_history", "Raw daily candles for a symbol (months back).",
             _obj({"symbol": _STR, "months": _INT}, ["symbol"]),
             lambda symbol, months=6: _schwab().get_price_history(
                 symbol, period_type="month", period=months, frequency_type="daily", frequency=1)),
        Tool("technical_analysis",
             "Fetch price history AND compute RSI/SMA/Bollinger + trend + realized volatility.",
             _obj({"symbol": _STR, "months": _INT}, ["symbol"]), _technical_analysis),
        Tool("get_option_chain",
             "Raw OTM call option chain for a symbol. Dates default to the 30–45-day-out window; "
             "past/invalid dates are auto-corrected (so it never 400s on a stale date).",
             _obj({"symbol": _STR, "from_date": _STR, "to_date": _STR}, ["symbol"]),
             _get_option_chain),
        Tool("find_best_covered_call",
             "Fetch the chain AND pick the best covered-call strike near a target delta in the 30–45 DTE band.",
             _obj({"symbol": _STR, "target_delta": _NUM, "min_dte": _INT, "max_dte": _INT}, ["symbol"]),
             _find_best_covered_call),
        Tool("get_option_expirations", "Listed option expirations for a symbol.",
             _obj({"symbol": _STR}, ["symbol"]), lambda symbol: _schwab().get_option_expirations(symbol)),
        Tool("is_optionable", "Whether a symbol has listed options.",
             _obj({"symbol": _STR}, ["symbol"]),
             lambda symbol: {"symbol": symbol.upper(), "is_optionable": _schwab().is_optionable(symbol)}),
        Tool("get_news", "Recent news headlines for a symbol (with provider sentiment).",
             _obj({"symbol": _STR, "limit": _INT}, ["symbol"]),
             lambda symbol, limit=8: {"headlines": _news().get_headlines(symbol, limit=limit, fetch_content=False)}),
        Tool("get_next_earnings_date",
             "Next earnings date for a symbol (Finnhub, then Google-search fallback).",
             _obj({"symbol": _STR}, ["symbol"]), _next_earnings),
    ]


def _workflow_tools(launcher: Optional[Callable[[str], Any]] = None) -> List[Tool]:
    """The three workflow triggers.

    If ``launcher`` is given (web context), each tool starts the workflow in the
    BACKGROUND via the shared job runner and returns immediately — so chat stays
    responsive and the user watches progress in the workflow window. Without a
    launcher (CLI), they run synchronously and return a result summary.
    """
    if launcher is not None:
        bg = ("Starts the {desc} in the BACKGROUND and returns immediately (non-blocking). "
              "After calling this, tell the user to open the '{win}' workflow window to watch "
              "live progress and logs.")
        return [
            Tool("run_stock_screener", bg.format(desc="entry-screener pipeline", win="Entry Screener"),
                 _obj({}), lambda: launcher("screener")),
            Tool("run_defense_scan", bg.format(desc="downside-defense scan", win="Downside Defense"),
                 _obj({}), lambda: launcher("defense")),
            Tool("run_performance_review", bg.format(desc="performance report", win="Performance Report"),
                 _obj({"period": _STR}), lambda period="weekly": launcher("report")),
        ]
    return [
        Tool("run_stock_screener",
             "Run the FULL entry-screener pipeline (Scout→Quant→News→Risk) over the watchlist and "
             "return graded covered-call recommendations. Long-running (live data + LLM; minutes).",
             _obj({}), _run_stock_screener),
        Tool("run_defense_scan",
             "Run the downside-defense (Tree-of-Thoughts) scan over ALL open holdings; returns which "
             "breached their threshold. Long-running (live data + LLM).",
             _obj({}), _run_defense_scan),
        Tool("run_performance_review",
             "Generate a performance report (P&L, win rate, annualized return) with an LLM narrative; "
             "stores the lesson. args: {period?: 'weekly'|'monthly'}.",
             _obj({"period": _STR}), _run_performance_review),
    ]


def build_tools(service: Any, memory: Any = None, *, include_data: bool = True,
                include_workflows: bool = True,
                workflow_launcher: Optional[Callable[[str], Any]] = None) -> Dict[str, Tool]:
    """Build the full tool registry bound to a ManagementService (+ optional memory).

    ``include_data`` adds live-market tools; ``include_workflows`` adds the
    pipeline triggers. Pass ``workflow_launcher`` (web context) to make those
    triggers start background jobs instead of blocking.
    """
    tools: List[Tool] = _management_tools(service) + _analysis_tools()
    if include_data:
        tools += _data_tools()
    if include_workflows:
        tools += _workflow_tools(workflow_launcher)
    return {t.name: t for t in tools}


def tool_catalog_text(tools: Dict[str, Tool]) -> str:
    """Render the catalog for an LLM prompt: name, args, and description."""
    lines = []
    for t in tools.values():
        props = t.parameters.get("properties", {})
        req = set(t.parameters.get("required", []))
        args = ", ".join(f"{n}{'*' if n in req else ''}: {s.get('type', 'any')}"
                         for n, s in props.items()) or "(none)"
        lines.append(f"- {t.name}({args}) — {t.description}")
    return "\n".join(lines)
