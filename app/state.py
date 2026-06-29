"""LangGraph state schemas — the typed contract shared across agent nodes.

In LangGraph, the graph's *state* is the single object every node reads from and
writes to. A node returns a partial dict; LangGraph merges it into the running
state using each field's "reducer". For most fields the default reducer simply
overwrites, which is what we want for a sequential pipeline (Scout sets
``scout_candidates``, Quant sets ``quant_candidates``, etc.).

Two fields — ``rejected`` and ``errors`` — use an *append* reducer
(``Annotated[list, operator.add]``) so that EVERY node can contribute to a
running audit trail without overwriting what earlier nodes recorded. That trail
("NVDA dropped at QUANT: delta 0.52 outside band") is exactly what the design's
human-critique feedback loop needs.

This module is intentionally dependency-free (stdlib typing only) so it imports
and tests without LangGraph installed.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

# ── Workflow stages (mirror the SQL decision_logs.workflow_stage values) ──
WorkflowStage = Literal["ENTRY_SCREENER", "DEFENSE_MONITOR", "REVIEW_SUBAGENT"]
PipelineStage = Literal["SCOUT", "QUANT", "NEWS", "RISK_MANAGER"]
SentimentLabel = Literal["VERY_NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "VERY_POSITIVE"]


# ══════════════════════════════════════════════════════════════════════
#  Per-symbol record types (nested inside the state)
# ══════════════════════════════════════════════════════════════════════
class Fundamentals(TypedDict, total=False):
    symbol: str
    asset_type: Optional[str]
    last_price: float
    total_volume: int
    avg_daily_volume: int
    dividend_yield_percent: float
    dividend_amount: float
    next_div_ex_date: Optional[str]
    pe_ratio: Optional[float]


class ScoutCandidate(TypedDict, total=False):
    """A symbol that cleared the Scout's liquidity/dividend/optionable filters."""
    symbol: str
    fundamentals: Fundamentals
    is_optionable: bool


class ContractPick(TypedDict, total=False):
    """The single chosen covered-call contract for a symbol."""
    symbol: str            # option symbol (OCC), not the underlying
    strike: float
    expiration_key: str
    days_to_expiration: int
    delta: float
    bid: float
    ask: float
    mark: float
    volume: int
    open_interest: int
    in_delta_band: bool


class QuantCandidate(TypedDict, total=False):
    """A fully-analyzed candidate produced by the Quant node."""
    symbol: str
    underlying_price: float
    snapshot: Dict[str, Any]          # RSI/SMA/Bollinger snapshot
    trend: Dict[str, Any]             # trend classification
    contract: ContractPick
    greeks: Dict[str, Any]            # Black-Scholes greeks + prob assignment
    yield_metrics: Dict[str, Any]     # AROC etc.
    iv_rank: Optional[Dict[str, Any]]
    expected_move: Optional[Dict[str, Any]]
    liquidity: Dict[str, Any]         # bid-ask spread guard result
    passes_quant: bool


class NewsReport(TypedDict, total=False):
    """The News/Sentiment node's verdict for one symbol."""
    symbol: str
    sentiment: SentimentLabel
    sentiment_score: int              # 1..5 mapped from the label
    rationale: str
    headlines_checked: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]     # concise {title, publisher, url, published_utc} for human review
    earnings_date: Optional[str]
    earnings_known: bool
    earnings_disqualifies: bool
    catastrophic_risk: bool
    catastrophic_keywords: List[str]
    passes_news: bool


class Recommendation(TypedDict, total=False):
    """A graded final recommendation surfaced to the human via Discord."""
    symbol: str
    grade: str                        # e.g. 'A', 'B', 'C'
    score: float                      # composite 0..100
    annualized_yield_percent: float
    contract: ContractPick
    yield_metrics: Dict[str, Any]
    sentiment: SentimentLabel
    iv_to_hv_ratio: float
    prob_keep_premium_percent: float
    earnings_known: bool
    earnings_date: Optional[str]
    sources: List[Dict[str, Any]]     # news sources behind the sentiment call
    rationale: str
    score_components: Dict[str, float]


class Rejection(TypedDict, total=False):
    """One entry in the audit trail of dropped candidates.

    ``sources`` is populated for NEWS-stage rejections so a human can review the
    exact headlines behind a negative/sentiment call.
    """
    symbol: str        # required in practice
    stage: str         # required in practice
    reason: str        # required in practice
    sources: List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════════════════
#  Graph states
# ══════════════════════════════════════════════════════════════════════
class ScreenerState(TypedDict, total=False):
    """State for the entry-screener graph: Scout → Quant → News → Risk."""
    # Inputs
    run_id: str
    run_timestamp: str
    mode: WorkflowStage
    account_cash: float
    watchlist: List[str]

    # Per-stage outputs (default reducer = overwrite)
    scout_candidates: List[ScoutCandidate]
    quant_candidates: List[QuantCandidate]
    news_reports: List[NewsReport]
    recommendations: List[Recommendation]

    # Cross-cutting, append-reduced audit trail
    rejected: Annotated[List[Rejection], operator.add]
    errors: Annotated[List[str], operator.add]

    # Optional: the human-readable summary the Risk Manager sends to Discord
    discord_summary: str
    notified: bool
    run_log_paths: Dict[str, str]     # {"markdown": ..., "json": ...}


class OpenPosition(TypedDict, total=False):
    """An existing covered-call position the Defense Monitor evaluates."""
    position_id: str
    symbol: str
    stock_purchase_price: float
    shares: int
    short_call_strike: float
    short_call_expiration: str
    original_premium: float
    historical_premiums_collected: float
    downside_buffer_percent: Optional[float]   # premium cushion at entry → dynamic breach threshold


class DefenseState(TypedDict, total=False):
    """State for the Tree-of-Thoughts downside-defense graph.

    Branches are evaluated Quant → News → Risk (per design §4)."""
    run_id: str
    run_timestamp: str
    mode: WorkflowStage
    position: OpenPosition
    current_stock_price: float
    current_call_ask: float
    roll_down_premium: float
    breach_detected: bool

    branch_analysis: Dict[str, Any]   # ToT P&L for branches A/B/C (Quant)
    news_report: NewsReport           # News node's read on the position
    defense_recommendation: Recommendation
    discord_summary: str
    notified: bool
    run_log_paths: Dict[str, str]

    rejected: Annotated[List[Rejection], operator.add]
    errors: Annotated[List[str], operator.add]


# ══════════════════════════════════════════════════════════════════════
#  Small constructors / helpers (keep nodes terse and consistent)
# ══════════════════════════════════════════════════════════════════════
def new_screener_state(
    *,
    watchlist: List[str],
    account_cash: float,
    run_id: str,
    run_timestamp: str,
    mode: WorkflowStage = "ENTRY_SCREENER",
) -> ScreenerState:
    """Build a fresh, fully-initialized screener state.

    All list fields start empty so reducers/append operations are always safe.
    """
    return {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "mode": mode,
        "account_cash": account_cash,
        "watchlist": list(watchlist),
        "scout_candidates": [],
        "quant_candidates": [],
        "news_reports": [],
        "recommendations": [],
        "rejected": [],
        "errors": [],
        "notified": False,
    }


def reject(
    symbol: str, stage: str, reason: str,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> Rejection:
    """Construct one audit-trail rejection record.

    Pass ``sources`` (the headlines reviewed) for NEWS-stage rejections so the
    human can verify a sentiment/catastrophic-risk call was correct.
    """
    rec: Rejection = {"symbol": symbol, "stage": stage, "reason": reason}
    if sources:
        rec["sources"] = sources
    return rec


SENTIMENT_TO_SCORE: Dict[str, int] = {
    "VERY_NEGATIVE": 1,
    "NEGATIVE": 2,
    "NEUTRAL": 3,
    "POSITIVE": 4,
    "VERY_POSITIVE": 5,
}


def sentiment_score(label: str) -> int:
    """Map a sentiment label → 1..5 (unknown labels default to NEUTRAL=3)."""
    return SENTIMENT_TO_SCORE.get(str(label).upper(), 3)
