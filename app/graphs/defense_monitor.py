"""Defense-monitor graph: the Tree-of-Thoughts downside engine (design §4).

    quant  → (breach?) → news → risk → END
              └ no breach ─────────────→ END

For an open position, the Quant node detects a downside breach and generates the
three escape branches; only on a breach do News + Risk Manager run. The Risk
Manager picks a branch and alerts the human (execution stays manual).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from app.config import rules as default_rules
from app.logging_config import setup_logging
from app.data.news_client import NewsClient
from app.data.schwab_client import SchwabClient
from app.llm import LocalLLM, get_llm
from app.nodes.defense import (
    build_defense_quant_node,
    build_defense_news_node,
    build_defense_risk_node,
)
from app.notify.discord_webhook import DiscordNotifier
from app.memory.positions_store import load_open_positions
from app.state import DefenseState

logger = logging.getLogger("graph.defense")


def _route_after_quant(state: DefenseState) -> str:
    return "news" if state.get("breach_detected") else "end"


def build_defense_monitor_graph(
    *,
    schwab_client: SchwabClient,
    news_client: NewsClient,
    llm: LocalLLM,
    notifier: Optional[DiscordNotifier] = None,
    rules=default_rules,
    today: Optional[date] = None,
):
    graph = StateGraph(DefenseState)
    graph.add_node("quant", build_defense_quant_node(schwab_client, rules=rules, today=today))
    graph.add_node("news", build_defense_news_node(news_client, llm, rules=rules))
    graph.add_node("risk", build_defense_risk_node(llm, notifier=notifier, rules=rules))

    graph.add_edge(START, "quant")
    graph.add_conditional_edges("quant", _route_after_quant, {"news": "news", "end": END})
    graph.add_edge("news", "risk")
    graph.add_edge("risk", END)
    return graph.compile()


def run_defense_monitor(
    position: Dict[str, Any],
    *,
    current_stock_price: Optional[float] = None,
    current_call_ask: Optional[float] = None,
    roll_down_premium: Optional[float] = None,
    rules=default_rules,
    today: Optional[date] = None,
    schwab_client: Optional[SchwabClient] = None,
    news_client: Optional[NewsClient] = None,
    llm: Optional[LocalLLM] = None,
    notifier: Optional[DiscordNotifier] = None,
    run_id: Optional[str] = None,
    run_timestamp: Optional[str] = None,
) -> DefenseState:
    """Evaluate downside defense for a single open position and return final state.

    Market inputs (price/call ask/roll premium) may be passed in; anything left
    None is fetched from Schwab.
    """
    setup_logging()
    now = datetime.now(timezone.utc)
    run_id = run_id or f"defense_{position.get('symbol','?')}_{now.strftime('%Y%m%d_%H%M%S')}"

    schwab_client = schwab_client or SchwabClient()
    news_client = news_client or NewsClient()
    llm = llm or get_llm()
    notifier = notifier if notifier is not None else DiscordNotifier()

    app = build_defense_monitor_graph(
        schwab_client=schwab_client, news_client=news_client, llm=llm,
        notifier=notifier, rules=rules, today=today)

    state: DefenseState = {
        "run_id": run_id,
        "run_timestamp": run_timestamp or now.isoformat(),
        "mode": "DEFENSE_MONITOR",
        "position": position,
        "rejected": [],
        "errors": [],
        "notified": False,
    }
    if current_stock_price is not None:
        state["current_stock_price"] = current_stock_price
    if current_call_ask is not None:
        state["current_call_ask"] = current_call_ask
    if roll_down_premium is not None:
        state["roll_down_premium"] = roll_down_premium

    logger.info("Running defense monitor %s for %s", run_id, position.get("symbol"))
    return app.invoke(state)


def run_defense_scan(
    *,
    db_path: Optional[str] = None,
    rules=default_rules,
    today: Optional[date] = None,
    schwab_client: Optional[SchwabClient] = None,
    news_client: Optional[NewsClient] = None,
    llm: Optional[LocalLLM] = None,
    notifier: Optional[DiscordNotifier] = None,
    run_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan ALL open positions and run downside defense on each (no manual loop).

    Loads open positions from the SQL ledger, fetches live market data per name,
    and evaluates the ToT branches. Breached positions get a branch recommendation
    sent to Discord. Returns a summary: how many scanned, which breached, and the
    per-position final states.
    """
    setup_logging()
    now = datetime.now(timezone.utc)
    run_timestamp = run_timestamp or now.isoformat()

    schwab_client = schwab_client or SchwabClient()
    news_client = news_client or NewsClient()
    llm = llm or get_llm()
    notifier = notifier if notifier is not None else DiscordNotifier()

    positions = load_open_positions(db_path)
    if not positions:
        logger.info("Defense scan: no open positions to evaluate.")
        return {"scanned": 0, "breached": [], "results": []}

    app = build_defense_monitor_graph(
        schwab_client=schwab_client, news_client=news_client, llm=llm,
        notifier=notifier, rules=rules, today=today)

    results, breached = [], []
    logger.info("Defense scan: evaluating %d open position(s).", len(positions))
    for pos in positions:
        run_id = f"defense_{pos['symbol']}_{now.strftime('%Y%m%d_%H%M%S')}"
        state: DefenseState = {
            "run_id": run_id, "run_timestamp": run_timestamp, "mode": "DEFENSE_MONITOR",
            "position": pos, "rejected": [], "errors": [], "notified": False,
        }
        try:
            final = app.invoke(state)
        except Exception as exc:  # noqa: BLE001 — one bad position shouldn't stop the scan
            logger.exception("Defense scan failed for %s", pos["symbol"])
            results.append({"symbol": pos["symbol"], "error": str(exc)})
            continue
        results.append(final)
        if final.get("breach_detected"):
            breached.append(pos["symbol"])

    logger.info("Defense scan complete: %d scanned, %d breached (%s).",
                len(positions), len(breached), ", ".join(breached) or "none")
    return {"scanned": len(positions), "breached": breached, "results": results}
