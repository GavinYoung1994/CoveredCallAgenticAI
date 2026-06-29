"""Entry-screener graph: Scout → Quant → News → Risk Manager.

This is the daily pipeline. It is a strictly sequential LangGraph: each node
narrows the candidate pool and writes to the shared ScreenerState, with the
``rejected``/``errors`` audit trail accumulating across all nodes (append
reducers). The Risk Manager surfaces the top-N to Discord + the local run log
for human approval — the graph itself NEVER writes to the SQL ledger (design §3).

``build_entry_screener_graph`` takes injected dependencies (so tests use mocks);
``run_entry_screener`` builds real dependencies from config and runs one pass.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, List, Optional

from langgraph.graph import END, START, StateGraph

from app.config import rules as default_rules
from app.config import settings
from app.logging_config import setup_logging
from app.data.earnings_client import EarningsClient
from app.data.earnings_search import EarningsSearchClient, CompositeEarningsClient
from app.data.news_client import NewsClient
from app.data.schwab_client import SchwabClient
from app.llm import LocalLLM, get_llm
from app.nodes.scout import build_scout_node, load_watchlist
from app.nodes.quant import build_quant_node
from app.nodes.news import build_news_node
from app.nodes.risk_manager import build_risk_manager_node
from app.notify.discord_webhook import DiscordNotifier
from app.memory.account_store import get_cash_balance
from app.state import ScreenerState, new_screener_state

logger = logging.getLogger("graph.entry")


def build_entry_screener_graph(
    *,
    schwab_client: SchwabClient,
    news_client: NewsClient,
    earnings_client: EarningsClient,
    llm: LocalLLM,
    notifier: Optional[DiscordNotifier] = None,
    rules=default_rules,
    today: Optional[date] = None,
):
    """Compile the Scout→Quant→News→Risk graph from injected dependencies."""
    graph = StateGraph(ScreenerState)

    graph.add_node("scout", build_scout_node(schwab_client, rules=rules))
    graph.add_node("quant", build_quant_node(schwab_client, rules=rules, today=today))
    graph.add_node("news", build_news_node(news_client, earnings_client, llm, rules=rules, today=today))
    graph.add_node("risk_manager", build_risk_manager_node(llm, notifier=notifier, rules=rules))

    graph.add_edge(START, "scout")
    graph.add_edge("scout", "quant")
    graph.add_edge("quant", "news")
    graph.add_edge("news", "risk_manager")
    graph.add_edge("risk_manager", END)

    return graph.compile()


def run_entry_screener(
    *,
    watchlist: Optional[List[str]] = None,
    account_cash: Optional[float] = None,
    rules=default_rules,
    today: Optional[date] = None,
    # Dependency overrides (tests inject these; production builds defaults).
    schwab_client: Optional[SchwabClient] = None,
    news_client: Optional[NewsClient] = None,
    earnings_client: Optional[EarningsClient] = None,
    llm: Optional[LocalLLM] = None,
    notifier: Optional[DiscordNotifier] = None,
    run_id: Optional[str] = None,
    run_timestamp: Optional[str] = None,
) -> ScreenerState:
    """Run one screening pass and return the final state.

    Defaults: watchlist from watchlist.json, cash from the SQL account table,
    real Schwab/news/earnings clients, the local LLM, and a Discord notifier.
    """
    setup_logging()  # make node progress visible on the console
    now = datetime.now(timezone.utc)
    run_id = run_id or f"entry_{now.strftime('%Y%m%d_%H%M%S')}"
    run_timestamp = run_timestamp or now.isoformat()

    symbols = watchlist if watchlist is not None else load_watchlist()
    cash = account_cash if account_cash is not None else get_cash_balance()

    schwab_client = schwab_client or SchwabClient()
    news_client = news_client or NewsClient()
    llm = llm or get_llm()
    if earnings_client is None:
        # Structured Finnhub first; fall back to the Google-search engine (which
        # uses the LLM to disambiguate the date) when Finnhub returns nothing.
        providers = [EarningsClient()]
        if settings.earnings_search_enabled:
            providers.append(EarningsSearchClient(llm=llm))
        earnings_client = CompositeEarningsClient(providers)
    notifier = notifier if notifier is not None else DiscordNotifier()

    app = build_entry_screener_graph(
        schwab_client=schwab_client, news_client=news_client,
        earnings_client=earnings_client, llm=llm, notifier=notifier,
        rules=rules, today=today,
    )
    state = new_screener_state(
        watchlist=symbols, account_cash=cash, run_id=run_id, run_timestamp=run_timestamp)

    logger.info("Running entry screener %s on %d symbols (cash=$%.0f)", run_id, len(symbols), cash)
    final: ScreenerState = app.invoke(state)
    logger.info(
        "Run %s complete: %d recommendations, %d rejected.",
        run_id, len(final.get("recommendations", [])), len(final.get("rejected", [])),
    )
    return final
