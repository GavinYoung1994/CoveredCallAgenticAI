"""Node 1 — The Scout Agent (Market & Data Scraper).

Job (per design §2): start from a broad watchlist and narrow it to liquid,
optionable, dividend-paying names. This is pure deterministic screening — no LLM
involved. Every dropped symbol is recorded in ``rejected`` with a reason so the
human can later audit *why* the universe shrank.

Filters applied (thresholds from ``StrategyRules``, all configurable):
  1. Has live market data (price & volume > 0)
  2. Average daily volume ≥ min_avg_daily_volume (default 1,000,000)
  3. Dividend yield ≥ min_dividend_yield_pct (default 2.0%)
  4. Optionable (has listed option expirations)  [if require_optionable]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional, Set

from app.config import rules as default_rules
from app.config import settings
from app.data.schwab_client import SchwabClient
from app.state import ScoutCandidate, ScreenerState, reject

logger = logging.getLogger("node.scout")


def load_watchlist(path: Optional[Path] = None) -> List[str]:
    """Read the symbol universe from watchlist.json."""
    p = path or settings.watchlist_path
    with open(p, "r") as f:
        data = json.load(f)
    return [s.upper() for s in data.get("symbols", [])]


def _held_open_symbols() -> Set[str]:
    """Symbols with an OPEN position — we don't re-screen names we already hold."""
    from app.memory.decision_store import list_positions
    return {str(p["symbol"]).upper() for p in list_positions("OPEN")}


def build_scout_node(
    client: SchwabClient,
    rules=default_rules,
    held_symbols_provider: Optional[Callable[[], Set[str]]] = None,
) -> Callable[[ScreenerState], dict]:
    """Return a Scout node bound to a Schwab client + strategy rules.

    ``held_symbols_provider`` returns the set of symbols to skip because they are
    already held (defaults to the OPEN positions in the SQL ledger). Injectable
    for testing.
    """
    held_provider = held_symbols_provider or _held_open_symbols

    def scout_node(state: ScreenerState) -> dict:
        symbols = state.get("watchlist") or []
        candidates: List[ScoutCandidate] = []
        rejections = []
        errors: List[str] = []

        if not symbols:
            return {"scout_candidates": [], "errors": ["Scout: empty watchlist."]}

        prefiltered = getattr(rules, "watchlist_is_prefiltered", False)
        try:
            held = {s.upper() for s in held_provider()}
        except Exception as exc:  # noqa: BLE001 — never let a memory hiccup crash the screen
            logger.warning("Scout could not load current holdings (%s); not filtering held names.", exc)
            held = set()
        logger.info(
            "Scout screening %d symbols (mode=%s, %d already held → skipped)",
            len(symbols), "liveness-only" if prefiltered else "full-filter", len(held),
        )

        # One chunked, rate-limited quote pass for the whole watchlist.
        try:
            quotes = client.get_quotes_chunked(symbols)
        except Exception as exc:  # noqa: BLE001 — surface as a run error, don't crash graph
            logger.exception("Scout failed to fetch quotes")
            return {"scout_candidates": [], "errors": [f"Scout quote fetch failed: {exc}"]}

        for sym in symbols:
            # Skip names we already hold (an open covered call) — no point
            # re-recommending one, and it keeps the portfolio diversified.
            if sym.upper() in held:
                rejections.append(reject(sym, "SCOUT", "Already held (open position)."))
                continue

            fund = client.extract_fundamentals(quotes, sym)

            # Liveness check (always): no live price ⇒ halted/delisted/bad symbol.
            if fund["last_price"] <= 0:
                rejections.append(reject(sym, "SCOUT", "No live price (halted/delisted/unknown symbol)."))
                continue

            if prefiltered:
                # Watchlist already curated → skip redundant fundamental filters
                # (and avoid 200 optionable API calls). Just carry the quote data.
                candidates.append({"symbol": sym, "fundamentals": fund, "is_optionable": True})
                continue

            # ── Full-filter mode (raw watchlist) ──────────────────────
            if fund["avg_daily_volume"] < rules.min_avg_daily_volume:
                rejections.append(reject(
                    sym, "SCOUT",
                    f"Illiquid: avg daily vol {fund['avg_daily_volume']:,} < {rules.min_avg_daily_volume:,}.",
                ))
                continue
            if fund["dividend_yield_percent"] < rules.min_dividend_yield_pct:
                rejections.append(reject(
                    sym, "SCOUT",
                    f"Dividend yield {fund['dividend_yield_percent']:.2f}% < {rules.min_dividend_yield_pct:.2f}%.",
                ))
                continue
            is_opt = True
            if rules.require_optionable:
                try:
                    is_opt = client.is_optionable(sym)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Scout optionable check failed for {sym}: {exc}")
                    is_opt = False
                if not is_opt:
                    rejections.append(reject(sym, "SCOUT", "Not optionable (no listed options)."))
                    continue
            candidates.append({"symbol": sym, "fundamentals": fund, "is_optionable": is_opt})

        logger.info("Scout passed %d/%d symbols", len(candidates), len(symbols))
        result = {"scout_candidates": candidates, "rejected": rejections}
        if errors:
            result["errors"] = errors
        return result

    return scout_node
