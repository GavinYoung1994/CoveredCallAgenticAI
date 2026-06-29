"""Natural-language DB management interface.

A small LLM-driven console for managing your trading data without writing SQL:

    ./venv/bin/python -m app.manage
    > what's my cash balance?
    > set cash to 50000
    > show my open holdings
    > mark KO_KO_20260626 as assigned, stock sold at 62.5
    > what have we learned about utilities?

Architecture mirrors the rest of the app: deterministic functions do the actual
DB work (``ManagementService``), and the LLM only maps a natural-language request
to one ``{action, args}`` call (``_Router``) — it never touches the database
directly. Both layers are independently testable.
"""

from __future__ import annotations

import inspect
import logging
import sqlite3
import sys
from typing import Any, Callable, Dict, List, Optional

from app.config import settings
from app.llm import LocalLLM
from app.memory.account_store import get_cash_balance, set_cash_balance
from app.memory.decision_store import close_position, list_positions, set_position_status
from app.memory.positions_store import list_holdings_detailed

logger = logging.getLogger("manage")

_CLOSING_STATUSES = {"ASSIGNED", "LIQUIDATED", "EXPIRED", "CLOSED"}


class ManagementService:
    """Deterministic data-management operations (the 'tools' the LLM can call)."""

    def __init__(self, db_path: Optional[str] = None, memory: Any = None) -> None:
        self.db_path = db_path or str(settings.sql_db_path)
        self._memory = memory  # a TradeMemory (or None)

    # ── cash ──────────────────────────────────────────────────────────
    def get_cash(self) -> Dict[str, Any]:
        return {"cash_balance": get_cash_balance(self.db_path)}

    def set_cash(self, amount: float) -> Dict[str, Any]:
        return {"cash_balance": set_cash_balance(float(amount), self.db_path)}

    # ── holdings ──────────────────────────────────────────────────────
    def list_holdings(self, status: Optional[str] = None) -> Dict[str, Any]:
        return {"positions": list_positions(status.upper() if status else None, self.db_path)}

    def get_position(self, position_id: str) -> Dict[str, Any]:
        match = [p for p in list_positions(None, self.db_path) if p["position_id"] == position_id]
        return {"position": match[0]} if match else {"error": f"No position {position_id}."}

    def _resolve_position(self, position_id: Optional[str], symbol: Optional[str]) -> Optional[Dict[str, Any]]:
        """Find a holding by id, or by symbol (prefers an OPEN one)."""
        holdings = list_holdings_detailed(db_path=self.db_path)
        if position_id:
            return next((h for h in holdings if h["position_id"] == position_id), None)
        if symbol:
            matches = [h for h in holdings if h["symbol"].upper() == symbol.upper()]
            return next((h for h in matches if h["status"] == "OPEN"), matches[0] if matches else None)
        return None

    def update_holding_status(
        self, status: str,
        position_id: Optional[str] = None,
        symbol: Optional[str] = None,
        stock_sale_price: Optional[float] = None,
        call_buyback_price: Optional[float] = None,
        contracts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Record an executed/closed trade for a holding (identify it by symbol OR
        position_id). For a closing status (ASSIGNED/LIQUIDATED/EXPIRED) this records
        the closing legs, adjusts cash, and computes realized P&L.

        Smart defaults: for ASSIGNED (called away), the stock sale price defaults to
        the short-call strike and the call buyback to $0 (the call was exercised), so
        you don't need to supply fills. LIQUIDATED needs a stock_sale_price."""
        status = status.upper()
        pos = self._resolve_position(position_id, symbol)
        if pos is None:
            return {"error": f"No holding found for {position_id or symbol!r}."}
        pid = pos["position_id"]
        contracts = contracts or pos.get("contracts") or 1

        if status in _CLOSING_STATUSES:
            if status == "ASSIGNED":   # called away at the strike; call exercised (no buyback)
                # Treat 0 / non-positive as "not provided" — a $0 assignment sale
                # is never valid, so always fall back to the strike.
                if not stock_sale_price or float(stock_sale_price) <= 0:
                    stock_sale_price = pos.get("short_call_strike")
                if call_buyback_price is None:
                    call_buyback_price = 0.0
                if not stock_sale_price or float(stock_sale_price) <= 0:
                    return {"error": f"Cannot determine the assignment (strike) sale price for "
                                     f"{pid}; pass stock_sale_price explicitly."}
            if status == "LIQUIDATED" and (not stock_sale_price or float(stock_sale_price) <= 0):
                return {"error": f"LIQUIDATED requires a positive stock_sale_price for {pid}."}
            if stock_sale_price is not None or call_buyback_price is not None:
                return close_position(
                    position_id=pid, status=status, stock_sale_price=stock_sale_price,
                    call_buyback_price=call_buyback_price, contracts=contracts, db_path=self.db_path)
        ok = set_position_status(pid, status, self.db_path)
        return {"position_id": pid, "status": status, "updated": ok}

    # ── learnings ─────────────────────────────────────────────────────
    def search_learnings(self, query: str, n: int = 5) -> Dict[str, Any]:
        if self._memory is None:
            return {"error": "Vector memory is not available."}
        return {"lessons": self._memory.query(query, n_results=int(n))}

    # ── decisions ─────────────────────────────────────────────────────
    def recent_decisions(self, n: int = 10) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT log_id, symbol, workflow_stage, is_human_approved, human_feedback_notes "
                "FROM decision_logs ORDER BY log_id DESC LIMIT ?", (int(n),)).fetchall()
            return {"decisions": [dict(r) for r in rows]}
        finally:
            conn.close()


# Catalog the LLM picks from. Each entry: method name → (description, arg list).
_ACTIONS: Dict[str, str] = {
    "get_cash": "Show the current account cash balance. args: {}",
    "set_cash": "Set the cash balance. args: {amount: number}",
    "list_holdings": "List positions. args: {status?: OPEN|ASSIGNED|LIQUIDATED|EXPIRED}",
    "get_position": "Show one position. args: {position_id: string}",
    "update_holding_status": ("Record an executed/closed trade (assigned/liquidated/expired). "
                              "args: {status: string, symbol?: string, position_id?: string, "
                              "stock_sale_price?: number, call_buyback_price?: number, contracts?: number}"),
    "search_learnings": "Semantic search of past trade lessons. args: {query: string, n?: number}",
    "recent_decisions": "Recent approve/deny decisions. args: {n?: number}",
}

_ROUTER_SYSTEM = (
    "You are a database assistant for a covered-call trading app. Map the user's "
    "request to exactly ONE action from the catalog and its arguments. Use only "
    "actions and argument names from the catalog. If nothing fits, use action "
    '"none".'
)


class _Router:
    def __init__(self, llm: LocalLLM) -> None:
        self._llm = llm

    def route(self, message: str) -> Dict[str, Any]:
        catalog = "\n".join(f"- {name}: {desc}" for name, desc in _ACTIONS.items())
        user = (f"Action catalog:\n{catalog}\n\nUser request: {message}\n\n"
                'Return JSON: {"action": "<name or none>", "args": {{...}}}.')
        try:
            obj = self._llm.structured(_ROUTER_SYSTEM, user, required_keys=["action", "args"])
        except Exception as exc:  # noqa: BLE001
            return {"action": "none", "args": {}, "error": str(exc)}
        if not isinstance(obj.get("args"), dict):
            obj["args"] = {}
        return obj


def _call_filtered(fn: Callable, args: Dict[str, Any]) -> Any:
    """Call ``fn`` with only the kwargs it actually accepts (ignore extras)."""
    accepted = set(inspect.signature(fn).parameters)
    return fn(**{k: v for k, v in args.items() if k in accepted})


def handle(message: str, service: ManagementService, router: _Router) -> Dict[str, Any]:
    """Route a natural-language request to a service action and execute it."""
    spec = router.route(message)
    action = str(spec.get("action", "none"))
    args = spec.get("args", {})
    if action == "none":
        return {"error": "Could not map the request to an action.", "available": list(_ACTIONS)}
    method = getattr(service, action, None)
    if method is None:
        return {"error": f"Unknown action '{action}'.", "available": list(_ACTIONS)}
    try:
        return _call_filtered(method, args)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Action %s failed", action)
        return {"error": f"{action} failed: {exc}"}


def main() -> int:  # pragma: no cover - interactive
    """Interactive console backed by the multi-step CoveredCallAgent (the rigid
    single-action ``_Router`` above is kept for back-compat / lightweight use)."""
    from app.llm import get_llm
    from app.memory.vector_db import TradeMemory
    from app.logging_config import setup_logging
    from app.agent.tools import build_tools
    from app.agent.agent import CoveredCallAgent

    setup_logging("WARNING")  # keep the console clean for a chat UI
    service = ManagementService(memory=TradeMemory())
    agent = CoveredCallAgent(get_llm(), build_tools(service, service._memory))
    print("Covered Call Command Center. Ask me anything (or 'quit').")
    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in ("quit", "exit", "q"):
            break
        if not message:
            continue
        print(agent.chat(message)["answer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
