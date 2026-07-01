"""SQL decision & trade recorder.

Per design §3, the agent may only write to the persistent store AFTER human
feedback. So these functions are invoked by the human-feedback step (Component
11), never autonomously by the screener graph.

  * ``log_decision``    — append a row to decision_logs (approved OR denied).
  * ``open_position``   — create a position + its opening legs (buy 100 shares,
                          sell 1 call) on approval.
  * ``close_position``  — record the closing legs (buy-to-close call and/or sell
                          stock), set status + realized P&L.

EVERY transaction adjusts the single-row ``account`` cash balance atomically, so
the cash figure is always the SQL source of truth: buying stock and buying back
calls reduce cash; selling stock and collecting premium increase it.

All functions accept a ``db_path`` so tests use a throwaway database. The schema
(including the account table) is applied on connect if missing.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.config import settings

logger = logging.getLogger("decision-store")

_SELL_ACTIONS = {"SELL_TO_OPEN", "SELL_TO_CLOSE"}


def _connect(db_path: Union[str, Path]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    # Apply the schema (idempotent: every CREATE uses IF NOT EXISTS).
    schema = Path(settings.sql_schema_path)
    if schema.exists():
        conn.executescript(schema.read_text(encoding="utf-8"))
    # Migrate older DBs that predate the downside_buffer_percent column.
    try:
        conn.execute("ALTER TABLE positions ADD COLUMN downside_buffer_percent REAL")
    except sqlite3.OperationalError:
        pass  # column already exists
    return conn


def cash_effect(asset_type: str, action: str, quantity: int, price: float, fees: float = 0.0) -> float:
    """Signed cash impact of one transaction (options use the ×100 multiplier).

    Sells add cash, buys remove it; fees always reduce cash. E.g. selling 1 call
    at $1.20 → +$120; buying 100 shares at $60 → −$6,000.
    """
    multiplier = 100 if asset_type == "OPTION" else 1
    notional = price * quantity * multiplier
    signed = notional if action in _SELL_ACTIONS else -notional
    return round(signed - fees, 2)


def _adjust_cash(conn: sqlite3.Connection, delta: float) -> None:
    conn.execute(
        "UPDATE account SET cash_balance = cash_balance + ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
        (delta,),
    )


def log_decision(
    *,
    symbol: str,
    workflow_stage: str,
    agent_recommendation: Dict[str, Any],
    agent_rationale: str,
    is_human_approved: bool,
    human_feedback_notes: str = "",
    position_id: Optional[str] = None,
    tot_branches: Optional[Dict[str, Any]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Insert one decision_logs row. Returns the new log_id.

    Stores the agent's full recommendation + rationale alongside the human's
    approve/deny verdict and notes — the complete HITL cognitive ledger.
    """
    conn = _connect(db_path or settings.sql_db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO decision_logs
                (position_id, symbol, workflow_stage, agent_recommendation_json,
                 tot_branches_json, agent_rationale, is_human_approved, human_feedback_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position_id,
                symbol,
                workflow_stage,
                json.dumps(agent_recommendation, default=str),
                json.dumps(tot_branches, default=str) if tot_branches else None,
                agent_rationale,
                1 if is_human_approved else 0,
                human_feedback_notes,
            ),
        )
        conn.commit()
        log_id = int(cur.lastrowid)
        logger.info("Logged decision %d for %s (%s, approved=%s)",
                    log_id, symbol, workflow_stage, is_human_approved)
        return log_id
    finally:
        conn.close()


def add_transaction(
    conn: sqlite3.Connection,
    *,
    transaction_id: str,
    position_id: str,
    asset_type: str,
    action: str,
    quantity: int,
    price: float,
    fees: float = 0.0,
    strike_price: Optional[float] = None,
    expiration_date: Optional[str] = None,
    adjust_cash: bool = True,
) -> float:
    """Insert a transaction and (by default) adjust the account cash balance.
    Returns the signed cash effect applied."""
    conn.execute(
        """
        INSERT INTO transactions
            (transaction_id, position_id, asset_type, action, quantity, price,
             fees, strike_price, expiration_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (transaction_id, position_id, asset_type, action, quantity, price,
         fees, strike_price, expiration_date),
    )
    delta = cash_effect(asset_type, action, quantity, price, fees)
    if adjust_cash:
        _adjust_cash(conn, delta)
    return delta


def open_position(
    *,
    position_id: str,
    symbol: str,
    stock_purchase_price: float,
    shares: int,
    call_strike: float,
    call_premium: float,
    call_expiration: str,
    contracts: int = 1,
    fees: float = 0.0,
    downside_buffer_percent: Optional[float] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> str:
    """Create a position and its two opening legs (buy shares, sell call).

    ``downside_buffer_percent`` (premium cushion at entry) is stored so the
    defense monitor can use a per-position, dynamic breach threshold.
    Called when a human APPROVES an entry recommendation. Returns position_id.
    """
    conn = _connect(db_path or settings.sql_db_path)
    try:
        conn.execute(
            """
            INSERT INTO positions (position_id, symbol, status, stock_purchase_price,
                                   downside_buffer_percent)
            VALUES (?, ?, 'OPEN', ?, ?)
            """,
            (position_id, symbol, stock_purchase_price, downside_buffer_percent),
        )
        add_transaction(
            conn, transaction_id=f"{position_id}_STK", position_id=position_id,
            asset_type="STOCK", action="BUY_TO_OPEN", quantity=shares,
            price=stock_purchase_price, fees=fees,
        )
        add_transaction(
            conn, transaction_id=f"{position_id}_OPT", position_id=position_id,
            asset_type="OPTION", action="SELL_TO_OPEN", quantity=contracts,
            price=call_premium, fees=fees, strike_price=call_strike,
            expiration_date=call_expiration,
        )
        conn.commit()
        logger.info("Opened position %s: %d sh %s @ %.2f, sold %d call(s) @ %.2f "
                    "(net cash %+.2f)", position_id, shares, symbol, stock_purchase_price,
                    contracts, call_premium,
                    -stock_purchase_price * shares + call_premium * contracts * 100 - 2 * fees)
        return position_id
    finally:
        conn.close()


def _position_realized_pnl(conn: sqlite3.Connection, position_id: str) -> float:
    """Realized P&L of a position = the net cash flow across ALL its legs."""
    rows = conn.execute(
        "SELECT asset_type, action, quantity, price, fees FROM transactions WHERE position_id = ?",
        (position_id,),
    ).fetchall()
    return round(sum(cash_effect(r[0], r[1], r[2], r[3], r[4] or 0.0) for r in rows), 2)


def _realized_option_pnl(conn: sqlite3.Connection, position_id: str) -> float:
    """Realized P&L on an OPEN (rolled) position = net cash flow of the option
    legs that are already CLOSED.

    A covered-call roll buys back the current call (BUY_TO_CLOSE) and sells a new
    one (SELL_TO_OPEN). The stock and the newest short call are still open
    (unrealized), so realized P&L is every option leg's cash effect MINUS the
    single still-open short call (the latest SELL_TO_OPEN). This captures the
    locked-in income from completed roll cycles: premiums collected − buybacks.
    """
    option_total = sum(
        cash_effect(r[0], r[1], r[2], r[3], r[4] or 0.0)
        for r in conn.execute(
            "SELECT asset_type, action, quantity, price, fees FROM transactions "
            "WHERE position_id = ? AND asset_type = 'OPTION'", (position_id,)).fetchall())
    open_call = conn.execute(
        "SELECT quantity, price, fees FROM transactions WHERE position_id = ? "
        "AND asset_type = 'OPTION' AND action = 'SELL_TO_OPEN' "
        "ORDER BY timestamp DESC, transaction_id DESC LIMIT 1", (position_id,)).fetchone()
    open_call_cash = cash_effect("OPTION", "SELL_TO_OPEN", open_call[0], open_call[1],
                                 open_call[2] or 0.0) if open_call else 0.0
    return round(option_total - open_call_cash, 2)


def close_position(
    *,
    position_id: str,
    status: str,
    stock_sale_price: Optional[float] = None,
    shares: Optional[int] = None,
    call_buyback_price: Optional[float] = None,
    contracts: int = 1,
    fees: float = 0.0,
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Close (or partially settle) a position and update cash + realized P&L.

    Records the closing legs that apply:
      * ``call_buyback_price`` → BUY_TO_CLOSE the short call (cash out),
      * ``stock_sale_price``   → SELL_TO_CLOSE the shares (cash in; for an
        ASSIGNED position this is the strike, for a LIQUIDATION the market price).

    ``total_realized_pnl`` is recomputed as the position's net lifetime cash flow.
    Returns a summary dict. ``status`` is typically ASSIGNED, LIQUIDATED, or
    EXPIRED (the call expired worthless; pass no prices to just keep the shares).
    """
    conn = _connect(db_path or settings.sql_db_path)
    try:
        row = conn.execute(
            "SELECT status FROM positions WHERE position_id = ?", (position_id,)).fetchone()
        if row is None:
            raise ValueError(f"Position {position_id} not found.")

        if shares is None:  # default to the shares originally bought
            sh = conn.execute(
                "SELECT COALESCE(SUM(quantity), 0) FROM transactions WHERE position_id = ? "
                "AND asset_type = 'STOCK' AND action = 'BUY_TO_OPEN'", (position_id,)).fetchone()
            shares = int(sh[0] or 0)

        if call_buyback_price is not None:
            add_transaction(conn, transaction_id=f"{position_id}_BTC", position_id=position_id,
                            asset_type="OPTION", action="BUY_TO_CLOSE", quantity=contracts,
                            price=call_buyback_price, fees=fees)
        if stock_sale_price is not None and shares:
            add_transaction(conn, transaction_id=f"{position_id}_STC", position_id=position_id,
                            asset_type="STOCK", action="SELL_TO_CLOSE", quantity=shares,
                            price=stock_sale_price, fees=fees)

        realized = _position_realized_pnl(conn, position_id)
        conn.execute(
            "UPDATE positions SET status = ?, close_date = CURRENT_TIMESTAMP, total_realized_pnl = ? "
            "WHERE position_id = ?", (status, realized, position_id))
        conn.commit()
        logger.info("Closed position %s as %s: realized P&L $%.2f", position_id, status, realized)
        return {"position_id": position_id, "status": status, "total_realized_pnl": realized}
    finally:
        conn.close()


def roll_position(
    *,
    position_id: str,
    call_buyback_price: float,
    new_call_strike: float,
    new_call_premium: float,
    new_call_expiration: str,
    contracts: int = 1,
    fees: float = 0.0,
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Roll the short call (defense Branch B): buy-to-close the current call and
    sell-to-open a new (lower-strike) call. Position stays OPEN; cash is adjusted
    by the net credit. Returns the net credit + new short-call details."""
    conn = _connect(db_path or settings.sql_db_path)
    try:
        row = conn.execute("SELECT status FROM positions WHERE position_id = ?",
                           (position_id,)).fetchone()
        if row is None:
            raise ValueError(f"Position {position_id} not found.")
        seq = conn.execute("SELECT COUNT(*) FROM transactions WHERE position_id = ?",
                           (position_id,)).fetchone()[0]
        add_transaction(conn, transaction_id=f"{position_id}_BTC{seq}", position_id=position_id,
                        asset_type="OPTION", action="BUY_TO_CLOSE", quantity=contracts,
                        price=call_buyback_price, fees=fees)
        add_transaction(conn, transaction_id=f"{position_id}_STO{seq + 1}", position_id=position_id,
                        asset_type="OPTION", action="SELL_TO_OPEN", quantity=contracts,
                        price=new_call_premium, fees=fees, strike_price=new_call_strike,
                        expiration_date=new_call_expiration)
        # A roll realizes P&L on the closed option legs (premiums collected minus
        # the buyback). The position stays OPEN, so persist that locked-in income.
        realized = _realized_option_pnl(conn, position_id)
        conn.execute("UPDATE positions SET total_realized_pnl = ? WHERE position_id = ?",
                     (realized, position_id))
        conn.commit()
        net_credit = round((new_call_premium - call_buyback_price) * contracts * 100 - 2 * fees, 2)
        logger.info("Rolled %s: bought back @ %.2f, sold %.1f call @ %.2f (net credit %+.2f, "
                    "realized option P&L $%.2f)",
                    position_id, call_buyback_price, new_call_strike, new_call_premium,
                    net_credit, realized)
        return {"position_id": position_id, "net_credit": net_credit, "total_realized_pnl": realized,
                "new_strike": new_call_strike, "new_expiration": new_call_expiration}
    finally:
        conn.close()


def repair_zero_price_assignments(db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Repair positions closed with a $0 stock sale (an earlier ASSIGNED bug).

    For each SELL_TO_CLOSE stock leg priced at 0 whose position has a short-call
    strike, set the sale price to that strike, credit cash by the now-correct
    proceeds, and recompute realized P&L. Idempotent (a fixed leg is no longer 0).
    """
    conn = _connect(db_path or settings.sql_db_path)
    conn.row_factory = sqlite3.Row
    try:
        fixed: List[Dict[str, Any]] = []
        legs = conn.execute(
            "SELECT transaction_id, position_id, quantity FROM transactions "
            "WHERE asset_type = 'STOCK' AND action = 'SELL_TO_CLOSE' AND price = 0").fetchall()
        for leg in legs:
            pid, qty = leg["position_id"], int(leg["quantity"] or 0)
            call = conn.execute(
                "SELECT strike_price FROM transactions WHERE position_id = ? AND asset_type = 'OPTION' "
                "AND action = 'SELL_TO_OPEN' AND strike_price IS NOT NULL "
                "ORDER BY timestamp DESC, transaction_id DESC LIMIT 1", (pid,)).fetchone()
            if not call:
                continue
            strike = float(call["strike_price"])
            conn.execute("UPDATE transactions SET price = ? WHERE transaction_id = ?",
                         (strike, leg["transaction_id"]))
            _adjust_cash(conn, strike * qty)                 # credit the corrected sale proceeds
            realized = _position_realized_pnl(conn, pid)
            conn.execute("UPDATE positions SET total_realized_pnl = ? WHERE position_id = ?",
                         (realized, pid))
            fixed.append({"position_id": pid, "sale_price": strike,
                          "credited_cash": round(strike * qty, 2), "total_realized_pnl": realized})
        conn.commit()
        logger.info("Repaired %d zero-price assignment(s).", len(fixed))
        return fixed
    finally:
        conn.close()


def set_position_status(position_id: str, status: str,
                        db_path: Optional[Union[str, Path]] = None) -> bool:
    """Update only a position's status label (no cash/P&L effect). For closes
    with cash effects, use ``close_position`` instead."""
    conn = _connect(db_path or settings.sql_db_path)
    try:
        cur = conn.execute(
            "UPDATE positions SET status = ? WHERE position_id = ?", (status, position_id))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def list_positions(status: Optional[str] = None,
                   db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Return positions (optionally filtered by status), newest first."""
    conn = _connect(db_path or settings.sql_db_path)
    conn.row_factory = sqlite3.Row
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status = ? ORDER BY entry_date DESC", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM positions ORDER BY entry_date DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
