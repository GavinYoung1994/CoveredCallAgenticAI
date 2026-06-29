"""Load open positions from the SQL ledger for the defense scanner.

Reconstructs the ``OpenPosition`` shape the defense graph expects by joining a
position row with its opening option leg (the SELL_TO_OPEN call transaction,
which carries the strike, expiration, and premium) and its stock leg (shares).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Union

from app.config import settings
from app.memory.decision_store import _connect   # ensures schema + column migrations
from app.state import OpenPosition

logger = logging.getLogger("positions-store")


def load_open_positions(db_path: Optional[Union[str, Path]] = None) -> List[OpenPosition]:
    """Return every OPEN position as an OpenPosition dict.

    Positions without an opening short-call leg are skipped (can't evaluate a
    covered-call defense without the short call), with a warning.
    """
    conn = _connect(db_path or settings.sql_db_path)   # runs schema + migrations
    conn.row_factory = sqlite3.Row
    try:
        positions: List[OpenPosition] = []
        rows = conn.execute(
            "SELECT position_id, symbol, stock_purchase_price, downside_buffer_percent "
            "FROM positions WHERE status = 'OPEN'"
        ).fetchall()
        for r in rows:
            pid = r["position_id"]
            # The most recent opening short-call leg defines the current short call.
            call = conn.execute(
                "SELECT strike_price, expiration_date, price, quantity FROM transactions "
                "WHERE position_id = ? AND asset_type = 'OPTION' AND action = 'SELL_TO_OPEN' "
                "ORDER BY timestamp DESC, transaction_id DESC LIMIT 1",
                (pid,),
            ).fetchone()
            if not call or call["strike_price"] is None:
                logger.warning("Skipping %s: no opening short-call leg found.", pid)
                continue

            shares_row = conn.execute(
                "SELECT COALESCE(SUM(quantity), 0) AS sh FROM transactions "
                "WHERE position_id = ? AND asset_type = 'STOCK' AND action = 'BUY_TO_OPEN'",
                (pid,),
            ).fetchone()
            shares = int(shares_row["sh"] or 0) or int((call["quantity"] or 1) * 100)

            # Sum of all premiums collected on this position (informational).
            hist_row = conn.execute(
                "SELECT COALESCE(SUM(price), 0) AS p FROM transactions "
                "WHERE position_id = ? AND asset_type = 'OPTION' AND action = 'SELL_TO_OPEN'",
                (pid,),
            ).fetchone()

            positions.append({
                "position_id": pid,
                "symbol": r["symbol"],
                "stock_purchase_price": float(r["stock_purchase_price"]),
                "shares": shares,
                "short_call_strike": float(call["strike_price"]),
                "short_call_expiration": call["expiration_date"],
                "original_premium": float(call["price"] or 0.0),
                "historical_premiums_collected": float(hist_row["p"] or 0.0),
                "downside_buffer_percent": (float(r["downside_buffer_percent"])
                                            if r["downside_buffer_percent"] is not None else None),
            })
        logger.info("Loaded %d open position(s) for defense scan.", len(positions))
        return positions
    finally:
        conn.close()


def list_holdings_detailed(status: Optional[str] = None,
                           db_path: Optional[Union[str, Path]] = None) -> List[dict]:
    """Positions enriched with their CURRENT short-call contract for a UI view.

    Each row: position_id, symbol, status, dates, shares, stock cost basis,
    realized P&L, downside buffer, plus the latest short call's strike /
    expiration / premium / contracts (JSON-serializable dicts, newest first).
    """
    conn = _connect(db_path or settings.sql_db_path)   # runs schema + migrations
    conn.row_factory = sqlite3.Row
    try:
        clause = "WHERE status = ?" if status else ""
        args = (status.upper(),) if status else ()
        rows = conn.execute(
            f"SELECT * FROM positions {clause} ORDER BY entry_date DESC", args).fetchall()
        out: List[dict] = []
        for r in rows:
            pid = r["position_id"]
            call = conn.execute(
                "SELECT strike_price, expiration_date, price, quantity FROM transactions "
                "WHERE position_id = ? AND asset_type = 'OPTION' AND action = 'SELL_TO_OPEN' "
                "ORDER BY timestamp DESC, transaction_id DESC LIMIT 1", (pid,)).fetchone()
            shares = int(conn.execute(
                "SELECT COALESCE(SUM(quantity), 0) FROM transactions WHERE position_id = ? "
                "AND asset_type = 'STOCK' AND action = 'BUY_TO_OPEN'", (pid,)).fetchone()[0] or 0)
            contracts = int(call["quantity"]) if call else None
            premium = float(call["price"]) if call else None
            ret_pct, ann_pct, days, basis = _holding_return(
                status=r["status"], stock_price=r["stock_purchase_price"], shares=shares,
                realized_pnl=r["total_realized_pnl"], premium=premium, contracts=contracts,
                entry_date=r["entry_date"], close_date=r["close_date"],
                expiration=call["expiration_date"] if call else None)
            out.append({
                "position_id": pid,
                "symbol": r["symbol"],
                "status": r["status"],
                "entry_date": r["entry_date"],
                "close_date": r["close_date"],
                "shares": shares,
                "stock_purchase_price": r["stock_purchase_price"],
                "total_realized_pnl": r["total_realized_pnl"],
                "downside_buffer_percent": r["downside_buffer_percent"],
                "short_call_strike": call["strike_price"] if call else None,
                "short_call_expiration": call["expiration_date"] if call else None,
                "short_call_premium": premium,
                "contracts": contracts,
                "return_percent": ret_pct,
                "annualized_return_percent": ann_pct,
                "days_held": days,
                "return_basis": basis,
            })
        return out
    finally:
        conn.close()


def _to_date(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s)).date()
    except ValueError:
        try:
            return date.fromisoformat(str(s)[:10])
        except ValueError:
            return None


def _holding_return(*, status, stock_price, shares, realized_pnl, premium, contracts,
                    entry_date, close_date, expiration):
    """Return (return_pct, annualized_pct, days, basis) for a holding.

    Closed → realized P&L over the actual holding period. Open → premium income
    annualized to expiration (no live quote available in this view). Returns
    (None, None, None, basis) when capital or dates are missing.
    """
    invested = (stock_price or 0) * (shares or 0)
    entry = _to_date(entry_date)
    if status and status != "OPEN":
        pnl, end, basis = (realized_pnl or 0.0), _to_date(close_date), "realized"
    else:
        pnl = (premium or 0.0) * 100 * (contracts or 1)        # locked-in premium income
        end, basis = _to_date(expiration), "premium income (to expiration)"
    if not invested or entry is None or end is None:
        return None, None, None, basis
    days = (end - entry).days
    ret_pct = round(pnl / invested * 100, 2)
    ann_pct = round(ret_pct * 365 / days, 2) if days > 0 else None
    return ret_pct, ann_pct, days, basis
