"""SQL-backed account cash balance — the single updatable source of funds.

Why SQL (vs a config value or JSON file)? The balance changes over time as
trades open/close and as the human deposits/withdraws, and it must stay
consistent with the positions/transactions in the same database. A single-row
``account`` table (enforced by CHECK(id = 1)) is the natural home.

Pure helpers: ``get_cash_balance()`` and ``set_cash_balance()``. Both accept a
``db_path`` so tests can point at a throwaway database.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional, Union

from app.config import settings

logger = logging.getLogger("account-store")

_ENSURE_SQL = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash_balance REAL NOT NULL DEFAULT 0.0,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT OR IGNORE INTO account (id, cash_balance) VALUES (1, 0.0);
"""


def _connect(db_path: Union[str, Path]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_ENSURE_SQL)  # self-healing: create the table if missing
    return conn


def get_cash_balance(db_path: Optional[Union[str, Path]] = None) -> float:
    """Return the current account cash balance (0.0 if never set)."""
    path = db_path or settings.sql_db_path
    conn = _connect(path)
    try:
        row = conn.execute("SELECT cash_balance FROM account WHERE id = 1").fetchone()
        return float(row[0]) if row else 0.0
    finally:
        conn.close()


def set_cash_balance(amount: float, db_path: Optional[Union[str, Path]] = None) -> float:
    """Update the account cash balance and return the stored value."""
    if amount < 0:
        raise ValueError("Cash balance cannot be negative.")
    path = db_path or settings.sql_db_path
    conn = _connect(path)
    try:
        conn.execute(
            "UPDATE account SET cash_balance = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
            (float(amount),),
        )
        conn.commit()
        logger.info("Account cash balance set to $%.2f", amount)
        return float(amount)
    finally:
        conn.close()
