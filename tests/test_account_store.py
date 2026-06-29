"""Tests for the SQL-backed account cash balance store (temp DB, no network)."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory.account_store import get_cash_balance, set_cash_balance


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db", dir=os.environ.get("TMPDIR"))
    os.close(fd)
    os.unlink(path)  # let sqlite create it fresh
    return path


def test_default_balance_is_zero():
    db = _tmp_db()
    try:
        assert get_cash_balance(db) == 0.0
    finally:
        os.path.exists(db) and os.unlink(db)


def test_set_and_get_balance():
    db = _tmp_db()
    try:
        assert set_cash_balance(25_000.0, db) == 25_000.0
        assert get_cash_balance(db) == 25_000.0
        set_cash_balance(31_500.50, db)
        assert get_cash_balance(db) == 31_500.50
    finally:
        os.path.exists(db) and os.unlink(db)


def test_negative_balance_rejected():
    db = _tmp_db()
    try:
        try:
            set_cash_balance(-1.0, db)
            assert False, "should have raised"
        except ValueError:
            pass
    finally:
        os.path.exists(db) and os.unlink(db)


def test_single_row_enforced():
    # The CHECK(id=1) + INSERT OR IGNORE means repeated opens never duplicate.
    db = _tmp_db()
    try:
        set_cash_balance(100.0, db)
        get_cash_balance(db)
        get_cash_balance(db)
        import sqlite3
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM account").fetchone()[0]
        conn.close()
        assert count == 1
    finally:
        os.path.exists(db) and os.unlink(db)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  ❌ {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  💥 {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    sys.exit(0 if passed == len(tests) else 1)
