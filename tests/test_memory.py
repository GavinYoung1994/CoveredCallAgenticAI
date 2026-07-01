"""Tests for the memory layer: TradeMemory (ChromaDB) + SQL decision_store.

TradeMemory wrapper logic is tested with an injected fake collection (no torch /
chromadb needed). A guarded test also exercises a REAL Chroma collection with a
light hash embedder (skipped if chromadb isn't installed). decision_store uses a
temp SQLite DB.
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory.vector_db import TradeMemory
from app.memory import decision_store as ds


# ── fake Chroma collection ────────────────────────────────────────────
class FakeCollection:
    def __init__(self):
        self.ids, self.docs, self.metas = [], [], []

    def add(self, ids, documents, metadatas):
        self.ids += ids; self.docs += documents; self.metas += metadatas

    def query(self, query_texts, n_results):
        k = min(n_results, len(self.ids))
        return {"ids": [self.ids[:k]], "documents": [self.docs[:k]],
                "metadatas": [self.metas[:k]], "distances": [[0.1] * k]}

    def count(self):
        return len(self.ids)


def test_trade_memory_add_and_query():
    coll = FakeCollection()
    mem = TradeMemory(collection=coll)
    mem.add_lesson("t1", "Sold KO calls in high IV, assigned for 12% annualized.",
                   {"symbol": "KO", "outcome": "ASSIGNED", "pnl": 350.0, "ignored": ["x"]})
    assert mem.count() == 1
    # Non-primitive metadata ('ignored' list) must be dropped, primitives kept.
    assert coll.metas[0] == {"symbol": "KO", "outcome": "ASSIGNED", "pnl": 350.0}
    results = mem.query("covered calls on consumer staples", n_results=3)
    assert results[0]["id"] == "t1"
    assert results[0]["metadata"]["symbol"] == "KO"
    assert results[0]["distance"] == 0.1


def test_embedding_backend_auto_falls_back_to_default():
    # sentence_transformers isn't installed in the test env → 'auto' must pick
    # ChromaDB's built-in ONNX default (so lessons store without torch).
    mem = TradeMemory(embedding_backend="auto")
    assert mem._pick_embedding_backend() == "default"


def test_embedding_backend_explicit_override():
    assert TradeMemory(embedding_backend="default")._pick_embedding_backend() == "default"
    assert TradeMemory(embedding_backend="sentence_transformers")._pick_embedding_backend() \
        == "sentence_transformers"


def test_trade_memory_real_chroma_roundtrip():
    try:
        import chromadb
        from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
    except Exception as exc:  # noqa: BLE001
        print(f"  ⏭  skipping real Chroma test (chromadb unavailable: {exc})")
        return

    class HashEF(EmbeddingFunction):
        # Deterministic light embedding — avoids downloading sentence-transformers.
        def __call__(self, input: "Documents") -> "Embeddings":
            return [[float(sum(t.encode()) % 1000), float(len(t)), 1.0] for t in input]

    client = chromadb.EphemeralClient()
    coll = client.get_or_create_collection(name="t_lessons", embedding_function=HashEF())
    mem = TradeMemory(collection=coll)
    mem.add_lesson("r1", "XOM covered call rolled down successfully.",
                   {"symbol": "XOM", "outcome": "ROLLED"})
    mem.add_lesson("r2", "PFE call expired worthless, kept full premium.",
                   {"symbol": "PFE", "outcome": "EXPIRED"})
    assert mem.count() == 2
    res = mem.query("energy stock roll", n_results=2)
    assert {r["id"] for r in res} == {"r1", "r2"}
    assert all("distance" in r for r in res)


# ── decision_store ────────────────────────────────────────────────────
def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db", dir=os.environ.get("TMPDIR"))
    os.close(fd); os.unlink(path)
    return path


def test_log_decision_approved_and_denied():
    db = _tmp_db()
    try:
        approved = ds.log_decision(
            symbol="KO", workflow_stage="ENTRY_SCREENER",
            agent_recommendation={"strike": 60, "delta": 0.33},
            agent_rationale="Rich IV, positive sentiment.",
            is_human_approved=True, human_feedback_notes="Looks good.", db_path=db)
        denied = ds.log_decision(
            symbol="XOM", workflow_stage="ENTRY_SCREENER",
            agent_recommendation={"strike": 120},
            agent_rationale="Yield ok.", is_human_approved=False,
            human_feedback_notes="Earnings too close.", db_path=db)
        assert approved == 1 and denied == 2
        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT symbol, is_human_approved, human_feedback_notes FROM decision_logs ORDER BY log_id"
        ).fetchall()
        conn.close()
        assert rows[0] == ("KO", 1, "Looks good.")
        assert rows[1] == ("XOM", 0, "Earnings too close.")
    finally:
        os.path.exists(db) and os.unlink(db)


def test_open_position_creates_legs():
    db = _tmp_db()
    try:
        pid = ds.open_position(
            position_id="KO_20260623", symbol="KO", stock_purchase_price=60.0,
            shares=100, call_strike=62.5, call_premium=1.20,
            call_expiration="2026-07-28", db_path=db)
        assert pid == "KO_20260623"
        conn = sqlite3.connect(db)
        pos = conn.execute("SELECT symbol, status, stock_purchase_price FROM positions").fetchone()
        txns = conn.execute(
            "SELECT asset_type, action, quantity, price FROM transactions ORDER BY transaction_id"
        ).fetchall()
        conn.close()
        assert pos == ("KO", "OPEN", 60.0)
        # One stock buy leg + one option sell leg.
        assert ("OPTION", "SELL_TO_OPEN", 1, 1.20) in txns
        assert ("STOCK", "BUY_TO_OPEN", 100, 60.0) in txns
    finally:
        os.path.exists(db) and os.unlink(db)


def test_decision_log_links_to_position():
    db = _tmp_db()
    try:
        ds.open_position(position_id="P1", symbol="KO", stock_purchase_price=60.0,
                         shares=100, call_strike=62.5, call_premium=1.2,
                         call_expiration="2026-07-28", db_path=db)
        log_id = ds.log_decision(
            symbol="KO", workflow_stage="ENTRY_SCREENER",
            agent_recommendation={}, agent_rationale="x", is_human_approved=True,
            position_id="P1", db_path=db)
        conn = sqlite3.connect(db)
        pid = conn.execute("SELECT position_id FROM decision_logs WHERE log_id=?", (log_id,)).fetchone()[0]
        conn.close()
        assert pid == "P1"
    finally:
        os.path.exists(db) and os.unlink(db)


def test_cash_effect_signs():
    # Buying 100 shares @ $60 → −$6000; selling 1 call @ $1.20 → +$120.
    assert ds.cash_effect("STOCK", "BUY_TO_OPEN", 100, 60.0) == -6000.0
    assert ds.cash_effect("OPTION", "SELL_TO_OPEN", 1, 1.20) == 120.0
    assert ds.cash_effect("OPTION", "BUY_TO_CLOSE", 1, 0.40) == -40.0
    assert ds.cash_effect("STOCK", "SELL_TO_CLOSE", 100, 62.5) == 6250.0
    assert ds.cash_effect("STOCK", "BUY_TO_OPEN", 100, 60.0, fees=1.0) == -6001.0


def test_open_position_adjusts_cash():
    db = _tmp_db()
    try:
        from app.memory.account_store import get_cash_balance, set_cash_balance
        set_cash_balance(50_000.0, db)
        # Buy 100 @ $60 (−6000), sell 1 call @ $1.20 (+120) → net −5880.
        ds.open_position(position_id="KO_1", symbol="KO", stock_purchase_price=60.0, shares=100,
                         call_strike=62.5, call_premium=1.20, call_expiration="2026-07-28", db_path=db)
        assert get_cash_balance(db) == 50_000.0 - 6000.0 + 120.0
    finally:
        os.path.exists(db) and os.unlink(db)


def test_close_position_assigned_cash_and_pnl():
    db = _tmp_db()
    try:
        from app.memory.account_store import get_cash_balance, set_cash_balance
        set_cash_balance(50_000.0, db)
        ds.open_position(position_id="KO_1", symbol="KO", stock_purchase_price=60.0, shares=100,
                         call_strike=62.5, call_premium=1.20, call_expiration="2026-07-28", db_path=db)
        # Assigned: shares called away at the $62.5 strike (no buyback).
        res = ds.close_position(position_id="KO_1", status="ASSIGNED",
                                stock_sale_price=62.5, db_path=db)
        # Realized P&L = -6000 (buy) +120 (premium) +6250 (sale) = +370.
        assert res["total_realized_pnl"] == 370.0
        # Cash = 50000 -6000 +120 +6250 = 50370.
        assert get_cash_balance(db) == 50_370.0
        assert ds.list_positions("ASSIGNED", db)[0]["position_id"] == "KO_1"
    finally:
        os.path.exists(db) and os.unlink(db)


def test_close_position_liquidated_with_buyback():
    db = _tmp_db()
    try:
        from app.memory.account_store import get_cash_balance, set_cash_balance
        set_cash_balance(50_000.0, db)
        ds.open_position(position_id="X_1", symbol="X", stock_purchase_price=100.0, shares=100,
                         call_strike=105.0, call_premium=2.0, call_expiration="2026-07-28", db_path=db)
        # Hard eject: buy back call @ $0.50 (−50), sell shares @ $90 (+9000).
        res = ds.close_position(position_id="X_1", status="LIQUIDATED",
                                call_buyback_price=0.50, stock_sale_price=90.0, db_path=db)
        # P&L = -10000 +200 -50 +9000 = -850.
        assert res["total_realized_pnl"] == -850.0
        assert get_cash_balance(db) == 50_000.0 - 10_000.0 + 200.0 - 50.0 + 9000.0
    finally:
        os.path.exists(db) and os.unlink(db)


def test_open_position_stores_downside_buffer():
    db = _tmp_db()
    try:
        ds.open_position(position_id="KO_1", symbol="KO", stock_purchase_price=60.0, shares=100,
                         call_strike=62.5, call_premium=1.2, call_expiration="2026-07-28",
                         downside_buffer_percent=2.0, db_path=db)
        pos = ds.list_positions("OPEN", db)[0]
        assert pos["downside_buffer_percent"] == 2.0
    finally:
        os.path.exists(db) and os.unlink(db)


def test_repair_zero_price_assignments():
    db = _tmp_db()
    try:
        from app.memory.account_store import get_cash_balance, set_cash_balance
        set_cash_balance(50_000.0, db)
        ds.open_position(position_id="SOLS_1", symbol="SOLS", stock_purchase_price=86.95, shares=100,
                         call_strike=95.0, call_premium=4.9, call_expiration="2026-07-28", db_path=db)
        # Simulate the bug: close ASSIGNED with a $0 stock sale.
        ds.close_position(position_id="SOLS_1", status="ASSIGNED",
                          stock_sale_price=0.0, call_buyback_price=0.0, db_path=db)
        cash_after_bug = get_cash_balance(db)
        # Repair: should fix the sale price to the 95 strike + credit cash $9,500.
        fixed = ds.repair_zero_price_assignments(db)
        assert fixed[0]["position_id"] == "SOLS_1" and fixed[0]["sale_price"] == 95.0
        assert fixed[0]["total_realized_pnl"] == 1295.0
        assert get_cash_balance(db) == cash_after_bug + 9500.0
        # Idempotent: a second run finds nothing to fix.
        assert ds.repair_zero_price_assignments(db) == []
    finally:
        os.path.exists(db) and os.unlink(db)


def test_roll_position_adjusts_cash_keeps_open():
    db = _tmp_db()
    try:
        from app.memory.account_store import get_cash_balance, set_cash_balance
        set_cash_balance(50_000.0, db)
        ds.open_position(position_id="X_1", symbol="X", stock_purchase_price=100.0, shares=100,
                         call_strike=105.0, call_premium=2.0, call_expiration="2026-07-28", db_path=db)
        # Roll: buy back @0.50 (-50), sell new 95 call @1.50 (+150) → net credit +100.
        res = ds.roll_position(position_id="X_1", call_buyback_price=0.50, new_call_strike=95.0,
                               new_call_premium=1.50, new_call_expiration="2026-08-21", db_path=db)
        assert res["net_credit"] == 100.0 and res["new_strike"] == 95.0
        # Cash = 50000 -10000 +200 -50 +150 = 40300; position stays OPEN.
        assert get_cash_balance(db) == 40_300.0
        assert ds.list_positions("OPEN", db)[0]["position_id"] == "X_1"
        # Realized P&L on the CLOSED option cycle = original premium 2.00 (+200)
        # minus buyback 0.50 (-50) = +150; the new 95 call is open (excluded).
        assert res["total_realized_pnl"] == 150.0
        from app.memory.positions_store import list_holdings_detailed
        h = next(x for x in list_holdings_detailed(db_path=db) if x["position_id"] == "X_1")
        assert h["total_realized_pnl"] == 150.0
    finally:
        os.path.exists(db) and os.unlink(db)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  ✅ {t.__name__}"); passed += 1
        except AssertionError as exc:
            print(f"  ❌ {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  💥 {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    sys.exit(0 if passed == len(tests) else 1)
