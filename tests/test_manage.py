"""Tests for the NL DB management interface (service ops + LLM routing)."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import LocalLLM
from app.memory import decision_store as ds
from app.manage import ManagementService, _Router, handle


def _db():
    fd, p = tempfile.mkstemp(suffix=".db", dir=os.environ.get("TMPDIR")); os.close(fd); os.unlink(p)
    return p


class FakeMemory:
    def query(self, text, n_results=5):
        return [{"id": "L1", "document": f"lesson about {text}", "metadata": {"symbol": "KO"}}]


# ── service-layer (deterministic) ─────────────────────────────────────
def test_service_cash_get_set():
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        assert svc.get_cash()["cash_balance"] == 0.0
        assert svc.set_cash(50_000)["cash_balance"] == 50_000.0
        assert svc.get_cash()["cash_balance"] == 50_000.0
    finally:
        os.path.exists(db) and os.unlink(db)


def test_service_list_holdings_and_close():
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        svc.set_cash(50_000)
        ds.open_position(position_id="KO_1", symbol="KO", stock_purchase_price=60.0, shares=100,
                         call_strike=62.5, call_premium=1.2, call_expiration="2026-07-28", db_path=db)
        assert [p["position_id"] for p in svc.list_holdings("OPEN")["positions"]] == ["KO_1"]
        # Close with a sale price → realized P&L computed, cash adjusted.
        res = svc.update_holding_status(status="assigned", position_id="KO_1", stock_sale_price=62.5)
        assert res["total_realized_pnl"] == 370.0
        assert svc.list_holdings("OPEN")["positions"] == []
        assert svc.get_cash()["cash_balance"] == 50_370.0


    finally:
        os.path.exists(db) and os.unlink(db)


def test_update_holding_by_symbol_assigned_defaults_to_strike():
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        svc.set_cash(50_000)
        ds.open_position(position_id="SOLS_1", symbol="SOLS", stock_purchase_price=80.0, shares=100,
                         call_strike=82.0, call_premium=1.5, call_expiration="2026-07-28", db_path=db)
        # By SYMBOL, ASSIGNED, no fills → sale defaults to the 82 strike, buyback 0.
        res = svc.update_holding_status(status="ASSIGNED", symbol="SOLS")
        # Realized = -8000 (buy) +150 (premium) +8200 (called away at 82) = +350.
        assert res["total_realized_pnl"] == 350.0
        assert svc.list_holdings("ASSIGNED")["positions"][0]["position_id"] == "SOLS_1"
    finally:
        os.path.exists(db) and os.unlink(db)


def test_update_holding_unknown_symbol():
    svc = ManagementService(db_path=_db())
    assert "error" in svc.update_holding_status(status="ASSIGNED", symbol="NOPE")


def test_assigned_zero_sale_price_falls_back_to_strike():
    # Regression: the LLM passing stock_sale_price=0 must NOT sell shares for $0 —
    # ASSIGNED always defaults a 0/missing price to the strike.
    from app.memory.account_store import get_cash_balance
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        svc.set_cash(50_000)
        ds.open_position(position_id="SOLS_1", symbol="SOLS", stock_purchase_price=86.95, shares=100,
                         call_strike=95.0, call_premium=4.9, call_expiration="2026-07-28", db_path=db)
        res = svc.update_holding_status(status="ASSIGNED", symbol="SOLS",
                                        stock_sale_price=0, call_buyback_price=0)
        # Realized = -8695 + 490 (premium) + 9500 (sold at the 95 strike) = +1295.
        assert res["total_realized_pnl"] == 1295.0
        assert get_cash_balance(db) == 50_000 - 8695 + 490 + 9500    # = 51,295
    finally:
        os.path.exists(db) and os.unlink(db)


def test_service_status_relabel_without_prices():
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        svc.set_cash(10_000)
        ds.open_position(position_id="X_1", symbol="X", stock_purchase_price=50.0, shares=100,
                         call_strike=52.5, call_premium=1.0, call_expiration="2026-07-28", db_path=db)
        cash_before = svc.get_cash()["cash_balance"]
        res = svc.update_holding_status(status="ON_HOLD", position_id="X_1")  # no prices → pure relabel
        assert res["updated"] is True
        assert svc.get_cash()["cash_balance"] == cash_before   # no cash effect
    finally:
        os.path.exists(db) and os.unlink(db)


def test_service_search_learnings():
    svc = ManagementService(db_path=_db(), memory=FakeMemory())
    out = svc.search_learnings("utilities", n=3)
    assert out["lessons"][0]["id"] == "L1"


# ── routing + handle (LLM mapped to an action) ────────────────────────
def _router(json_response):
    return _Router(LocalLLM(backend=lambda msgs: json_response))


def test_handle_routes_set_cash():
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        router = _router('{"action": "set_cash", "args": {"amount": 25000}}')
        out = handle("put my cash at 25k", svc, router)
        assert out["cash_balance"] == 25_000.0
    finally:
        os.path.exists(db) and os.unlink(db)


def test_handle_routes_list_holdings():
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        svc.set_cash(50_000)
        ds.open_position(position_id="KO_1", symbol="KO", stock_purchase_price=60.0, shares=100,
                         call_strike=62.5, call_premium=1.2, call_expiration="2026-07-28", db_path=db)
        router = _router('{"action": "list_holdings", "args": {"status": "OPEN"}}')
        out = handle("show open positions", svc, router)
        assert out["positions"][0]["symbol"] == "KO"
    finally:
        os.path.exists(db) and os.unlink(db)


def test_handle_unknown_action():
    svc = ManagementService(db_path=_db())
    out = handle("do something weird", svc, _router('{"action": "none", "args": {}}'))
    assert "error" in out and "available" in out


def test_handle_ignores_extra_args():
    # LLM hallucinates an extra arg → filtered out, call still succeeds.
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        router = _router('{"action": "get_cash", "args": {"bogus": 1, "extra": "x"}}')
        out = handle("cash?", svc, router)
        assert out["cash_balance"] == 0.0
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
