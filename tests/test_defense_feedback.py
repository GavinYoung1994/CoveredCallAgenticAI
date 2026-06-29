"""Tests for downside-defense feedback recording (branches A/B/C, deny, prompt)."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory import decision_store as ds
from app.memory.account_store import get_cash_balance, set_cash_balance
from app.feedback import process_defense_feedback, _prompt_defense_decision


class FakeMemory:
    def __init__(self):
        self.lessons = []

    def add_lesson(self, lesson_id, text, metadata):
        self.lessons.append({"id": lesson_id, "metadata": metadata})


def _db_with_open_position():
    fd, db = tempfile.mkstemp(suffix=".db", dir=os.environ.get("TMPDIR")); os.close(fd); os.unlink(db)
    set_cash_balance(50_000.0, db)
    ds.open_position(position_id="X_1", symbol="X", stock_purchase_price=100.0, shares=100,
                     call_strike=105.0, call_premium=2.0, call_expiration="2026-07-28", db_path=db)
    return db


_BA = {"drop_percent": -10.0, "current_stock_price": 90.0, "current_call_ask": 0.5,
       "roll_down_premium": 1.5,
       "branches": {"Branch_A_Liquidate": {"realized_cash_loss": -850.0},
                    "Branch_B_Roll_Down": {"net_credit_received": 100.0, "is_valid": True},
                    "Branch_C_Hold": {"unrealized_net_pnl": -850.0}}}


def _run(branch="B"):
    return {"run_id": "defense_X_1", "workflow": "defense_monitor",
            "recommendations": [{"symbol": "X", "position_id": "X_1", "branch": branch,
                                 "rationale": "test", "sentiment": "NEGATIVE", "branch_analysis": _BA}]}


def _decision_runsdir():
    return tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))


def test_defense_branch_a_liquidates():
    db = _db_with_open_position()
    try:
        mem = FakeMemory()
        decision = {"verdict": "APPROVE", "branch": "A", "notes": "eject",
                    "fill": {"stock_sale_price": 90.0, "call_buyback_price": 0.5, "contracts": 1}}
        out = process_defense_feedback(_run("A"), decision, db_path=db, memory=mem,
                                       runs_dir=_decision_runsdir())
        assert out["action"] == "LIQUIDATED"
        assert ds.list_positions("LIQUIDATED", db)[0]["position_id"] == "X_1"
        # Cash = 50000 -10000 +200 -50 +9000 = 49150.
        assert get_cash_balance(db) == 49_150.0
        # Decision logged at the DEFENSE_MONITOR stage; lesson stored.
        conn = sqlite3.connect(db)
        stage, approved = conn.execute(
            "SELECT workflow_stage, is_human_approved FROM decision_logs").fetchone()
        conn.close()
        assert stage == "DEFENSE_MONITOR" and approved == 1
        assert mem.lessons[0]["metadata"]["branch"] == "A"
    finally:
        os.path.exists(db) and os.unlink(db)


def test_defense_branch_b_rolls():
    db = _db_with_open_position()
    try:
        decision = {"verdict": "APPROVE", "branch": "B", "notes": "roll for credit",
                    "fill": {"call_buyback_price": 0.5, "new_call_strike": 95.0,
                             "new_call_premium": 1.5, "new_call_expiration": "2026-08-21", "contracts": 1}}
        out = process_defense_feedback(_run("B"), decision, db_path=db, runs_dir=_decision_runsdir())
        assert out["action"] == "ROLLED" and out["net_credit"] == 100.0
        assert ds.list_positions("OPEN", db)[0]["position_id"] == "X_1"   # stays open
        assert get_cash_balance(db) == 40_300.0
    finally:
        os.path.exists(db) and os.unlink(db)


def test_defense_branch_c_hold_no_trade():
    db = _db_with_open_position()
    try:
        cash_before = get_cash_balance(db)
        out = process_defense_feedback(_run("C"), {"verdict": "APPROVE", "branch": "C", "notes": "wait"},
                                       db_path=db, runs_dir=_decision_runsdir())
        assert out["action"] == "HELD (no trade)"
        assert get_cash_balance(db) == cash_before          # no cash effect
        assert ds.list_positions("OPEN", db)[0]["position_id"] == "X_1"
    finally:
        os.path.exists(db) and os.unlink(db)


def test_defense_deny_logs_no_trade():
    db = _db_with_open_position()
    try:
        cash_before = get_cash_balance(db)
        out = process_defense_feedback(_run("A"), {"verdict": "DENY", "notes": "I'll handle manually"},
                                       db_path=db, runs_dir=_decision_runsdir())
        assert out["action"] == "denied (no trade)"
        assert get_cash_balance(db) == cash_before
        conn = sqlite3.connect(db)
        approved = conn.execute("SELECT is_human_approved FROM decision_logs").fetchone()[0]
        conn.close()
        assert approved == 0
    finally:
        os.path.exists(db) and os.unlink(db)


def test_defense_prompt_collector_scripted():
    rec = _run("B")["recommendations"][0]
    inputs = iter(["B", "rolling", "0.5", "95", "1.5", "2026-08-21", "1"])
    decision = _prompt_defense_decision(rec, input_func=lambda _p: next(inputs), output_func=lambda _m: None)
    assert decision["verdict"] == "APPROVE" and decision["branch"] == "B"
    assert decision["fill"]["new_call_strike"] == 95.0
    assert decision["fill"]["new_call_expiration"] == "2026-08-21"


def test_defense_prompt_default_branch_and_skip():
    rec = _run("C")["recommendations"][0]
    # Empty input → default to recommended branch C (Hold, no fill prompts).
    decision = _prompt_defense_decision(rec, input_func=lambda _p: "" if "Execute" in _p else "ok",
                                        output_func=lambda _m: None)
    assert decision["verdict"] == "APPROVE" and decision["branch"] == "C" and "fill" not in decision


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
