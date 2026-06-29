"""Tests for the performance reporting loop (temp DB, fake LLM/memory)."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory import decision_store as ds
from app.reporting import gather_performance, generate_report


class FakeMemory:
    def __init__(self):
        self.lessons = []

    def add_lesson(self, lesson_id, text, metadata):
        self.lessons.append({"id": lesson_id, "metadata": metadata})


def _seed_db():
    fd, db = tempfile.mkstemp(suffix=".db", dir=os.environ.get("TMPDIR")); os.close(fd); os.unlink(db)
    # 3 positions: 1 open, 1 closed-win (+300), 1 closed-loss (-150).
    ds.open_position(position_id="P1", symbol="KO", stock_purchase_price=60.0, shares=100,
                     call_strike=62.5, call_premium=1.0, call_expiration="2026-07-28", db_path=db)
    ds.open_position(position_id="P2", symbol="XOM", stock_purchase_price=110.0, shares=100,
                     call_strike=115.0, call_premium=2.0, call_expiration="2026-07-28", db_path=db)
    ds.open_position(position_id="P3", symbol="ABC", stock_purchase_price=50.0, shares=100,
                     call_strike=52.5, call_premium=1.5, call_expiration="2026-07-28", db_path=db)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE positions SET status='ASSIGNED', total_realized_pnl=300 WHERE position_id='P2'")
    conn.execute("UPDATE positions SET status='LIQUIDATED', total_realized_pnl=-150 WHERE position_id='P3'")
    conn.commit(); conn.close()
    # 3 decisions: 2 approved, 1 denied (with notes).
    ds.log_decision(symbol="KO", workflow_stage="ENTRY_SCREENER", agent_recommendation={},
                    agent_rationale="x", is_human_approved=True, position_id="P1", db_path=db)
    ds.log_decision(symbol="XOM", workflow_stage="ENTRY_SCREENER", agent_recommendation={},
                    agent_rationale="x", is_human_approved=True, position_id="P2", db_path=db)
    ds.log_decision(symbol="ZZZ", workflow_stage="ENTRY_SCREENER", agent_recommendation={},
                    agent_rationale="x", is_human_approved=False,
                    human_feedback_notes="Earnings too close.", db_path=db)
    return db


def test_gather_performance_stats():
    db = _seed_db()
    try:
        s = gather_performance(db)
        assert s["decisions_total"] == 3 and s["approved"] == 2 and s["denied"] == 1
        assert s["open_positions"] == 1 and s["closed_positions"] == 2
        assert s["realized_pnl"] == 150.0          # 300 - 150
        assert s["wins"] == 1 and s["win_rate_percent"] == 50.0
        # Premium harvested = (1.0 + 2.0 + 1.5) * 100 = 450.
        assert s["total_premium_harvested"] == 450.0
        assert s["worst_losers"][0]["symbol"] == "ABC"
        assert s["recent_denials"][0]["symbol"] == "ZZZ"
    finally:
        os.path.exists(db) and os.unlink(db)


def test_annualized_return_on_invested_cash():
    from app.memory.account_store import set_cash_balance
    fd, db = tempfile.mkstemp(suffix=".db", dir=os.environ.get("TMPDIR")); os.close(fd); os.unlink(db)
    try:
        set_cash_balance(50_000.0, db)
        ds.open_position(position_id="KO_1", symbol="KO", stock_purchase_price=60.0, shares=100,
                         call_strike=62.5, call_premium=1.20, call_expiration="2026-07-28", db_path=db)
        ds.close_position(position_id="KO_1", status="ASSIGNED", stock_sale_price=62.5, db_path=db)
        # Force a 30-day holding period for a deterministic annualization.
        conn = sqlite3.connect(db)
        conn.execute("UPDATE positions SET entry_date = datetime(close_date, '-30 days') WHERE position_id='KO_1'")
        conn.commit(); conn.close()

        s = gather_performance(db)
        assert s["invested_capital_closed"] == 6000.0
        assert s["realized_pnl"] == 370.0
        # ROIC = 370/6000 = 6.17%; annualized = 6.17 * 365/30 ≈ 75.0%.
        assert abs(s["return_on_invested_capital_percent"] - 6.17) < 0.05
        assert s["avg_holding_days"] == 30.0
        assert abs(s["annualized_return_percent"] - 75.03) < 0.5
    finally:
        os.path.exists(db) and os.unlink(db)


def test_generate_report_with_llm_and_memory():
    db = _seed_db()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as rd:
        try:
            class FakeLLM:
                def chat(self, system, user, **kw):
                    return "Solid premium capture; one loss from a sector drop. Tighten the trend filter."
            mem = FakeMemory()
            out = generate_report(db_path=db, llm=FakeLLM(), memory=mem,
                                  period_label="weekly", asof="2026-06-24", runs_dir=rd)
            assert "Performance Report" in out["markdown"]
            assert "Analyst review" in out["markdown"]
            assert "Tighten the trend filter" in out["markdown"]
            assert out["stats"]["realized_pnl"] == 150.0
            # Lesson stored in vector memory for future recall.
            assert mem.lessons and mem.lessons[0]["metadata"]["type"] == "performance_report"
            # Report markdown persisted to the runs dir.
            assert (Path(rd) / "report_weekly_2026-06-24.md").exists()
        finally:
            os.path.exists(db) and os.unlink(db)


def test_generate_report_no_llm_no_memory():
    db = _seed_db()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as rd:
        try:
            out = generate_report(db_path=db, period_label="monthly", asof="2026-06-24", runs_dir=rd)
            assert out["narrative"] == "" and "Analyst review" not in out["markdown"]
            assert "Performance Report" in out["markdown"]
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
