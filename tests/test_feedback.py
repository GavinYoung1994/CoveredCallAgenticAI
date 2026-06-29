"""Tests for the human-feedback recording logic (process_feedback).

Temp SQLite DB + a fake TradeMemory; verifies that approvals open positions and
log decisions, denials log without a position, skips do nothing, and lessons are
stored. The interactive prompt collector is tested with scripted inputs.
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.feedback import process_feedback, _prompt_decisions


class FakeMemory:
    def __init__(self):
        self.lessons = []

    def add_lesson(self, lesson_id, text, metadata):
        self.lessons.append({"id": lesson_id, "text": text, "metadata": metadata})


def _rec(sym, strike=105.0, mark=2.05, yld=20.0):
    return {"symbol": sym, "grade": "A", "score": 80.0, "annualized_yield_percent": yld,
            "underlying_price": 100.0, "sentiment": "POSITIVE", "rationale": "Good trade.",
            "contract": {"symbol": f"{sym}_C", "strike": strike, "mark": mark,
                         "expiration_key": "2026-07-28:35", "days_to_expiration": 35}}


def _run_data(recs):
    return {"run_id": "run-x", "status": "PENDING_APPROVAL", "recommendations": recs}


def _tmp():
    fd, p = tempfile.mkstemp(suffix=".db", dir=os.environ.get("TMPDIR"))
    os.close(fd); os.unlink(p)
    return p


def test_approve_opens_position_and_logs():
    db = _tmp()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as rd:
        mem = FakeMemory()
        run = _run_data([_rec("KO")])
        decisions = [{"verdict": "APPROVE", "notes": "Looks great.",
                      "fill": {"shares": 100, "stock_price": 60.0, "premium": 1.25, "contracts": 1}}]
        summary = process_feedback(run, decisions, db_path=db, memory=mem, runs_dir=rd)
        assert summary["approved"] == ["KO"] and len(summary["log_ids"]) == 1

        conn = sqlite3.connect(db)
        pos = conn.execute("SELECT position_id, symbol, status, stock_purchase_price FROM positions").fetchone()
        dec = conn.execute("SELECT symbol, is_human_approved, human_feedback_notes, position_id FROM decision_logs").fetchone()
        ntx = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        conn.close()
        assert pos == ("KO_run-x", "KO", "OPEN", 60.0)
        assert dec == ("KO", 1, "Looks great.", "KO_run-x")
        assert ntx == 2                                  # stock + option legs
        assert mem.lessons[0]["metadata"]["verdict"] == "APPROVE"


def test_deny_logs_without_position():
    db = _tmp()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as rd:
        mem = FakeMemory()
        summary = process_feedback(_run_data([_rec("XOM")]),
                                   [{"verdict": "DENY", "notes": "Don't like the chart."}],
                                   db_path=db, memory=mem, runs_dir=rd)
        assert summary["denied"] == ["XOM"]
        conn = sqlite3.connect(db)
        npos = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        approved = conn.execute("SELECT is_human_approved FROM decision_logs").fetchone()[0]
        conn.close()
        assert npos == 0 and approved == 0


def test_skip_records_nothing():
    db = _tmp()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as rd:
        summary = process_feedback(_run_data([_rec("PFE")]),
                                   [{"verdict": "SKIP", "notes": ""}], db_path=db, runs_dir=rd)
        assert summary["skipped"] == ["PFE"] and summary["log_ids"] == []


def test_mixed_batch_and_run_marked_reviewed():
    db = _tmp()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as rd:
        run = _run_data([_rec("A"), _rec("B"), _rec("C")])
        decisions = [
            {"verdict": "APPROVE", "notes": "yes", "fill": {"shares": 100, "stock_price": 100.0, "premium": 2.0, "contracts": 1}},
            {"verdict": "DENY", "notes": "no"},
            {"verdict": "SKIP", "notes": ""},
        ]
        summary = process_feedback(run, decisions, db_path=db, runs_dir=rd)
        assert summary["approved"] == ["A"] and summary["denied"] == ["B"] and summary["skipped"] == ["C"]
        # The run file is rewritten with REVIEWED status.
        import json
        saved = json.loads((Path(rd) / "run-x.json").read_text())
        assert saved["status"] == "REVIEWED"


def test_mismatched_decisions_raises():
    try:
        process_feedback(_run_data([_rec("A")]), [], db_path=_tmp())
        assert False, "should have raised"
    except ValueError:
        pass


def test_prompt_collector_scripted():
    # Drive the interactive collector with scripted inputs (approve then deny).
    recs = [_rec("AAA"), _rec("BBB")]
    inputs = iter([
        "a", "great setup", "100", "99.5", "2.10", "1",   # approve AAA + fill
        "d", "earnings risk",                              # deny BBB
    ])
    decisions = _prompt_decisions(recs, input_func=lambda _p: next(inputs), output_func=lambda _m: None)
    assert decisions[0]["verdict"] == "APPROVE"
    assert decisions[0]["fill"] == {"shares": 100, "stock_price": 99.5, "premium": 2.10, "contracts": 1}
    assert decisions[1]["verdict"] == "DENY" and decisions[1]["notes"] == "earnings risk"


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
