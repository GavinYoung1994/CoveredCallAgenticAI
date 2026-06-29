"""Tests for the tool registry and the multi-step CoveredCallAgent."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import LocalLLM
from app.manage import ManagementService
from app.memory import decision_store as ds
from app.agent.tools import build_tools, tool_catalog_text
from app.agent.agent import CoveredCallAgent


def _db():
    fd, p = tempfile.mkstemp(suffix=".db", dir=os.environ.get("TMPDIR")); os.close(fd); os.unlink(p)
    return p


class FakeMemory:
    def query(self, text, n_results=5):
        return [{"id": "L1", "document": f"lesson:{text}", "metadata": {}}]


def _scripted_llm(responses):
    """A fake LLM whose .complete returns each scripted string in turn."""
    state = {"i": 0}

    def backend(msgs):
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return r

    return LocalLLM(backend=backend)


# ── tool registry ─────────────────────────────────────────────────────
def test_build_tools_and_catalog():
    svc = ManagementService(db_path=_db())
    tools = build_tools(svc)
    # Management + full math engine + live data + composites are all present.
    assert {"get_cash", "analyze_covered_call", "defense_branches", "performance_report",
            "black_scholes_greeks", "iv_rank", "macro_divergence", "position_size",  # math
            "get_quote", "get_option_chain", "get_news", "get_next_earnings_date",     # live data
            "technical_analysis", "find_best_covered_call"} <= set(tools)              # composites
    assert len(tools) >= 30
    catalog = tool_catalog_text(tools)
    assert "get_quote" in catalog and "technical_analysis" in catalog


def test_build_tools_without_data():
    tools = build_tools(ManagementService(db_path=_db()), include_data=False, include_workflows=False)
    assert "get_quote" not in tools and "run_stock_screener" not in tools
    assert "analyze_covered_call" in tools


def test_option_window_sanitizer():
    from datetime import date, timedelta
    from app.agent.tools import _sane_option_window
    today = date.today()
    # Past dates (LLM hallucination) → coerced into a future window.
    fd, td = _sane_option_window("2023-08-01", "2023-10-31")
    assert fd > today.isoformat() and td > fd
    # Missing dates → default 30–45 day window.
    fd2, td2 = _sane_option_window(None, None)
    assert fd2 >= today.isoformat() and td2 > fd2
    # Valid future window is preserved as-is.
    f, t = (today + timedelta(days=35)).isoformat(), (today + timedelta(days=50)).isoformat()
    assert _sane_option_window(f, t) == (f, t)


def test_workflow_tools_registered():
    tools = build_tools(ManagementService(db_path=_db()))
    assert {"run_stock_screener", "run_defense_scan", "run_performance_review"} <= set(tools)


def test_workflow_tools_use_launcher_nonblocking():
    # With a launcher, the workflow tools delegate (background) instead of running
    # the heavy pipeline synchronously.
    calls = []
    launcher = lambda name: (calls.append(name) or {"status": "started", "workflow": name})
    tools = build_tools(ManagementService(db_path=_db()), workflow_launcher=launcher)
    assert tools["run_stock_screener"].run()["workflow"] == "screener"
    assert tools["run_defense_scan"].run()["workflow"] == "defense"
    assert tools["run_performance_review"].run(period="monthly")["workflow"] == "report"
    assert calls == ["screener", "defense", "report"]


def test_run_stock_screener_tool_summarizes():
    import app.graphs as graphs
    fake_final = {"run_id": "r1", "recommendations": [
        {"symbol": "KO", "grade": "A", "score": 80, "annualized_yield_percent": 15.0}],
        "rejected": [1, 2, 3], "notified": True}
    orig = graphs.run_entry_screener
    graphs.run_entry_screener = lambda **kw: fake_final   # handler does `from app.graphs import ...`
    try:
        tools = build_tools(ManagementService(db_path=_db()))
        out = tools["run_stock_screener"].run()
        assert out["recommendation_count"] == 1 and out["rejected_count"] == 3
        assert out["recommendations"][0]["symbol"] == "KO" and out["notified"] is True
    finally:
        graphs.run_entry_screener = orig


def test_analyze_covered_call_tool_math():
    tools = build_tools(ManagementService(db_path=_db()))
    out = tools["analyze_covered_call"].run(
        underlying_price=100, strike=105, premium=2.0, days_to_expiration=30, volatility=0.25)
    assert out["yield_metrics"]["downside_breakeven_price"] == 98.0
    assert out["greeks"]["prob_assignment_percent"] > 0
    assert "composite_score" in out


def test_get_cash_tool():
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        svc.set_cash(40_000)
        assert build_tools(svc)["get_cash"].run()["cash_balance"] == 40_000.0
    finally:
        os.path.exists(db) and os.unlink(db)


# ── agent loop ─────────────────────────────────────────────────────────
def test_agent_single_tool_then_final():
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        svc.set_cash(33_000)
        tools = build_tools(svc)
        # Step 1: call get_cash. Step 2: final answer.
        llm = _scripted_llm([
            '{"thought": "check cash", "tool": "get_cash", "args": {}}',
            '{"thought": "done", "final": "Your cash balance is $33,000."}',
        ])
        agent = CoveredCallAgent(llm, tools)
        out = agent.chat("how much cash do I have?")
        assert out["answer"] == "Your cash balance is $33,000."
        assert out["steps"][0]["tool"] == "get_cash"
        assert out["steps"][0]["observation"]["cash_balance"] == 33_000.0
    finally:
        os.path.exists(db) and os.unlink(db)


def test_agent_chains_two_tools():
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        tools = build_tools(svc)
        llm = _scripted_llm([
            '{"tool": "set_cash", "args": {"amount": 25000}}',
            '{"tool": "get_cash", "args": {}}',
            '{"final": "Set and confirmed: $25,000."}',
        ])
        out = CoveredCallAgent(llm, tools).chat("set my cash to 25k and confirm")
        assert [s["tool"] for s in out["steps"]] == ["set_cash", "get_cash"]
        assert out["steps"][1]["observation"]["cash_balance"] == 25_000.0
        assert "25,000" in out["answer"]
    finally:
        os.path.exists(db) and os.unlink(db)


def test_agent_unknown_tool_recovers():
    db = _db()
    try:
        tools = build_tools(ManagementService(db_path=db))
        llm = _scripted_llm([
            '{"tool": "teleport", "args": {}}',           # bogus
            '{"final": "Sorry, I cannot do that."}',
        ])
        out = CoveredCallAgent(llm, tools).chat("teleport me")
        assert out["steps"][0]["observation"]["error"].startswith("unknown tool")
        assert "cannot" in out["answer"].lower()
    finally:
        os.path.exists(db) and os.unlink(db)


def test_agent_guard_blocks_hallucinated_change():
    # The agent first FAKES success (no tool). The guard must re-prompt, after
    # which it actually calls update_holding_status — and the DB is updated.
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        svc.set_cash(50_000)
        ds.open_position(position_id="P_1", symbol="P", stock_purchase_price=69.5, shares=100,
                         call_strike=72.5, call_premium=1.0, call_expiration="2026-07-28", db_path=db)
        tools = build_tools(svc)
        llm = _scripted_llm([
            '{"final": "The position for P has been recorded as assigned. Cash balance is now $70000."}',
            '{"tool": "update_holding_status", "args": {"status": "ASSIGNED", "symbol": "P"}}',
            '{"final": "Recorded P as assigned at its strike."}',
        ])
        out = CoveredCallAgent(llm, tools).chat("P got executed, please record the trade")
        # Guard fired → the mutating tool actually ran, and the DB reflects it.
        assert any(s["tool"] == "update_holding_status" for s in out["steps"])
        assert svc.list_holdings("ASSIGNED")["positions"][0]["symbol"] == "P"
    finally:
        os.path.exists(db) and os.unlink(db)


def test_agent_filters_bad_args():
    db = _db()
    try:
        svc = ManagementService(db_path=db)
        svc.set_cash(10_000)
        tools = build_tools(svc)
        llm = _scripted_llm([
            '{"tool": "get_cash", "args": {"bogus": 1}}',   # extra arg ignored
            '{"final": "$10,000."}',
        ])
        out = CoveredCallAgent(llm, tools).chat("cash?")
        assert out["steps"][0]["observation"]["cash_balance"] == 10_000.0
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
