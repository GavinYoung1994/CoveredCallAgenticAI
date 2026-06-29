"""Tests for the Flask management UI backend (fake agent + sync runners)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.web.server import create_app, LogBuffer, Jobs, launch_workflow


class FakeAgent:
    def chat(self, message, history=None):
        return {"answer": f"echo: {message}", "steps": [{"tool": "get_cash", "args": {}}]}


def _report_runner():
    logging.getLogger("web").info("fake report ran")
    return {"stats": {"realized_pnl": 150.0, "annualized_return_percent": 75.0}}


def _app(buffer=None):
    return create_app(
        agent_provider=lambda: FakeAgent(),
        runners={"report": _report_runner, "screener": lambda: {"recommendations": [1], "rejected": []}},
        log_buffer=buffer or LogBuffer(),
        run_async=False,             # run jobs inline for deterministic tests
    ).test_client()


def test_index_served():
    r = _app().get("/")
    assert r.status_code == 200 and b"Command Center" in r.data
    assert b"react" in r.data.lower()              # React app


def test_workflow_window_served():
    r = _app().get("/workflow")
    assert r.status_code == 200 and b"WorkflowWindow" in r.data


def test_holdings_window_served():
    r = _app().get("/holdings")
    assert r.status_code == 200 and b"My Holdings" in r.data


def test_holdings_api_returns_positions():
    provider = lambda: {"cash_balance": 50_000.0, "positions": [
        {"position_id": "KO_1", "symbol": "KO", "status": "OPEN", "shares": 100,
         "stock_purchase_price": 60.0, "short_call_strike": 62.5,
         "short_call_expiration": "2026-07-28", "short_call_premium": 1.2,
         "downside_buffer_percent": 2.0, "total_realized_pnl": 0.0}]}
    client = create_app(agent_provider=lambda: FakeAgent(), runners={},
                        holdings_provider=provider, log_buffer=LogBuffer(), run_async=False).test_client()
    d = client.get("/api/holdings").get_json()
    assert d["cash_balance"] == 50_000.0
    assert d["positions"][0]["symbol"] == "KO" and d["positions"][0]["short_call_strike"] == 62.5


def test_static_css_served():
    r = _app().get("/static/styles.css")
    assert r.status_code == 200
    assert b'data-theme="emerald"' in r.data or b'--accent' in r.data


def test_chat_returns_answer():
    r = _app().post("/api/chat", json={"message": "hello"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["answer"] == "echo: hello"
    assert body["steps"][0]["tool"] == "get_cash"


def test_chat_empty_message_400():
    r = _app().post("/api/chat", json={"message": "   "})
    assert r.status_code == 400


def test_chat_passes_history_to_agent():
    captured = {}

    class RecordingAgent:
        def chat(self, message, history=None):
            captured["history"] = history
            return {"answer": "ok", "steps": []}

    client = create_app(agent_provider=lambda: RecordingAgent(),
                        runners={}, log_buffer=LogBuffer(), run_async=False).test_client()
    client.post("/api/chat", json={"message": "follow up",
                                   "history": [{"role": "user", "content": "earlier question"}]})
    assert captured["history"] == [{"role": "user", "content": "earlier question"}]


def test_setup_logging_preserves_existing_handlers():
    # The web LogBuffer must survive a later setup_logging() call (workflows call
    # it) — previously basicConfig(force=True) wiped it, so UI logs vanished.
    import logging
    from app.logging_config import setup_logging
    buf = LogBuffer()
    root = logging.getLogger()
    root.addHandler(buf)
    try:
        setup_logging("INFO")
        assert buf in root.handlers
    finally:
        root.removeHandler(buf)


def test_run_workflow_and_status():
    client = _app()
    r = client.post("/api/run/report")
    assert r.status_code == 200 and r.get_json()["started"] == "report"
    # run_async=False → already finished; status reflects the last result.
    st = client.get("/api/status").get_json()
    assert st["running"] is None
    assert st["last"]["report"]["ok"] is True
    assert "annualized 75.0%" in st["last"]["report"]["summary"]


def test_unknown_workflow_404():
    r = _app().post("/api/run/teleport")
    assert r.status_code == 404


def test_logs_capture_workflow_output():
    buf = LogBuffer()
    client = _app(buffer=buf)
    client.post("/api/run/report")          # logs "▶ Starting..." + "fake report ran" + "✔ ..."
    data = client.get("/api/logs?since=0").get_json()
    msgs = " ".join(r["msg"] for r in data["records"])
    assert "fake report ran" in msgs
    assert data["cursor"] >= 1
    # Cursor advances: nothing new since the latest cursor.
    again = client.get(f"/api/logs?since={data['cursor']}").get_json()
    assert again["records"] == []


def test_launch_workflow_starts_in_background():
    jobs = Jobs(run_async=False)          # inline → completes immediately
    ran = []
    runners = {"report": lambda: (ran.append(1), {"stats": {}})[1]}
    out = launch_workflow(jobs, runners, "report")
    assert out["status"] == "started" and out["workflow"] == "report" and ran == [1]
    assert "Performance Report" in out["message"]


def test_launch_workflow_unknown():
    out = launch_workflow(Jobs(run_async=False), {"report": lambda: {}}, "teleport")
    assert "error" in out


def test_logbuffer_ignores_noisy_loggers():
    buf = LogBuffer()
    logging.getLogger("werkzeug").info('GET /api/logs?since=31 HTTP/1.1" 200 -')
    logging.getLogger("httpx").info("HTTP Request: GET ...")
    logging.getLogger("node.quant").info("Quant PASS KO")
    # Only the app logger's record is kept; HTTP/access noise is dropped.
    recs = buf.since(0)["records"]
    # buf isn't attached to root here, so emit it directly to test the filter:
    buf.emit(logging.LogRecord("werkzeug", logging.INFO, "", 0, 'GET /api/logs 200', None, None))
    buf.emit(logging.LogRecord("node.quant", logging.INFO, "", 0, "Quant PASS KO", None, None))
    msgs = [r["msg"] for r in buf.since(0)["records"]]
    names = [r["name"] for r in buf.since(0)["records"]]
    assert "werkzeug" not in names
    assert "Quant PASS KO" in msgs


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
