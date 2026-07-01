"""Flask backend for the management UI.

Endpoints
---------
GET  /                 → the single-page UI
POST /api/chat         → {message} → agent answer (+ tool steps)
POST /api/run/<name>   → start a workflow in the background (screener/defense/report)
GET  /api/status       → which workflow (if any) is running + last results
GET  /api/logs?since=N → new log records since cursor N (for the live console)

Design: dependencies are injected via ``create_app`` so tests pass a fake agent
and fake runners (no LLM, no live APIs) and can run jobs synchronously.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from flask import Flask, jsonify, request, send_from_directory

logger = logging.getLogger("web")
_STATIC = Path(__file__).resolve().parent / "static"


# ── live log capture ──────────────────────────────────────────────────
# Loggers that are pure noise for the UI console (HTTP servers/clients). The UI
# polls a few endpoints every second, so Werkzeug's access log would flood it.
_IGNORED_LOGGERS = ("werkzeug", "httpx", "httpcore", "urllib3", "hpack")


class LogBuffer(logging.Handler):
    """A logging handler that keeps the most recent records for the UI console,
    skipping noisy infrastructure loggers (e.g. Werkzeug request logs)."""

    def __init__(self, capacity: int = 800, ignore: tuple = _IGNORED_LOGGERS) -> None:
        super().__init__()
        self._records: deque = deque(maxlen=capacity)
        self._seq = 0
        self._lock = threading.Lock()
        self._ignore = ignore

    def emit(self, record: logging.LogRecord) -> None:
        if any(record.name == n or record.name.startswith(n + ".") for n in self._ignore):
            return
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            msg = str(record.msg)
        with self._lock:
            self._seq += 1
            self._records.append({
                "id": self._seq,
                "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "name": record.name,
                "msg": msg,
            })

    def since(self, cursor: int) -> Dict[str, Any]:
        with self._lock:
            new = [r for r in self._records if r["id"] > cursor]
            last = self._records[-1]["id"] if self._records else cursor
        return {"records": new, "cursor": last}


# ── background workflow jobs ───────────────────────────────────────────
class Jobs:
    """Runs one workflow at a time in a background thread (or inline in tests)."""

    def __init__(self, run_async: bool = True) -> None:
        self.run_async = run_async
        self._running: Optional[str] = None
        self._last: Dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def running(self) -> Optional[str]:
        return self._running

    @property
    def last(self) -> Dict[str, Any]:
        return dict(self._last)

    def start(self, name: str, fn: Callable[[], Any]) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = name

        def task() -> None:
            logger.info("▶ Starting workflow: %s", name)
            try:
                result = fn()
                self._last[name] = {"ok": True, "summary": _summarize(name, result)}
                logger.info("✔ Workflow %s finished.", name)
            except Exception as exc:  # noqa: BLE001
                self._last[name] = {"ok": False, "error": str(exc)}
                logger.exception("✖ Workflow %s failed", name)
            finally:
                with self._lock:
                    self._running = None

        if self.run_async:
            threading.Thread(target=task, daemon=True).start()
        else:
            task()
        return True


def _summarize(name: str, result: Any) -> str:
    """A short human summary of a workflow result for the status panel."""
    if not isinstance(result, dict):
        return str(result)[:200]
    if name == "screener":
        return f"{len(result.get('recommendations', []))} recommendations, " \
               f"{len(result.get('rejected', []))} rejected."
    if name == "defense":
        return f"scanned {result.get('scanned', 0)}, breached {len(result.get('breached', []))}."
    if name == "report":
        s = result.get("stats", {})
        return f"P&L ${s.get('realized_pnl', 0):,.0f}, " \
               f"annualized {s.get('annualized_return_percent', 'n/a')}%."
    return "done."


# ── default (production) dependency builders ───────────────────────────
def _default_runners() -> Dict[str, Callable[[], Any]]:
    def screener() -> Any:
        from app.graphs import run_entry_screener
        return run_entry_screener()

    def defense() -> Any:
        from app.graphs import run_defense_scan
        return run_defense_scan()

    def report() -> Any:
        from app.reporting import generate_report
        from app.llm import get_llm
        from app.memory.vector_db import TradeMemory
        return generate_report(llm=get_llm(), memory=TradeMemory(), period_label="weekly")

    return {"screener": screener, "defense": defense, "report": report}


def launch_workflow(jobs: "Jobs", runners: Dict[str, Callable[[], Any]], name: str) -> Dict[str, Any]:
    """Start a workflow in the background (shared with the UI buttons) and return
    immediately. Used by the agent's workflow tools so chat never blocks."""
    if name not in runners:
        return {"error": f"unknown workflow '{name}'", "available": list(runners)}
    if not jobs.start(name, runners[name]):
        return {"status": "busy",
                "message": f"'{jobs.running}' is already running — try again once it finishes."}
    window = {"screener": "Entry Screener", "defense": "Downside Defense",
              "report": "Performance Report"}.get(name, name)
    return {"status": "started", "workflow": name,
            "message": f"Started '{name}' in the background. Open the '{window}' workflow window "
                       f"to watch live progress and logs."}


def _live_price_provider(symbols):
    """Fetch current market prices {symbol: last_price} via Schwab. Best-effort:
    any failure yields an empty map so the holdings view still renders."""
    if not symbols:
        return {}
    try:
        from app.data.schwab_client import SchwabClient
        client = SchwabClient()
        quotes = client.get_quotes_chunked(list(symbols))
        out = {}
        for sym in symbols:
            px = client.extract_fundamentals(quotes, sym).get("last_price", 0.0)
            if px and px > 0:
                out[sym] = px
        return out
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("web").warning("Live price lookup failed: %s", exc)
        return {}


def _default_holdings_provider() -> Dict[str, Any]:
    """Cash + every position with its covered-call contract details (from SQL),
    enriched with a live current stock price per holding."""
    from app.memory.positions_store import list_holdings_detailed
    from app.memory.account_store import get_cash_balance
    from app.config import settings as _s
    return {"cash_balance": get_cash_balance(_s.sql_db_path),
            "positions": list_holdings_detailed(
                db_path=_s.sql_db_path, price_provider=_live_price_provider)}


def _default_agent_provider(launcher: Optional[Callable[[str], Any]] = None) -> Callable[[], Any]:
    holder: Dict[str, Any] = {}

    def provider() -> Any:
        if "agent" not in holder:  # lazy: don't load the 9GB model until first chat
            from app.llm import get_llm
            from app.manage import ManagementService
            from app.memory.vector_db import TradeMemory
            from app.agent.tools import build_tools
            from app.agent.agent import CoveredCallAgent
            svc = ManagementService(memory=TradeMemory())
            tools = build_tools(svc, svc._memory, workflow_launcher=launcher)
            holder["agent"] = CoveredCallAgent(get_llm(), tools)
        return holder["agent"]

    return provider


# ── app factory ─────────────────────────────────────────────────────────
def create_app(
    *,
    agent_provider: Optional[Callable[[], Any]] = None,
    runners: Optional[Dict[str, Callable[[], Any]]] = None,
    holdings_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    log_buffer: Optional[LogBuffer] = None,
    run_async: bool = True,
    attach_logging: bool = True,
) -> Flask:
    app = Flask(__name__, static_folder=str(_STATIC))
    # Silence Werkzeug's per-request access log — the UI polls constantly.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    buffer = log_buffer or LogBuffer()
    if attach_logging:
        buffer.setLevel(logging.INFO)
        root = logging.getLogger()
        if root.level > logging.INFO or root.level == 0:
            root.setLevel(logging.INFO)
        root.addHandler(buffer)

    jobs = Jobs(run_async=run_async)
    runners = runners if runners is not None else _default_runners()
    holdings_provider = holdings_provider or _default_holdings_provider
    # The agent's workflow tools start background jobs via this same launcher, so
    # agent-triggered runs are non-blocking and show up in the workflow windows.
    if agent_provider is None:
        agent_provider = _default_agent_provider(lambda name: launch_workflow(jobs, runners, name))

    @app.get("/")
    def index():  # noqa: ANN202
        return send_from_directory(_STATIC, "index.html")

    @app.get("/workflow")
    def workflow_window():  # noqa: ANN202
        return send_from_directory(_STATIC, "workflow.html")

    @app.get("/holdings")
    def holdings_window():  # noqa: ANN202
        return send_from_directory(_STATIC, "holdings.html")

    @app.get("/api/holdings")
    def api_holdings():  # noqa: ANN202
        try:
            return jsonify(holdings_provider())
        except Exception as exc:  # noqa: BLE001
            logger.exception("Holdings fetch failed")
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/chat")
    def chat():  # noqa: ANN202
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "empty message"}), 400
        try:
            result = agent_provider().chat(message, history=data.get("history"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent chat failed")
            return jsonify({"error": str(exc)}), 500
        return jsonify(result)

    @app.post("/api/run/<name>")
    def run_workflow(name: str):  # noqa: ANN202
        if name not in runners:
            return jsonify({"error": f"unknown workflow '{name}'", "available": list(runners)}), 404
        started = jobs.start(name, runners[name])
        if not started:
            return jsonify({"error": f"'{jobs.running}' is already running"}), 409
        return jsonify({"started": name})

    @app.get("/api/status")
    def status():  # noqa: ANN202
        return jsonify({"running": jobs.running, "last": jobs.last,
                        "workflows": list(runners)})

    @app.get("/api/logs")
    def logs():  # noqa: ANN202
        cursor = request.args.get("since", default=0, type=int)
        return jsonify(buffer.since(cursor))

    app.config["LOG_BUFFER"] = buffer
    app.config["JOBS"] = jobs
    return app


def run(host: str = "127.0.0.1", port: int = 8765) -> None:  # pragma: no cover
    from app.logging_config import setup_logging
    setup_logging("INFO")
    app = create_app()
    logger.info("Covered Call Command Center UI → http://%s:%d", host, port)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":  # pragma: no cover
    run()
