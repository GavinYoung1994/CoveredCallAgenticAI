"""Continuous improvement & feedback loop (design §5).

On a weekly/monthly cadence, summarize the agent's track record from the SQL
ledger — trades, approve/deny decisions, realized P&L, premiums harvested, win
rate, notable losses — and have the LLM write an analytical narrative. The
report is saved to runs/ and its lessons are stored in ChromaDB so future
decision cycles can recall them.

  ``gather_performance`` — pure SQL aggregation (deterministic, testable).
  ``generate_report``    — stats + optional LLM narrative + optional memory store.

Run:  ./venv/bin/python -m app.reporting [weekly|monthly]
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.config import settings
from app.runlog import save_run

logger = logging.getLogger("reporting")


def gather_performance(db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Aggregate the SQL ledger into a performance stats dict (deterministic)."""
    conn = sqlite3.connect(str(db_path or settings.sql_db_path))
    conn.row_factory = sqlite3.Row
    try:
        def scalar(sql: str, default=0):
            row = conn.execute(sql).fetchone()
            return (row[0] if row and row[0] is not None else default)

        decisions_total = scalar("SELECT COUNT(*) FROM decision_logs")
        approved = scalar("SELECT COUNT(*) FROM decision_logs WHERE is_human_approved = 1")
        denied = scalar("SELECT COUNT(*) FROM decision_logs WHERE is_human_approved = 0")

        open_positions = scalar("SELECT COUNT(*) FROM positions WHERE status = 'OPEN'")
        closed_positions = scalar("SELECT COUNT(*) FROM positions WHERE status != 'OPEN'")
        realized_pnl = scalar("SELECT SUM(total_realized_pnl) FROM positions WHERE status != 'OPEN'", 0.0)
        wins = scalar("SELECT COUNT(*) FROM positions WHERE status != 'OPEN' AND total_realized_pnl > 0")

        # Premium harvested: per-share premium × contracts × 100, over all sold calls.
        premium = scalar(
            "SELECT SUM(price * quantity * 100) FROM transactions "
            "WHERE asset_type = 'OPTION' AND action = 'SELL_TO_OPEN'", 0.0)

        # Capital invested in shares (cost basis) — closed vs currently-deployed.
        invested_closed = scalar(
            "SELECT SUM(t.price * t.quantity) FROM transactions t "
            "JOIN positions p ON t.position_id = p.position_id "
            "WHERE t.asset_type = 'STOCK' AND t.action = 'BUY_TO_OPEN' AND p.status != 'OPEN'", 0.0)
        invested_open = scalar(
            "SELECT SUM(t.price * t.quantity) FROM transactions t "
            "JOIN positions p ON t.position_id = p.position_id "
            "WHERE t.asset_type = 'STOCK' AND t.action = 'BUY_TO_OPEN' AND p.status = 'OPEN'", 0.0)
        avg_holding_days = scalar(
            "SELECT AVG(julianday(close_date) - julianday(entry_date)) FROM positions "
            "WHERE status != 'OPEN' AND close_date IS NOT NULL", None)

        losers = [dict(r) for r in conn.execute(
            "SELECT position_id, symbol, total_realized_pnl FROM positions "
            "WHERE status != 'OPEN' AND total_realized_pnl < 0 "
            "ORDER BY total_realized_pnl ASC LIMIT 5").fetchall()]

        recent_denials = [dict(r) for r in conn.execute(
            "SELECT symbol, human_feedback_notes FROM decision_logs "
            "WHERE is_human_approved = 0 AND human_feedback_notes != '' "
            "ORDER BY log_id DESC LIMIT 5").fetchall()]

        win_rate = (wins / closed_positions * 100.0) if closed_positions else 0.0

        # Return on invested capital (closed trades) + its annualization.
        invested_closed = float(invested_closed)
        realized_pnl = float(realized_pnl)
        roic_pct = (realized_pnl / invested_closed * 100.0) if invested_closed else 0.0
        avg_days = float(avg_holding_days) if avg_holding_days else None
        annualized_pct = (roic_pct * 365.0 / avg_days) if avg_days and avg_days > 0 else None

        return {
            "decisions_total": int(decisions_total),
            "approved": int(approved),
            "denied": int(denied),
            "open_positions": int(open_positions),
            "closed_positions": int(closed_positions),
            "realized_pnl": round(realized_pnl, 2),
            "wins": int(wins),
            "win_rate_percent": round(win_rate, 1),
            "total_premium_harvested": round(float(premium), 2),
            "invested_capital_closed": round(invested_closed, 2),
            "invested_capital_open": round(float(invested_open), 2),
            "return_on_invested_capital_percent": round(roic_pct, 2),
            "avg_holding_days": round(avg_days, 1) if avg_days else None,
            "annualized_return_percent": round(annualized_pct, 2) if annualized_pct is not None else None,
            "worst_losers": losers,
            "recent_denials": recent_denials,
        }
    finally:
        conn.close()


def _stats_markdown(stats: Dict[str, Any], period_label: str, asof: str) -> str:
    lines = [
        f"# 📊 Performance Report — {period_label} (as of {asof})",
        "",
        f"- Decisions logged: **{stats['decisions_total']}** "
        f"({stats['approved']} approved / {stats['denied']} denied)",
        f"- Positions: **{stats['open_positions']} open**, {stats['closed_positions']} closed",
        f"- Realized P&L (closed): **${stats['realized_pnl']:,.2f}**",
        f"- Win rate (closed): **{stats['win_rate_percent']}%** ({stats['wins']}/{stats['closed_positions']})",
        f"- Total premium harvested: **${stats['total_premium_harvested']:,.2f}**",
        f"- Capital invested (closed / still open): "
        f"${stats['invested_capital_closed']:,.0f} / ${stats['invested_capital_open']:,.0f}",
        f"- Return on invested capital (closed): **{stats['return_on_invested_capital_percent']}%**"
        + (f" over avg {stats['avg_holding_days']}d held" if stats['avg_holding_days'] else ""),
        f"- **Annualized return on invested cash: "
        + (f"{stats['annualized_return_percent']}%**" if stats['annualized_return_percent'] is not None
           else "n/a** (need closed trades with holding dates)"),
    ]
    if stats["worst_losers"]:
        lines.append("\n**Worst losers:**")
        for l in stats["worst_losers"]:
            lines.append(f"- {l['symbol']} ({l['position_id']}): ${l['total_realized_pnl']:,.2f}")
    if stats["recent_denials"]:
        lines.append("\n**Recent denials (human reasoning):**")
        for d in stats["recent_denials"]:
            lines.append(f"- {d['symbol']}: {d['human_feedback_notes']}")
    return "\n".join(lines)


_NARRATIVE_SYSTEM = (
    "You are reviewing an options-income agent's track record. Given the summary "
    "statistics, write a concise (4-6 sentence) analytical review: what worked, "
    "what didn't, any pattern in the losses or denials, and one concrete "
    "suggestion to improve the strategy or its thresholds. Use only the numbers "
    "provided; do not fabricate."
)


def generate_report(
    *,
    db_path: Optional[Union[str, Path]] = None,
    llm: Any = None,
    memory: Any = None,
    period_label: str = "weekly",
    asof: Optional[str] = None,
    runs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the performance report (stats + optional narrative), persist it, and
    store the lesson in ChromaDB. Returns {stats, narrative, markdown, paths}."""
    asof = asof or datetime.now(timezone.utc).date().isoformat()
    stats = gather_performance(db_path)
    markdown = _stats_markdown(stats, period_label, asof)

    narrative = ""
    if llm is not None:
        user = f"{period_label} performance stats:\n{stats}"
        try:
            narrative = llm.chat(_NARRATIVE_SYSTEM, user, max_tokens=300).strip()
            markdown += f"\n\n## Analyst review\n{narrative}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Report narrative LLM failed: %s", exc)

    report_id = f"report_{period_label}_{asof}"
    paths = save_run(report_id, markdown, [], run_timestamp=asof, runs_dir=runs_dir)

    # Persist the lesson so future decision cycles can recall it (design §5).
    if memory is not None:
        try:
            memory.add_lesson(
                lesson_id=report_id,
                text=(narrative or markdown)[:2000],
                metadata={
                    "type": "performance_report", "period": period_label, "asof": asof,
                    "realized_pnl": stats["realized_pnl"], "win_rate": stats["win_rate_percent"],
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to store report lesson: %s", exc)

    logger.info("Generated %s report (%s): P&L $%.2f, win rate %.1f%%",
                period_label, asof, stats["realized_pnl"], stats["win_rate_percent"])
    return {"stats": stats, "narrative": narrative, "markdown": markdown, "paths": paths}


def main(period: str = "weekly") -> int:
    from app.llm import get_llm
    from app.memory.vector_db import TradeMemory
    from app.logging_config import setup_logging

    setup_logging()
    result = generate_report(period_label=period, llm=get_llm(), memory=TradeMemory())
    print(result["markdown"])
    return 0


if __name__ == "__main__":
    period_arg = sys.argv[1] if len(sys.argv) > 1 else "weekly"
    sys.exit(main(period_arg))
