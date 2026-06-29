"""Human-In-The-Loop feedback CLI.

After a screener run sends candidates to Discord and saves them to
``runs/<run_id>.json``, the human reviews and records a verdict here. ONLY at
this point does anything get written to the SQL decision ledger / positions and
to the ChromaDB lesson memory (design §3: "only after final human feedback").

Run:  ./venv/bin/python -m app.feedback <run_id>

Architecture: ``process_feedback`` is a pure function (verdict list in →
records out) so it is unit-testable with a temp DB and a fake memory. The
interactive ``main`` just collects verdicts via prompts and delegates to it.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from app.config import settings
from app.memory import decision_store as ds
from app.runlog import load_run
from app.state import sentiment_score  # noqa: F401  (kept for parity/imports)

logger = logging.getLogger("feedback")

VERDICT_APPROVE = "APPROVE"
VERDICT_DENY = "DENY"
VERDICT_SKIP = "SKIP"


def _lesson_text(rec: Dict[str, Any], verdict: str, notes: str) -> str:
    c = rec.get("contract", {})
    exp = str(c.get("expiration_key", "")).split(":")[0]
    return (
        f"{verdict} covered call on {rec.get('symbol')}: sell {c.get('strike')} call "
        f"exp {exp}, grade {rec.get('grade')}, annualized yield "
        f"{rec.get('annualized_yield_percent')}%, sentiment {rec.get('sentiment')}. "
        f"Human notes: {notes or '(none)'}"
    )


def process_feedback(
    run_data: Dict[str, Any],
    decisions: List[Dict[str, Any]],
    *,
    db_path: Optional[Union[str, Path]] = None,
    memory: Any = None,
    runs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Apply human verdicts to a run's proposals.

    ``decisions`` is parallel to ``run_data['recommendations']``; each item:
        {"verdict": APPROVE|DENY|SKIP, "notes": str,
         "fill": {"shares": int, "stock_price": float,
                  "premium": float, "contracts": int}}   # required for APPROVE

    Writes decision_logs rows (+ a position on APPROVE) and a ChromaDB lesson,
    then marks the run JSON REVIEWED. Returns a summary of what was recorded.
    """
    run_id = run_data.get("run_id", "")
    recs = run_data.get("recommendations", [])
    if len(decisions) != len(recs):
        raise ValueError(f"Got {len(decisions)} decisions for {len(recs)} recommendations.")

    summary = {"run_id": run_id, "approved": [], "denied": [], "skipped": [], "log_ids": []}

    for rec, decision in zip(recs, decisions):
        verdict = str(decision.get("verdict", "")).upper()
        notes = decision.get("notes", "")
        symbol = rec.get("symbol")

        if verdict == VERDICT_SKIP:
            summary["skipped"].append(symbol)
            continue
        if verdict not in (VERDICT_APPROVE, VERDICT_DENY):
            raise ValueError(f"Invalid verdict {verdict!r} for {symbol}.")

        position_id: Optional[str] = None
        if verdict == VERDICT_APPROVE:
            fill = decision.get("fill") or {}
            contract = rec.get("contract", {})
            position_id = f"{symbol}_{run_id}"
            ds.open_position(
                position_id=position_id,
                symbol=symbol,
                stock_purchase_price=float(fill["stock_price"]),
                shares=int(fill["shares"]),
                call_strike=float(contract.get("strike")),
                call_premium=float(fill["premium"]),
                call_expiration=str(contract.get("expiration_key", "")).split(":")[0],
                contracts=int(fill.get("contracts", 1)),
                # Store the premium cushion → the defense monitor uses it as this
                # position's dynamic breach threshold.
                downside_buffer_percent=rec.get("yield_metrics", {}).get("downside_buffer_percent"),
                db_path=db_path,
            )
            summary["approved"].append(symbol)
        else:
            summary["denied"].append(symbol)

        log_id = ds.log_decision(
            symbol=symbol,
            workflow_stage="ENTRY_SCREENER",
            agent_recommendation=rec,
            agent_rationale=rec.get("rationale", ""),
            is_human_approved=(verdict == VERDICT_APPROVE),
            human_feedback_notes=notes,
            position_id=position_id,
            db_path=db_path,
        )
        summary["log_ids"].append(log_id)

        # Store the lesson for future semantic recall (design §5).
        if memory is not None:
            try:
                memory.add_lesson(
                    lesson_id=f"{run_id}_{symbol}",
                    text=_lesson_text(rec, verdict, notes),
                    metadata={
                        "symbol": symbol, "verdict": verdict, "grade": rec.get("grade"),
                        "annualized_yield_percent": rec.get("annualized_yield_percent"),
                        "sentiment": rec.get("sentiment"), "run_id": run_id,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — memory is non-critical
                logger.warning("Failed to store lesson for %s: %s", symbol, exc)

    # Mark the run reviewed (best-effort).
    try:
        run_data["status"] = "REVIEWED"
        run_data["decisions"] = decisions
        path = Path(runs_dir or settings.runs_dir) / f"{run_id}.json"
        path.write_text(json.dumps(run_data, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not update run file status: %s", exc)

    logger.info("Feedback for %s: %d approved, %d denied, %d skipped.",
                run_id, len(summary["approved"]), len(summary["denied"]), len(summary["skipped"]))
    return summary


# ══════════════════════════════════════════════════════════════════════
#  Defense-workflow feedback (acts on an EXISTING position, not a new one)
# ══════════════════════════════════════════════════════════════════════
_BRANCH_LABEL = {"A": "Hard Eject (liquidate)", "B": "Roll Down", "C": "Hold & Wait"}


def _defense_lesson_text(rec: Dict[str, Any], verdict: str, branch: str, notes: str) -> str:
    ba = rec.get("branch_analysis", {})
    return (
        f"{verdict} defense on {rec.get('symbol')} (branch {branch} = {_BRANCH_LABEL.get(branch, branch)}): "
        f"down {ba.get('drop_percent')}% from entry, sentiment {rec.get('sentiment')}. "
        f"Agent recommended branch {rec.get('branch')}. Human notes: {notes or '(none)'}"
    )


def _mark_reviewed(run_data: Dict[str, Any], decisions: List[Dict[str, Any]],
                   runs_dir: Optional[Path]) -> None:
    try:
        run_data["status"] = "REVIEWED"
        run_data["decisions"] = decisions
        path = Path(runs_dir or settings.runs_dir) / f"{run_data.get('run_id', '')}.json"
        path.write_text(json.dumps(run_data, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not update run file status: %s", exc)


def process_defense_feedback(
    run_data: Dict[str, Any],
    decision: Dict[str, Any],
    *,
    db_path: Optional[Union[str, Path]] = None,
    memory: Any = None,
    runs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Record the human's decision on a downside-defense recommendation.

    ``decision``: {"verdict": APPROVE|DENY|SKIP, "branch": A|B|C (executed; default
    = recommended), "notes": str, "fill": {...}}. APPROVE executes the branch:
      A → close_position(LIQUIDATED) with stock_sale_price (+ optional buyback),
      B → roll_position(buyback + new lower call), C → hold (no trade).
    Always logs to decision_logs (stage DEFENSE_MONITOR) + stores a lesson.
    """
    recs = run_data.get("recommendations", [])
    if not recs:
        raise ValueError("No defense recommendation to act on.")
    rec = recs[0]
    run_id = run_data.get("run_id", "")
    symbol, position_id = rec.get("symbol"), rec.get("position_id")
    verdict = str(decision.get("verdict", "")).upper()
    branch = str(decision.get("branch") or rec.get("branch", "C")).upper()[:1]
    notes = decision.get("notes", "")
    out: Dict[str, Any] = {"run_id": run_id, "symbol": symbol, "position_id": position_id,
                           "verdict": verdict, "branch": branch, "action": None}

    if verdict == VERDICT_SKIP:
        out["action"] = "skipped"
        _mark_reviewed(run_data, [decision], runs_dir)
        return out
    if verdict not in (VERDICT_APPROVE, VERDICT_DENY):
        raise ValueError(f"Invalid verdict {verdict!r}.")

    approved = verdict == VERDICT_APPROVE
    if approved:
        fill = decision.get("fill") or {}
        if branch == "A":
            ds.close_position(
                position_id=position_id, status="LIQUIDATED",
                stock_sale_price=float(fill["stock_sale_price"]),
                call_buyback_price=(float(fill["call_buyback_price"])
                                    if fill.get("call_buyback_price") is not None else None),
                contracts=int(fill.get("contracts", 1)), db_path=db_path)
            out["action"] = "LIQUIDATED"
        elif branch == "B":
            res = ds.roll_position(
                position_id=position_id, call_buyback_price=float(fill["call_buyback_price"]),
                new_call_strike=float(fill["new_call_strike"]),
                new_call_premium=float(fill["new_call_premium"]),
                new_call_expiration=str(fill["new_call_expiration"]),
                contracts=int(fill.get("contracts", 1)), db_path=db_path)
            out["action"], out["net_credit"] = "ROLLED", res["net_credit"]
        else:  # C — hold
            out["action"] = "HELD (no trade)"
    else:
        out["action"] = "denied (no trade)"

    branches = (rec.get("branch_analysis") or {}).get("branches")
    out["log_id"] = ds.log_decision(
        symbol=symbol, workflow_stage="DEFENSE_MONITOR", agent_recommendation=rec,
        tot_branches=branches, agent_rationale=rec.get("rationale", ""),
        is_human_approved=approved, human_feedback_notes=f"[branch {branch}] {notes}",
        position_id=position_id, db_path=db_path)

    if memory is not None:
        try:
            memory.add_lesson(lesson_id=f"{run_id}_{symbol}_defense",
                              text=_defense_lesson_text(rec, verdict, branch, notes),
                              metadata={"symbol": symbol, "verdict": verdict, "branch": branch,
                                        "workflow": "defense_monitor", "run_id": run_id})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to store defense lesson for %s: %s", symbol, exc)

    _mark_reviewed(run_data, [decision], runs_dir)
    logger.info("Defense feedback %s: %s branch %s (%s)", run_id, verdict, branch, out["action"])
    return out


def _prompt_defense_decision(rec: Dict[str, Any], *, input_func=input, output_func=print) -> Dict[str, Any]:
    ba = rec.get("branch_analysis", {})
    b = ba.get("branches", {})
    rec_branch = str(rec.get("branch", "C")).upper()[:1]
    output_func(
        f"\n{rec['symbol']} — position {rec.get('position_id')} is down "
        f"{ba.get('drop_percent')}% (now ${ba.get('current_stock_price')}).\n"
        f"  Agent recommends: branch {rec_branch} — {_BRANCH_LABEL.get(rec_branch, rec_branch)}\n"
        f"  A Hard Eject: realized loss ${b.get('Branch_A_Liquidate', {}).get('realized_cash_loss')}\n"
        f"  B Roll Down: net credit ${b.get('Branch_B_Roll_Down', {}).get('net_credit_received')} "
        f"(valid={b.get('Branch_B_Roll_Down', {}).get('is_valid')})\n"
        f"  C Hold: unrealized P&L ${b.get('Branch_C_Hold', {}).get('unrealized_net_pnl')}\n"
        f"  Reasoning: {rec.get('rationale', '')}")
    choice = ""
    while choice not in ("A", "B", "C", "D", "S"):
        choice = (input_func(f"  Execute [A/B/C], [D]eny, [S]kip (default {rec_branch}): ")
                  .strip().upper()[:1] or rec_branch)
    if choice == "D":
        return {"verdict": VERDICT_DENY, "branch": rec_branch,
                "notes": input_func("  Notes: ").strip()}
    if choice == "S":
        return {"verdict": VERDICT_SKIP, "branch": rec_branch, "notes": ""}

    notes = input_func("  Notes (your reasoning): ").strip()
    decision: Dict[str, Any] = {"verdict": VERDICT_APPROVE, "branch": choice, "notes": notes}
    cur_price = ba.get("current_stock_price", 0)
    cur_ask = ba.get("current_call_ask", 0)
    roll_prem = ba.get("roll_down_premium", 0)
    if choice == "A":
        decision["fill"] = {
            "stock_sale_price": float(input_func(f"  Stock sale price [{cur_price}]: ").strip() or cur_price),
            "call_buyback_price": float(input_func(f"  Call buyback price [{cur_ask}]: ").strip() or cur_ask),
            "contracts": int(input_func("  Contracts [1]: ").strip() or 1)}
    elif choice == "B":
        decision["fill"] = {
            "call_buyback_price": float(input_func(f"  Call buyback price [{cur_ask}]: ").strip() or cur_ask),
            "new_call_strike": float(input_func("  New (lower) call strike: ").strip()),
            "new_call_premium": float(input_func(f"  New call premium [{roll_prem}]: ").strip() or roll_prem),
            "new_call_expiration": input_func("  New call expiration (YYYY-MM-DD): ").strip(),
            "contracts": int(input_func("  Contracts [1]: ").strip() or 1)}
    return decision


# ── interactive prompt collection (entry screener) ────────────────────
_VERDICT_KEYS = {"A": VERDICT_APPROVE, "D": VERDICT_DENY, "S": VERDICT_SKIP}


def _render_candidate(rec: Dict[str, Any], i: int, n: int) -> str:
    c = rec.get("contract", {})
    return (
        f"\n[{i}/{n}] {rec['symbol']} — Grade {rec.get('grade')} (score {rec.get('score')})\n"
        f"  Sell {c.get('strike')} call exp {str(c.get('expiration_key','')).split(':')[0]} "
        f"({c.get('days_to_expiration')}d), mark ${c.get('mark')}\n"
        f"  Annualized {rec.get('annualized_yield_percent')}% | sentiment {rec.get('sentiment')}\n"
        f"  {rec.get('rationale','')}"
    )


def _prompt_verdict(input_func: Callable[[str], str]) -> str:
    verdict = ""
    while verdict not in _VERDICT_KEYS:
        verdict = input_func("  [A]pprove / [D]eny / [S]kip? ").strip().upper()[:1]
    return verdict


def _prompt_fill(rec: Dict[str, Any], input_func: Callable[[str], str]) -> Dict[str, Any]:
    """Collect the actual fill details for an approved trade."""
    c = rec.get("contract", {})
    shares = int(input_func("  Shares filled [100]: ").strip() or 100)
    stock_price = float(input_func(f"  Stock price paid [{c.get('strike','')}]: ").strip()
                        or rec.get("underlying_price", c.get("strike", 0)))
    premium = float(input_func(f"  Premium received per share [{c.get('mark', 0)}]: ").strip()
                    or c.get("mark", 0))
    contracts = int(input_func("  Contracts [1]: ").strip() or 1)
    return {"shares": shares, "stock_price": stock_price, "premium": premium, "contracts": contracts}


def _prompt_decisions(
    recs: List[Dict[str, Any]],
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    decisions = []
    for i, rec in enumerate(recs, 1):
        output_func(_render_candidate(rec, i, len(recs)))
        verdict = _prompt_verdict(input_func)
        decision: Dict[str, Any] = {
            "verdict": _VERDICT_KEYS[verdict],
            "notes": input_func("  Notes (your reasoning — the learning signal): ").strip(),
        }
        if verdict == "A":
            decision["fill"] = _prompt_fill(rec, input_func)
        decisions.append(decision)
    return decisions


def main(run_id: str) -> int:
    from app.memory.vector_db import TradeMemory

    run_data = load_run(run_id)
    if run_data is None:
        print(f"❌ No run found at {settings.runs_dir / (run_id + '.json')}")
        return 1
    recs = run_data.get("recommendations", [])
    if not recs:
        print(f"Run {run_id} has no recommendations to review.")
        return 0

    # Route by workflow: defense decisions act on an existing position, entry
    # decisions open new ones. Fall back to detecting a defense rec by its "branch".
    workflow = run_data.get("workflow") or (
        "defense_monitor" if "branch" in recs[0] else "entry_screener")

    if workflow == "defense_monitor":
        print(f"Reviewing downside-defense decision from run {run_id}.")
        decision = _prompt_defense_decision(recs[0])
        summary = process_defense_feedback(run_data, decision, memory=TradeMemory())
        print(f"\n✅ Recorded defense decision: {summary['verdict']} branch "
              f"{summary['branch']} → {summary['action']}.")
        return 0

    print(f"Reviewing {len(recs)} candidate(s) from run {run_id}.")
    decisions = _prompt_decisions(recs)
    summary = process_feedback(run_data, decisions, memory=TradeMemory())
    print(f"\n✅ Recorded: {len(summary['approved'])} approved, "
          f"{len(summary['denied'])} denied, {len(summary['skipped'])} skipped.")
    if summary["approved"]:
        print(f"   Opened positions for: {', '.join(summary['approved'])}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m app.feedback <run_id>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
