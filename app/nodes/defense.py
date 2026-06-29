"""Downside-defense nodes (Tree-of-Thoughts engine, design §4).

When an open covered-call position breaches its downside threshold, the agent
evaluates three escape routes — and per the design these branches are assessed
Quant → News → Risk Manager:

  Branch A (Hard Eject): buy-to-close the call, sell the shares, take the loss.
  Branch B (Roll Down):  buy-to-close, sell a new lower-strike call for credit.
  Branch C (Hold & Wait): do nothing, let the position develop.

The Quant node computes exact P&L for all three (deterministic). The News node
checks whether the drop is driven by catastrophic news. The Risk Manager picks a
branch (LLM choice grounded in the deterministic numbers) and sends it to the
human via Discord — execution stays human-only.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import rules as default_rules
from app.engine import math_engine as eng
from app.data.schwab_client import SchwabClient
from app.data.news_client import NewsClient
from app.llm import LocalLLM
from app.notify.discord_webhook import DiscordNotifier
from app.nodes.news import evaluate_news, _sources
from app.runlog import save_run
from app.state import DefenseState, NewsReport, sentiment_score

logger = logging.getLogger("node.defense")

BRANCH_LABELS = {
    "A": "Branch A — Hard Eject (close call + sell shares)",
    "B": "Branch B — Roll Down (buy-to-close, sell lower-strike call)",
    "C": "Branch C — Hold & Wait",
}


def _underlying_price(client: SchwabClient, sym: str) -> float:
    payload = client.get_quote(sym)
    entry = payload.get(sym.upper(), {}) if isinstance(payload, dict) else {}
    return float((entry.get("quote", {}) or {}).get("lastPrice", 0.0) or 0.0)


def _find_call_ask_by_strike(chain: Dict[str, Any], strike: float, expiration: Optional[str]) -> float:
    """Find the ask of the existing short call (match strike, optionally expiry)."""
    for exp_key, strikes in (chain.get("callExpDateMap", {}) or {}).items():
        if expiration and not str(exp_key).startswith(str(expiration)):
            continue
        for sk, contracts in strikes.items():
            try:
                if abs(float(sk) - float(strike)) < 1e-6 and contracts:
                    return float(contracts[0].get("ask", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    return 0.0


def _with_errors(out: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
    if errors:
        out["errors"] = errors
    return out


def _breach_threshold(pos: Dict[str, Any], rules) -> float:
    """The % drop (negative) that triggers defense for this position.

    Dynamic: use the position's stored downside-buffer % (premium cushion at
    entry) → defend once the drop exceeds the premium protection (i.e. past
    breakeven). Falls back to the static ``downside_breach_pct`` when no buffer
    was recorded.
    """
    buffer = pos.get("downside_buffer_percent")
    if buffer is not None and buffer > 0:
        return -abs(float(buffer))
    return rules.downside_breach_pct


def _resolve_current_price(client: SchwabClient, state: DefenseState, sym: str) -> Tuple[float, Optional[str]]:
    """Current underlying price — from state if provided, else fetched. Returns
    (price, error_or_None)."""
    price = state.get("current_stock_price")
    if price is not None:
        return float(price), None
    try:
        return _underlying_price(client, sym), None
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"Defense quote failed for {sym}: {exc}"


def _resolve_branch_inputs(
    client: SchwabClient, state: DefenseState, pos: Dict[str, Any], rules
) -> Tuple[float, float, Optional[str]]:
    """Resolve the two option inputs the branch P&L needs — the current short-call
    ask (to buy it back) and a roll-down premium — from state or the chain.
    Returns (current_call_ask, roll_down_premium, error_or_None)."""
    current_call_ask = state.get("current_call_ask")
    roll_premium = state.get("roll_down_premium")
    err = None
    if current_call_ask is None or roll_premium is None:
        try:
            chain = client.get_option_chain(pos["symbol"], contract_type="CALL", range_filter="ALL")
        except Exception as exc:  # noqa: BLE001
            chain, err = {}, f"Defense option-chain failed for {pos['symbol']}: {exc}"
        if current_call_ask is None:
            current_call_ask = _find_call_ask_by_strike(
                chain, pos.get("short_call_strike", 0.0), pos.get("short_call_expiration"))
        if roll_premium is None:
            best = eng.find_optimal_covered_call(
                chain, target_delta=rules.target_delta, delta_band=rules.delta_band,
                min_dte=rules.min_days_to_expiration, max_dte=rules.max_days_to_expiration)
            roll_premium = best.get("mark", 0.0) if "error" not in best else 0.0
    return float(current_call_ask or 0.0), float(roll_premium or 0.0), err


# ══════════════════════════════════════════════════════════════════════
#  Defense Quant — generate the three branches with exact P&L
# ══════════════════════════════════════════════════════════════════════
def build_defense_quant_node(
    client: SchwabClient, rules=default_rules, today: Optional[date] = None
) -> Callable[[DefenseState], dict]:

    def node(state: DefenseState) -> dict:
        pos = state["position"]
        sym = pos["symbol"]
        errors: List[str] = []

        price, price_err = _resolve_current_price(client, state, sym)
        if price_err:
            errors.append(price_err)
        entry = float(pos["stock_purchase_price"])
        drop_pct = ((price - entry) / entry * 100.0) if entry else 0.0

        # Dynamic breach threshold: a position's own premium cushion (downside
        # buffer % at entry) is where it crosses breakeven — defend there. Fall
        # back to the static rule when no buffer was stored.
        threshold = _breach_threshold(pos, rules)

        if not (price > 0 and drop_pct <= threshold):
            logger.info("Defense: %s drop %.1f%% within tolerance (threshold %.1f%%); no action.",
                        sym, drop_pct, threshold)
            return _with_errors({"current_stock_price": price, "breach_detected": False}, errors)

        logger.info("Defense: %s BREACH — drop %.1f%% <= %.1f%% (premium-cushion threshold); "
                    "generating ToT branches.", sym, drop_pct, threshold)

        current_call_ask, roll_premium, chain_err = _resolve_branch_inputs(client, state, pos, rules)
        if chain_err:
            errors.append(chain_err)

        # Loss is computed against the RAW cost basis (entry stock price), NOT an
        # adjusted basis — generate_tot_defense_branches uses entry_stock_price
        # directly, so the stock loss reflects the true drop from purchase price.
        branches = eng.generate_tot_defense_branches(
            entry_stock_price=entry, current_stock_price=price,
            original_premium=float(pos.get("original_premium", 0.0)),
            current_call_ask=current_call_ask, roll_down_premium=roll_premium)

        branch_analysis = {
            "drop_percent": round(drop_pct, 2), "current_stock_price": price,
            "raw_cost_basis": entry, "current_call_ask": current_call_ask,
            "roll_down_premium": roll_premium, "branches": branches,
        }
        return _with_errors({"current_stock_price": price, "current_call_ask": current_call_ask,
                             "breach_detected": True, "branch_analysis": branch_analysis}, errors)

    return node


# ══════════════════════════════════════════════════════════════════════
#  Defense News — is the drop driven by catastrophic news?
# ══════════════════════════════════════════════════════════════════════
def build_defense_news_node(
    news_client: NewsClient, llm: LocalLLM, rules=default_rules
) -> Callable[[DefenseState], dict]:

    def node(state: DefenseState) -> dict:
        sym = state["position"]["symbol"]
        errors: List[str] = []
        headlines: List[Dict[str, Any]] = []
        try:
            headlines = news_client.get_headlines(sym, limit=rules.headlines_per_symbol)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Defense news fetch failed for {sym}: {exc}")
        verdict = evaluate_news(llm, sym, headlines, rules)
        report: NewsReport = {
            "symbol": sym,
            "sentiment": verdict["sentiment"],
            "sentiment_score": sentiment_score(verdict["sentiment"]),
            "rationale": verdict.get("rationale", ""),
            "catastrophic_risk": verdict["catastrophic_risk"],
            "catastrophic_keywords": verdict["catastrophic_keywords"],
            "headlines_checked": headlines,
            "sources": _sources(headlines),
            "passes_news": True,
        }
        kw = verdict["catastrophic_keywords"]
        logger.info("Defense news %s: %s%s", sym, verdict["sentiment"],
                    f" (CATASTROPHIC: {', '.join(kw)})" if verdict["catastrophic_risk"] else "")
        out = {"news_report": report}
        if errors:
            out["errors"] = errors
        return out

    return node


# ══════════════════════════════════════════════════════════════════════
#  Defense Risk Manager — pick a branch, surface to the human
# ══════════════════════════════════════════════════════════════════════
_DECIDE_SYSTEM = (
    "You are a risk manager deciding how to defend a covered-call position that "
    "has dropped below its downside threshold. You are given exact P&L for three "
    "branches and a news read. Choose ONE branch:\n"
    "A = Hard Eject (realize the loss, free capital),\n"
    "B = Roll Down (only valid if it collects a net credit),\n"
    "C = Hold & Wait.\n"
    "Guidance: if news shows catastrophic risk, prefer A. If the roll collects a "
    "healthy credit and news is benign, prefer B. If the drop looks like noise "
    "and news is fine, C is acceptable. Never recommend B if its net credit is "
    "not positive."
)


def _choose_branch(llm: LocalLLM, sym: str, ba: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    branches = ba.get("branches", {})
    a = branches.get("Branch_A_Liquidate", {})
    b = branches.get("Branch_B_Roll_Down", {})
    c = branches.get("Branch_C_Hold", {})
    user = (
        f"Symbol {sym} is down {ba.get('drop_percent')}% from entry (now "
        f"${ba.get('current_stock_price')}).\n"
        f"Branch A (Hard Eject): realized cash loss ${a.get('realized_cash_loss')}, "
        f"capital freed ${a.get('capital_freed_up')}.\n"
        f"Branch B (Roll Down): net credit ${b.get('net_credit_received')}, "
        f"valid={b.get('is_valid')}, unrealized stock loss ${b.get('unrealized_stock_loss')}.\n"
        f"Branch C (Hold): unrealized net P&L ${c.get('unrealized_net_pnl')}.\n"
        f"News sentiment: {report.get('sentiment')}, catastrophic_risk="
        f"{report.get('catastrophic_risk')}. {report.get('rationale','')}\n"
        "Return JSON with keys: recommended_branch (A, B, or C) and rationale."
    )
    try:
        obj = llm.structured(_DECIDE_SYSTEM, user, required_keys=["recommended_branch", "rationale"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Defense decision LLM failed for %s: %s", sym, exc)
        return {"recommended_branch": "C", "rationale": f"Defaulted to Hold ({exc})."}
    branch = str(obj.get("recommended_branch", "C")).strip().upper()[:1]
    if branch not in ("A", "B", "C"):
        branch = "C"
    # Deterministic guardrail: never roll for a non-positive credit.
    if branch == "B" and not b.get("is_valid", False):
        obj["rationale"] = ("Overrode B→C: roll-down credit not positive. " + obj.get("rationale", ""))
        branch = "C"
    obj["recommended_branch"] = branch
    return obj


def _format_defense(pos, ba, report, branch, rationale) -> str:
    b = ba.get("branches", {})
    lines = [
        "🛡️ **Downside Defense — HUMAN DECISION REQUIRED**",
        f"**{pos['symbol']}** is down **{ba.get('drop_percent')}%** "
        f"(now ${ba.get('current_stock_price')}, entry ${pos.get('stock_purchase_price')}).",
        "⚠️ Autonomous trading is disabled. Review and execute manually.\n",
        f"**Recommended: {BRANCH_LABELS.get(branch, branch)}**",
        f"_{rationale}_\n",
        "**Branches evaluated:**",
        f"• A — Hard Eject: realized loss ${b.get('Branch_A_Liquidate', {}).get('realized_cash_loss')}",
        f"• B — Roll Down: net credit ${b.get('Branch_B_Roll_Down', {}).get('net_credit_received')} "
        f"(valid={b.get('Branch_B_Roll_Down', {}).get('is_valid')})",
        f"• C — Hold: unrealized P&L ${b.get('Branch_C_Hold', {}).get('unrealized_net_pnl')}",
        f"\nNews: {report.get('sentiment')} — {report.get('rationale','')}",
    ]
    return "\n".join(lines)


def build_defense_risk_node(
    llm: LocalLLM, notifier: Optional[DiscordNotifier] = None, rules=default_rules
) -> Callable[[DefenseState], dict]:

    def node(state: DefenseState) -> dict:
        pos = state["position"]
        sym = pos["symbol"]
        ba = state.get("branch_analysis") or {}
        report = state.get("news_report") or {}

        decision = _choose_branch(llm, sym, ba, report)
        branch = decision["recommended_branch"]
        rationale = decision.get("rationale", "")

        rec = {
            "symbol": sym,
            "position_id": pos.get("position_id"),
            "action": BRANCH_LABELS.get(branch, branch),
            "branch": branch,
            "rationale": rationale,
            "sentiment": report.get("sentiment"),
            "branch_analysis": ba,
        }
        summary = _format_defense(pos, ba, report, branch, rationale)
        run_id = state.get("run_id", "")
        run_paths = save_run(run_id, summary, [rec], run_timestamp=state.get("run_timestamp", ""),
                             workflow="defense_monitor")

        notified = False
        errors: List[str] = []
        if notifier is not None:
            try:
                notified = notifier.send(summary)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Defense Discord notify failed: {exc}")

        logger.info("Defense decision for %s: %s (notified=%s)", sym, branch, notified)
        out = {"defense_recommendation": rec, "discord_summary": summary,
               "notified": notified, "run_log_paths": run_paths}
        if errors:
            out["errors"] = errors
        return out

    return node
