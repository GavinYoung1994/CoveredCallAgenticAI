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
from datetime import date, timedelta
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


def _roll_contract_details(best: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the human-facing details of the proposed roll-down call."""
    if not best or "error" in best:
        return None
    return {
        "strike": best.get("strike"),
        "expiration": str(best.get("expiration_key", "")).split(":")[0] or None,
        "days_to_expiration": best.get("days_to_expiration"),
        "delta": best.get("delta"),
        "premium": best.get("mark", 0.0),
    }


def _earnings_capped_max_dte(rules, earnings_date, run_today: date) -> Tuple[int, Optional[str]]:
    """Cap the roll-down max DTE so the new call expires BEFORE the next earnings
    date (the same guardrail entry uses). Returns (max_dte, note_or_None)."""
    default_max = rules.max_days_to_expiration
    ed = eng._coerce_date(earnings_date) if earnings_date else None
    if ed is None:
        return default_max, None
    days_to_earnings = (ed - run_today).days
    # Expire strictly before earnings.
    capped = min(default_max, days_to_earnings - 1)
    if capped < default_max:
        return capped, (f"Roll capped to expire before earnings on {ed.isoformat()} "
                        f"({days_to_earnings}d out).")
    return default_max, None


def _resolve_branch_inputs(
    client: SchwabClient, state: DefenseState, pos: Dict[str, Any], rules,
    *, earnings_date=None, run_today: Optional[date] = None,
) -> Tuple[float, float, Optional[Dict[str, Any]], List[str], Optional[str]]:
    """Resolve the two option inputs the branch P&L needs — the current short-call
    ask (to buy it back) and a roll-down premium — from state or the chain, plus
    the details of the proposed roll-down contract (strike/expiration/DTE).

    The roll-down contract's expiration is constrained to fall BEFORE the next
    earnings date (earnings guardrail). Returns
    (current_call_ask, roll_down_premium, roll_contract, notes, error_or_None)."""
    current_call_ask = state.get("current_call_ask")
    roll_premium = state.get("roll_down_premium")
    roll_contract: Optional[Dict[str, Any]] = None
    notes: List[str] = []
    err = None
    run_today = run_today or date.today()

    max_dte, cap_note = _earnings_capped_max_dte(rules, earnings_date, run_today)
    if cap_note:
        notes.append(cap_note)

    if current_call_ask is None or roll_premium is None:
        try:
            chain = client.get_option_chain(pos["symbol"], contract_type="CALL", range_filter="ALL")
        except Exception as exc:  # noqa: BLE001
            chain, err = {}, f"Defense option-chain failed for {pos['symbol']}: {exc}"
        if current_call_ask is None:
            current_call_ask = _find_call_ask_by_strike(
                chain, pos.get("short_call_strike", 0.0), pos.get("short_call_expiration"))
        if max_dte < rules.min_days_to_expiration:
            # Earnings is so close no safe roll expiration exists.
            notes.append(f"No roll-down contract fits before earnings (need ≥{rules.min_days_to_expiration}d, "
                         f"only {max_dte}d available).")
        else:
            # Always identify the best roll-down candidate for its contract
            # details, even when the premium was injected via state.
            best = eng.find_optimal_covered_call(
                chain, target_delta=rules.target_delta, delta_band=rules.delta_band,
                min_dte=rules.min_days_to_expiration, max_dte=max_dte)
            roll_contract = _roll_contract_details(best)
            if roll_premium is None:
                roll_premium = best.get("mark", 0.0) if "error" not in best else 0.0
    return (float(current_call_ask or 0.0), float(roll_premium or 0.0), roll_contract, notes, err)


# ══════════════════════════════════════════════════════════════════════
#  Defense Quant — generate the three branches with exact P&L
# ══════════════════════════════════════════════════════════════════════
def _lookup_next_earnings(earnings_client, sym: str, run_today: date) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort next-earnings-date lookup. Returns (date_or_None, error_or_None)."""
    if earnings_client is None:
        return None, None
    try:
        to_d = (run_today + timedelta(days=120)).isoformat()
        return earnings_client.get_next_earnings_date(sym, run_today.isoformat(), to_d), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Defense earnings lookup failed for {sym}: {exc}"


def build_defense_quant_node(
    client: SchwabClient, earnings_provider: Optional[Callable[[], Any]] = None,
    rules=default_rules, today: Optional[date] = None,
) -> Callable[[DefenseState], dict]:
    """``earnings_provider`` is a zero-arg callable returning an earnings client
    (or None). It's invoked lazily — only when a breach needs a roll contract —
    so the (network) client isn't constructed on every graph build."""
    _cache: Dict[str, Any] = {}

    def _earnings_client():
        if earnings_provider is None:
            return None
        if "client" not in _cache:
            try:
                _cache["client"] = earnings_provider()
            except Exception as exc:  # noqa: BLE001 — a failed client just means no earnings cap
                logger.warning("Could not build earnings client (%s); roll will not be earnings-capped.", exc)
                _cache["client"] = None
        return _cache["client"]

    def node(state: DefenseState) -> dict:
        pos = state["position"]
        sym = pos["symbol"]
        errors: List[str] = []
        run_today = today or date.today()

        price, price_err = _resolve_current_price(client, state, sym)
        if price_err:
            errors.append(price_err)
        entry = float(pos["stock_purchase_price"])
        drop_pct = ((price - entry) / entry * 100.0) if entry else 0.0

        # Dynamic breach threshold: a position's own premium cushion (downside
        # buffer % at entry) is where it crosses breakeven — defend there. Fall
        # back to the static rule when no buffer was stored.
        threshold = _breach_threshold(pos, rules)

        # drop_pct is a SIGNED change: positive = stock up, negative = stock down.
        if not (price > 0 and drop_pct <= threshold):
            logger.info("Defense: %s change %+.1f%% — within the %.1f%% downside cushion; no action.",
                        sym, drop_pct, abs(threshold))
            return _with_errors({"current_stock_price": price, "breach_detected": False}, errors)

        logger.info("Defense: %s BREACH — down %.1f%% exceeds the %.1f%% downside cushion; "
                    "generating ToT branches.", sym, abs(drop_pct), abs(threshold))

        # Earnings guardrail: the roll-down contract must expire before the next
        # earnings report (same rule the entry screener enforces). Only look it up
        # when we actually need to pick a roll contract from the chain (i.e. the
        # roll inputs weren't pre-supplied) — avoids a needless earnings API call.
        need_chain = state.get("current_call_ask") is None or state.get("roll_down_premium") is None
        earnings_date, earn_err = (None, None)
        if need_chain:
            earnings_date, earn_err = _lookup_next_earnings(_earnings_client(), sym, run_today)
        if earn_err:
            errors.append(earn_err)

        current_call_ask, roll_premium, roll_contract, roll_notes, chain_err = _resolve_branch_inputs(
            client, state, pos, rules, earnings_date=earnings_date, run_today=run_today)
        if chain_err:
            errors.append(chain_err)
        for n in roll_notes:
            logger.info("Defense roll (%s): %s", sym, n)

        # Loss is computed against the RAW cost basis (entry stock price), NOT an
        # adjusted basis — generate_tot_defense_branches uses entry_stock_price
        # directly, so the stock loss reflects the true drop from purchase price.
        # new_call_strike lets Branch B report the loss locked in if assigned at
        # a strike below the cost basis.
        new_strike = roll_contract.get("strike") if roll_contract else None
        branches = eng.generate_tot_defense_branches(
            entry_stock_price=entry, current_stock_price=price,
            original_premium=float(pos.get("original_premium", 0.0)),
            current_call_ask=current_call_ask, roll_down_premium=roll_premium,
            new_call_strike=new_strike)

        branch_analysis = {
            "drop_percent": round(drop_pct, 2), "current_stock_price": price,
            "raw_cost_basis": entry, "current_call_ask": current_call_ask,
            "roll_down_premium": roll_premium, "roll_down_contract": roll_contract,
            "next_earnings_date": earnings_date, "roll_notes": roll_notes,
            "existing_short_call": {
                "strike": pos.get("short_call_strike"),
                "expiration": pos.get("short_call_expiration"),
                "original_premium": pos.get("original_premium"),
                "shares": pos.get("shares"),
                "buy_to_close_ask": current_call_ask,
            },
            "branches": branches,
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
    "not positive.\n"
    "IMPORTANT — when the roll's new strike is BELOW the cost basis, rolling down "
    "caps the upside at a loss: if the shares are later called away at that strike "
    "you LOCK IN 'net_pnl_if_assigned_at_new_strike'. Weigh that locked-in loss "
    "against Branch A's realized loss. If rolling down merely defers a comparable "
    "(or worse) loss while capping recovery, prefer A or C over B."
)


def _choose_branch(llm: LocalLLM, sym: str, ba: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    branches = ba.get("branches", {})
    a = branches.get("Branch_A_Liquidate", {})
    b = branches.get("Branch_B_Roll_Down", {})
    c = branches.get("Branch_C_Hold", {})
    ex = ba.get("existing_short_call", {}) or {}
    rc = ba.get("roll_down_contract") or {}
    cost_basis = ba.get("raw_cost_basis")
    roll_line = (
        f" Roll into the {rc.get('strike')} call exp {rc.get('expiration')} "
        f"({rc.get('days_to_expiration')} days, premium ${rc.get('premium')})."
        if rc else " (new roll-down contract details unavailable.)"
    )
    # The assignment-loss detail the user wants factored in.
    assign_line = ""
    if b.get("net_pnl_if_assigned_at_new_strike") is not None:
        assign_line = (
            f" New strike {b.get('new_call_strike')} vs cost basis ${cost_basis} "
            f"(below basis={b.get('new_strike_below_cost_basis')}); if called away at the "
            f"new strike the TOTAL realized P&L would be "
            f"${b.get('net_pnl_if_assigned_at_new_strike')} "
            f"(stock {b.get('stock_loss_if_assigned_at_new_strike')} + premiums "
            f"{b.get('total_premiums_collected')})."
        )
    earn_line = (f"\nNext earnings: {ba.get('next_earnings_date')} — roll expiration is capped before it."
                 if ba.get('next_earnings_date') else "")
    user = (
        f"Symbol {sym} is down {abs(ba.get('drop_percent', 0))}% from entry (now "
        f"${ba.get('current_stock_price')}, cost basis ${cost_basis}).\n"
        f"Existing short call: {ex.get('strike')} strike exp {ex.get('expiration')}, "
        f"original premium ${ex.get('original_premium')}, buy-to-close ask "
        f"${ex.get('buy_to_close_ask')}.\n"
        f"Branch A (Hard Eject): realized cash loss ${a.get('realized_cash_loss')}, "
        f"capital freed ${a.get('capital_freed_up')}.\n"
        f"Branch B (Roll Down): net credit ${b.get('net_credit_received')}, "
        f"valid={b.get('is_valid')}, unrealized stock loss ${b.get('unrealized_stock_loss')}."
        f"{roll_line}{assign_line}\n"
        f"Branch C (Hold): unrealized net P&L ${c.get('unrealized_net_pnl')}.\n"
        f"News sentiment: {report.get('sentiment')}, catastrophic_risk="
        f"{report.get('catastrophic_risk')}. {report.get('rationale','')}"
        f"{earn_line}\n"
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
    ex = ba.get("existing_short_call", {}) or {}
    rc = ba.get("roll_down_contract") or {}
    lines = [
        "🛡️ **Downside Defense — HUMAN DECISION REQUIRED**",
        f"**{pos['symbol']}** is down **{abs(ba.get('drop_percent', 0))}%** "
        f"(now ${ba.get('current_stock_price')}, entry ${pos.get('stock_purchase_price')}).",
        "⚠️ Autonomous trading is disabled. Review and execute manually.\n",
        "**Current holding:**",
        f"• {ex.get('shares') or pos.get('shares')} shares @ ${pos.get('stock_purchase_price')} "
        f"(now ${ba.get('current_stock_price')})",
        f"• Short call: {ex.get('strike')} strike exp "
        f"{str(ex.get('expiration') or '').split(':')[0] or '—'}, "
        f"original premium ${ex.get('original_premium')}, buy-to-close ask ${ex.get('buy_to_close_ask')}\n",
        f"**Recommended: {BRANCH_LABELS.get(branch, branch)}**",
        f"_{rationale}_\n",
        "**Branches evaluated:**",
        f"• A — Hard Eject: realized loss ${b.get('Branch_A_Liquidate', {}).get('realized_cash_loss')}",
        f"• B — Roll Down: net credit ${b.get('Branch_B_Roll_Down', {}).get('net_credit_received')} "
        f"(valid={b.get('Branch_B_Roll_Down', {}).get('is_valid')})"
        + (f"\n    ↳ new call: {rc.get('strike')} strike exp {rc.get('expiration')} "
           f"({rc.get('days_to_expiration')}d, Δ{rc.get('delta')}, premium ${rc.get('premium')})"
           if rc else "\n    ↳ new roll-down contract details unavailable"),
    ]
    bb = b.get("Branch_B_Roll_Down", {})
    if bb.get("net_pnl_if_assigned_at_new_strike") is not None:
        flag = " ⚠️ new strike is BELOW your cost basis" if bb.get("new_strike_below_cost_basis") else ""
        lines.append(
            f"    ↳ if called away at ${bb.get('new_call_strike')}: total realized P&L "
            f"${bb.get('net_pnl_if_assigned_at_new_strike')} "
            f"(stock ${bb.get('stock_loss_if_assigned_at_new_strike')} + premiums "
            f"${bb.get('total_premiums_collected')}){flag}")
    if ba.get("next_earnings_date"):
        lines.append(f"    ↳ roll expiration capped before earnings on {ba.get('next_earnings_date')}")
    lines += [
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
