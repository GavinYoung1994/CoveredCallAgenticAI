"""Node 4 — The Portfolio & Risk Manager Agent (The Decision Maker).

Job (design §2/§3): join the Quant shortlist with the News reports, enforce the
>10% annualized-yield target, grade each survivor with a composite score, pick
the top-N, have the LLM write a human-readable rationale, and send the summary
to Discord for HUMAN approval (HITL — autonomous trading is forbidden).

Math (scoring, yield gate) is deterministic; the LLM only writes prose rationale.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import rules as default_rules
from app.engine import math_engine as eng
from app.llm import LocalLLM
from app.notify.discord_webhook import DiscordNotifier, format_recommendations
from app.runlog import save_run
from app.state import Recommendation, ScreenerState, reject

logger = logging.getLogger("node.risk")

_RATIONALE_SYSTEM = (
    "You are a portfolio risk manager presenting a covered-call trade to a human "
    "for approval. Write 3-4 plain, specific sentences that cover, in order:\n"
    "1. TECHNICALS — explain WHY the chart favors selling a covered call, citing "
    "the concrete indicators given (the detected trend and 50-day SMA slope, the "
    "RSI level and what it says about being over/under-bought, where price sits "
    "relative to the Bollinger bands and its moving averages, and whether "
    "volatility is contracting). Name the actual numbers.\n"
    "2. NEWS — summarize what the recent news says and why the sentiment is "
    "neutral-to-positive (i.e. no severe negative overhang), referencing the "
    "news finding provided.\n"
    "3. INCOME & RISK — the annualized yield, downside buffer, IV richness, and "
    "probability of keeping the premium; then name the single biggest risk.\n"
    "Use ONLY the numbers and text provided — do NOT invent data."
)


def _fmt_technicals(snapshot: Dict[str, Any], trend: Dict[str, Any]) -> str:
    """Render the Quant technical indicators as a compact briefing for the LLM."""
    price = snapshot.get("price")
    sma20, sma50, sma200 = snapshot.get("SMA_20"), snapshot.get("SMA_50"), snapshot.get("SMA_200")
    bb_up, bb_lo = snapshot.get("Bollinger_Upper"), snapshot.get("Bollinger_Lower")
    parts = [
        f"detected trend: {trend.get('detected_trend', 'n/a')} "
        f"(50-day SMA slope {trend.get('sma_50_slope_percent', 0):+.1f}% over "
        f"{trend.get('lookback_period_days', '?')}d)",
        f"RSI(14): {snapshot.get('RSI_14', 'n/a')}",
        f"price: {price}, SMA20: {sma20}, SMA50: {sma50}"
        + (f", SMA200: {sma200}" if sma200 is not None else ""),
        f"Bollinger band: {bb_lo} — {bb_up}",
        "volatility " + ("contracting (Bollinger squeeze)" if trend.get("is_volatility_contracting")
                         else "not contracting"),
    ]
    return "; ".join(parts)


def _llm_rationale(llm: LocalLLM, rec_data: Dict[str, Any]) -> str:
    news_line = rec_data.get("news_rationale") or "No notable news."
    headlines = rec_data.get("news_titles") or []
    headline_line = ("\n  Recent headlines: " + " | ".join(headlines[:3])) if headlines else ""
    user = (
        f"Symbol {rec_data['symbol']}: sell the {rec_data['strike']} call expiring "
        f"{rec_data['expiration']} ({rec_data['dte']} days, delta {rec_data['delta']:.2f}).\n"
        f"TECHNICALS — {rec_data['technicals']}\n"
        f"NEWS — sentiment {rec_data['sentiment']}; analyst finding: {news_line}{headline_line}\n"
        f"INCOME — annualized yield {rec_data['annualized_yield']:.1f}%, downside buffer "
        f"{rec_data['buffer']:.1f}%, IV/HV richness {rec_data['iv_ratio']:.2f}x, "
        f"probability of keeping premium {rec_data['prob_keep']:.0f}%. Composite grade "
        f"{rec_data['grade']}."
    )
    try:
        return llm.chat(_RATIONALE_SYSTEM, user, max_tokens=320).strip()
    except Exception as exc:  # noqa: BLE001 — prose is non-critical; never crash the run
        logger.warning("Rationale LLM failed for %s: %s", rec_data["symbol"], exc)
        return (f"Grade {rec_data['grade']}: {rec_data['annualized_yield']:.1f}% annualized, "
                f"{rec_data['buffer']:.1f}% buffer, sentiment {rec_data['sentiment']}. "
                f"Technicals — {rec_data['technicals']}.")


def _target_yield(ym: Dict[str, Any], rules) -> float:
    """The annualized figure the >10% gate uses: flat premium yield or return-if-assigned."""
    if rules.yield_target_metric == "flat":
        return float(ym.get("aroc_if_flat_percent", 0.0))
    return float(ym.get("aroc_if_assigned_percent", 0.0))


def _build_recommendation(
    c: Dict[str, Any], report: Dict[str, Any], target: float, llm: LocalLLM, rules
) -> Tuple[float, Recommendation]:
    """Score one candidate, assign a grade, write the LLM rationale, and assemble
    the Recommendation. Returns (composite_score, recommendation)."""
    contract = c.get("contract", {})
    ym = c.get("yield_metrics", {})
    iv_ratio = float(c.get("iv_rank", {}).get("iv_to_hv_ratio", 1.0) or 1.0)
    prob_keep = float(c.get("greeks", {}).get(
        "prob_expire_otm_percent", (1.0 - contract.get("delta", 0.0)) * 100.0))
    buffer = float(ym.get("downside_buffer_percent", 0.0))

    scoring = eng.score_covered_call_candidate(
        target, iv_ratio, report["sentiment_score"], buffer, prob_keep, rules.score_weights)
    grade = eng.grade_from_score(scoring["score"], rules.grade_thresholds)
    technicals = _fmt_technicals(c.get("snapshot", {}), c.get("trend", {}))
    news_titles = [s.get("title") for s in (report.get("sources") or []) if s.get("title")]
    rationale = _llm_rationale(llm, {
        "symbol": c["symbol"], "strike": contract.get("strike"),
        "expiration": str(contract.get("expiration_key", "")).split(":")[0],
        "dte": contract.get("days_to_expiration"), "delta": contract.get("delta", 0.0),
        "annualized_yield": target, "buffer": buffer, "iv_ratio": iv_ratio,
        "sentiment": report["sentiment"], "prob_keep": prob_keep, "grade": grade,
        "technicals": technicals, "news_rationale": report.get("rationale", ""),
        "news_titles": news_titles,
    })
    rec: Recommendation = {
        "symbol": c["symbol"], "grade": grade, "score": scoring["score"],
        "underlying_price": round(float(c.get("underlying_price", 0.0) or 0.0), 2),
        "annualized_yield_percent": round(target, 2), "contract": contract, "yield_metrics": ym,
        "sentiment": report["sentiment"], "iv_to_hv_ratio": round(iv_ratio, 2),
        "prob_keep_premium_percent": round(prob_keep, 1),
        "earnings_known": report.get("earnings_known", True),
        "earnings_date": report.get("earnings_date"), "sources": report.get("sources", []),
        "rationale": rationale, "score_components": scoring["components"],
    }
    return scoring["score"], rec


def _publish(top: List[Recommendation], state: ScreenerState, all_rejected: list,
             notifier: Optional[DiscordNotifier]) -> Dict[str, Any]:
    """Format the HITL summary, persist run artifacts (full rejection trail
    included), and send to Discord. Returns the state-update fields."""
    run_id = state.get("run_id", "")
    summary = format_recommendations(top, run_id=run_id, account_cash=state.get("account_cash"))
    run_paths = save_run(run_id, summary, top, run_timestamp=state.get("run_timestamp", ""),
                         rejected=all_rejected)
    notified, errors = False, []
    if notifier is not None and top:
        try:
            notified = notifier.send(summary)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Discord notify failed: {exc}")
    logger.info("Risk Manager surfaced %d candidates (notified=%s)", len(top), notified)
    out: Dict[str, Any] = {"discord_summary": summary, "notified": notified, "run_log_paths": run_paths}
    if errors:
        out["errors"] = errors
    return out


def build_risk_manager_node(
    llm: LocalLLM,
    notifier: Optional[DiscordNotifier] = None,
    rules=default_rules,
) -> Callable[[ScreenerState], dict]:
    """Return a Risk Manager node bound to the LLM, an optional notifier, rules."""

    def risk_node(state: ScreenerState) -> dict:
        quant = state.get("quant_candidates") or []
        news_by_sym = {r["symbol"]: r for r in (state.get("news_reports") or [])}

        scored: List[Tuple[float, Recommendation]] = []
        rejections = []

        for c in quant:
            report = news_by_sym.get(c["symbol"])
            # Skip names that weren't news-screened or failed news (already in trail).
            if report is None or not report.get("passes_news"):
                continue

            target = _target_yield(c.get("yield_metrics", {}), rules)
            if target < rules.min_annualized_yield_pct:  # hard >10% gate
                rejections.append(reject(c["symbol"], "RISK_MANAGER",
                    f"Annualized yield {target:.1f}% < {rules.min_annualized_yield_pct:.0f}% target "
                    f"(metric={rules.yield_target_metric})."))
                continue

            score, rec = _build_recommendation(c, report, target, llm, rules)
            scored.append((score, rec))
            logger.info("Risk grade %s=%s score %.0f (yield %.1f%%)",
                        c["symbol"], rec["grade"], score, target)

        scored.sort(key=lambda t: t[0], reverse=True)
        top = [rec for _, rec in scored[: rules.top_n_candidates]]

        all_rejected = (state.get("rejected") or []) + rejections
        out = {"recommendations": top, "rejected": rejections}
        out.update(_publish(top, state, all_rejected, notifier))
        return out

    return risk_node
