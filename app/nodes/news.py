"""Node 3 — The News & Sentiment Agent (The Guardrail).

Job (design §3): make sure a candidate's rich IV isn't being driven by an
imminent catastrophic event (lawsuit, bankruptcy, fraud), and enforce the hard
earnings guardrail. This is the ONLY entry-pipeline node that uses the LLM, and
only for semantic work — sentiment classification, not math.

Per-candidate logic:
  1. Cost control: only the top-N Quant survivors by annualized yield are
     news-screened (protects the news API's 5/min free tier). The rest are
     recorded in the audit trail (no silent truncation).
  2. Earnings guardrail (hard): if a known earnings date falls on/before the
     option expiration → reject. If earnings is UNKNOWN → flag but allow.
  3. Sentiment: the LLM reads trimmed headlines and returns a structured verdict
     (label + catastrophic_risk flag). Reject on catastrophic risk or sentiment
     below the configured floor (default NEUTRAL).
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional

from app.config import rules as default_rules
from app.engine import math_engine as eng
from app.data.news_client import NewsClient
from app.data.earnings_client import EarningsClient
from app.llm import LocalLLM
from app.state import NewsReport, ScreenerState, reject, sentiment_score

logger = logging.getLogger("node.news")

_VALID_LABELS = {"VERY_NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "VERY_POSITIVE"}

def _sources(headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Concise, reviewable source list extracted from the fetched headlines."""
    return [
        {
            "title": h.get("title"),
            "publisher": h.get("publisher"),
            "url": h.get("url"),
            "published_utc": h.get("published_utc"),
        }
        for h in headlines
    ]


_SYSTEM_PROMPT = (
    "You are a financial risk analyst screening stocks for covered-call selling. "
    "Given recent news for a single stock — headline, summary, AND the article "
    "body when available — read the actual CONTENT (not just the headline) and "
    "assess the overall sentiment, then decide whether there is an imminent "
    "CATASTROPHIC risk event such as bankruptcy, fraud, accounting scandal, major "
    "lawsuit, regulatory/criminal action, delisting, or going-concern doubt. Be "
    "conservative: a covered-call seller mainly wants to avoid a collapse in the "
    "underlying. Headlines can mislead — base your judgment on the article text."
)

# Per-article body chars sent to the LLM (kept short for the context window even
# though the client may have fetched more).
_CONTENT_SNIPPET_CHARS = 1200


def _score_sentiment(llm: LocalLLM, symbol: str, headlines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ask the LLM for a structured sentiment verdict, reading article CONTENT
    (not just headlines) when available. Falls back to NEUTRAL on no news or
    invalid JSON."""
    if not headlines:
        return {"sentiment": "NEUTRAL", "catastrophic_risk": False,
                "rationale": "No recent news found."}

    blocks = []
    for i, h in enumerate(headlines, 1):
        api_sent = (h.get("api_sentiment") or {}).get("sentiment")
        hint = f" [provider sentiment: {api_sent}]" if api_sent else ""
        body = (h.get("content") or "").strip()
        body_line = f"\n  CONTENT: {body[:_CONTENT_SNIPPET_CHARS]}" if body else \
                    "\n  CONTENT: (full text unavailable — judge from headline/summary)"
        blocks.append(
            f"[{i}] {h.get('title', '')}{hint}\n  SUMMARY: {h.get('description', '') or ''}{body_line}")
    user = (
        f"Stock: {symbol}\nArticles:\n" + "\n\n".join(blocks) +
        "\n\nEvaluate the CONTENT of each article. Return JSON with keys: sentiment "
        "(one of VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, VERY_POSITIVE), "
        "catastrophic_risk (true/false), rationale (1-2 sentences citing the content)."
    )
    try:
        obj = llm.structured(_SYSTEM_PROMPT, user,
                             required_keys=["sentiment", "catastrophic_risk", "rationale"])
    except Exception as exc:  # noqa: BLE001 — never let the LLM crash the pipeline
        logger.warning("Sentiment LLM failed for %s: %s", symbol, exc)
        return {"sentiment": "NEUTRAL", "catastrophic_risk": False,
                "rationale": f"Sentiment unavailable ({exc}); defaulted NEUTRAL.",
                "llm_failed": True}
    label = str(obj.get("sentiment", "NEUTRAL")).upper()
    if label not in _VALID_LABELS:
        label = "NEUTRAL"
    obj["sentiment"] = label
    obj["catastrophic_risk"] = bool(obj.get("catastrophic_risk", False))
    return obj


def scan_catastrophic_keywords(headlines: List[Dict[str, Any]], keywords: List[str]) -> List[str]:
    """Scan title + summary + full body for catastrophic keywords, matching on
    WORD BOUNDARIES (so 'sued' doesn't match 'issued'/'pursued'). Returns the
    matched keywords. These are advisory — see ``evaluate_news``."""
    corpus = " ".join(
        f"{h.get('title', '')} {h.get('description', '') or ''} {h.get('content', '') or ''}"
        for h in headlines
    ).lower()
    matched = []
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", corpus):
            matched.append(kw)
    return sorted(set(matched))


def evaluate_news(llm: LocalLLM, symbol: str, headlines: List[Dict[str, Any]], rules) -> Dict[str, Any]:
    """LLM content assessment + an advisory keyword scan.

    The LLM — which reads the article CONTENT and associates events with THIS
    company — is the primary judge of catastrophic risk. Keyword matches are
    surfaced as advisory context and do NOT auto-reject, UNLESS
    ``catastrophic_keyword_veto`` is enabled (then any match is a hard veto).
    """
    verdict = _score_sentiment(llm, symbol, headlines)
    matched = scan_catastrophic_keywords(headlines, getattr(rules, "catastrophic_keywords", []))
    veto = getattr(rules, "catastrophic_keyword_veto", False)
    catastrophic = bool(verdict.get("catastrophic_risk")) or (bool(matched) and veto)
    return {
        "sentiment": verdict["sentiment"],
        "rationale": verdict.get("rationale", ""),
        "catastrophic_risk": catastrophic,
        "catastrophic_keywords": matched,   # advisory unless veto is on
    }


def _defer_beyond_cap(candidates: List[Dict[str, Any]], rules) -> tuple:
    """Cost control: news-screen only the top-N by annualized yield; record the
    rest as deferred rejections (no silent truncation). Returns (to_screen, rejections)."""
    ranked = sorted(
        candidates,
        key=lambda c: c.get("yield_metrics", {}).get("aroc_if_assigned_percent", 0.0),
        reverse=True,
    )
    to_screen = ranked[: rules.news_max_candidates]
    deferred = ranked[rules.news_max_candidates:]
    if deferred:
        logger.info("News: deferring %d candidates beyond top-%d (rate cap)",
                    len(deferred), rules.news_max_candidates)
    rejections = [
        reject(c["symbol"], "NEWS",
               f"Not news-screened: outside top-{rules.news_max_candidates} by annualized yield this run.")
        for c in deferred
    ]
    return to_screen, rejections


def _earnings_guardrail(earnings_client: EarningsClient, sym: str, exp_date: str, run_today: date):
    """Look up the next earnings date and apply the hard guardrail.
    Returns (earnings_date, guard_dict, error_or_None)."""
    earnings_date, err = None, None
    try:
        to_d = exp_date or (run_today + timedelta(days=60)).isoformat()
        earnings_date = earnings_client.get_next_earnings_date(sym, run_today.isoformat(), to_d)
    except Exception as exc:  # noqa: BLE001
        err = f"News earnings lookup failed for {sym}: {exc}"
    guard = eng.is_earnings_within_cycle(earnings_date, exp_date)
    return earnings_date, guard, err


def _sentiment_verdict_to_reason(verdict: Dict[str, Any], score: int, floor_score: int, rules):
    """Decide pass/fail from the news verdict. Returns (passes, fail_reason_or_None)."""
    if verdict["catastrophic_risk"]:
        kw = verdict["catastrophic_keywords"]
        note = f" (keywords: {', '.join(kw)})" if kw else ""
        return False, f"Catastrophic risk flagged{note}: {verdict.get('rationale', '')}"
    if score < floor_score:
        return False, f"Sentiment {verdict['sentiment']} below floor {rules.min_acceptable_sentiment}."
    return True, None


def _screen_candidate(c, *, news_client, earnings_client, llm, rules, run_today, floor_score):
    """Screen one Quant candidate through earnings + sentiment. Returns
    (report_or_None, rejection_or_None, errors). A report is omitted only when
    the earnings guardrail disqualifies before sentiment is assessed."""
    sym = c["symbol"]
    errors: List[str] = []
    exp_date = str(c.get("contract", {}).get("expiration_key", "")).split(":")[0]

    earnings_date, guard, err = _earnings_guardrail(earnings_client, sym, exp_date, run_today)
    if err:
        errors.append(err)
    if guard.get("earnings_known") and guard.get("disqualify"):
        return None, reject(sym, "NEWS", guard["reason"]), errors

    headlines: List[Dict[str, Any]] = []
    try:
        headlines = news_client.get_headlines(sym, limit=rules.headlines_per_symbol)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"News fetch failed for {sym}: {exc}")

    verdict = evaluate_news(llm, sym, headlines, rules)
    score = sentiment_score(verdict["sentiment"])
    passes, fail_reason = _sentiment_verdict_to_reason(verdict, score, floor_score, rules)
    sources = _sources(headlines)

    report: NewsReport = {
        "symbol": sym, "sentiment": verdict["sentiment"], "sentiment_score": score,
        "rationale": verdict.get("rationale", ""), "headlines_checked": headlines, "sources": sources,
        "earnings_date": earnings_date, "earnings_disqualifies": False,
        "earnings_known": bool(guard.get("earnings_known")),
        "catastrophic_risk": verdict["catastrophic_risk"],
        "catastrophic_keywords": verdict["catastrophic_keywords"], "passes_news": passes,
    }
    if not passes:
        logger.info("News REJECT %s: %s (%d sources)", sym, fail_reason, len(sources))
        return report, reject(sym, "NEWS", fail_reason, sources=sources), errors
    flag = "" if report["earnings_known"] else " [earnings UNKNOWN — flagged]"
    logger.info("News PASS %s: %s%s", sym, verdict["sentiment"], flag)
    return report, None, errors


def build_news_node(
    news_client: NewsClient,
    earnings_client: EarningsClient,
    llm: LocalLLM,
    rules=default_rules,
    today: Optional[date] = None,
) -> Callable[[ScreenerState], dict]:
    """Return a News node bound to its clients + LLM + rules."""

    floor_score = sentiment_score(rules.min_acceptable_sentiment)

    def news_node(state: ScreenerState) -> dict:
        candidates = state.get("quant_candidates") or []
        run_today = today or date.today()

        to_screen, rejections = _defer_beyond_cap(candidates, rules)
        reports: List[NewsReport] = []
        errors: List[str] = []

        logger.info("News screening %d candidates", len(to_screen))
        for c in to_screen:
            report, rejection, errs = _screen_candidate(
                c, news_client=news_client, earnings_client=earnings_client,
                llm=llm, rules=rules, run_today=run_today, floor_score=floor_score)
            errors.extend(errs)
            if report is not None:
                reports.append(report)
            if rejection is not None:
                rejections.append(rejection)

        out = {"news_reports": reports, "rejected": rejections}
        if errors:
            out["errors"] = errors
        return out

    return news_node
