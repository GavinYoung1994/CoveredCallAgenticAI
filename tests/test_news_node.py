"""Tests for the News/Sentiment node.

Real NewsClient + EarningsClient over MockTransport; the LLM is a fake backend
that returns a sentiment verdict keyed off the symbol in the prompt.
"""

import dataclasses
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import rules as base_rules
from app.data.news_client import NewsClient
from app.data.earnings_client import EarningsClient
from app.llm import LocalLLM
from app.nodes.news import build_news_node
from app.state import new_screener_state


# ── fake LLM backend: sentiment by symbol ─────────────────────────────
def _fake_llm_backend(msgs):
    content = " ".join(m["content"] for m in msgs)
    m = re.search(r"Stock:\s*([A-Z]+)", content)
    sym = m.group(1) if m else "?"
    if sym == "BANKRUPT":
        return '{"sentiment": "VERY_NEGATIVE", "catastrophic_risk": true, "rationale": "Chapter 11 filing rumored."}'
    if sym == "NEG":
        return '{"sentiment": "NEGATIVE", "catastrophic_risk": false, "rationale": "Soft guidance."}'
    return '{"sentiment": "POSITIVE", "catastrophic_risk": false, "rationale": "Solid quarter."}'


# ── earnings calendar fixture ─────────────────────────────────────────
def _earnings_handler(request: httpx.Request) -> httpx.Response:
    sym = request.url.params.get("symbol")
    if sym == "EARN":      # earnings BEFORE the 2026-07-28 expiration → guardrail
        return httpx.Response(200, json={"earningsCalendar": [{"date": "2026-07-15", "symbol": sym}]})
    if sym == "UNKNOWN":   # no scheduled earnings → unknown → flag but allow
        return httpx.Response(200, json={"earningsCalendar": []})
    # everyone else: earnings AFTER expiration → known & clear
    return httpx.Response(200, json={"earningsCalendar": [{"date": "2026-09-01", "symbol": sym}]})


# ── news headlines fixture (same shape as massive.com) ────────────────
def _news_handler(request: httpx.Request) -> httpx.Response:
    sym = request.url.params.get("ticker")
    return httpx.Response(200, json={"results": [
        {"title": f"{sym} update", "description": "Something happened.",
         "article_url": "https://x/y", "published_utc": "2026-06-20T00:00:00Z",
         "publisher": {"name": "Wire"}, "tickers": [sym], "insights": []},
    ]})


def _news_client():
    return NewsClient(api_key="k", base_url="https://mock", fetch_content=False,
                      http_client=httpx.Client(transport=httpx.MockTransport(_news_handler)))


def _earnings_client():
    return EarningsClient(api_key="k",
                          http_client=httpx.Client(transport=httpx.MockTransport(_earnings_handler)))


def _qc(sym, aroc=20.0, exp="2026-07-28"):
    return {"symbol": sym, "underlying_price": 100.0,
            "contract": {"symbol": f"{sym}_C", "strike": 105.0, "expiration_key": f"{exp}:35"},
            "yield_metrics": {"aroc_if_assigned_percent": aroc}}


def _run(quant_candidates, rules=base_rules):
    node = build_news_node(_news_client(), _earnings_client(),
                           LocalLLM(backend=_fake_llm_backend), rules=rules, today=date(2026, 6, 23))
    state = new_screener_state(watchlist=[], account_cash=0, run_id="r", run_timestamp="t")
    state["quant_candidates"] = quant_candidates
    return node(state)


def _report(out, sym):
    return next((r for r in out["news_reports"] if r["symbol"] == sym), None)


def test_news_positive_passes():
    out = _run([_qc("GOOD")])
    rep = _report(out, "GOOD")
    assert rep["passes_news"] is True
    assert rep["sentiment"] == "POSITIVE"
    assert rep["earnings_known"] is True and rep["earnings_date"] == "2026-09-01"
    # Sources are present for human review of the sentiment call.
    assert rep["sources"] and rep["sources"][0]["publisher"] == "Wire"
    assert rep["sources"][0]["url"] == "https://x/y"


def test_news_negative_rejected():
    out = _run([_qc("NEG")])
    assert _report(out, "NEG")["passes_news"] is False
    neg_rej = next(r for r in out["rejected"] if r["symbol"] == "NEG")
    assert "below floor" in neg_rej["reason"]
    # The rejection carries the reviewed sources so a human can double-check.
    assert neg_rej["sources"] and neg_rej["sources"][0]["title"] == "NEG update"


def test_news_catastrophic_rejected():
    out = _run([_qc("BANKRUPT")])
    assert _report(out, "BANKRUPT")["passes_news"] is False
    assert any("Catastrophic risk" in r["reason"] for r in out["rejected"])


def test_news_earnings_guardrail_rejects():
    out = _run([_qc("EARN")])
    # Rejected at the earnings guardrail BEFORE sentiment → no report emitted.
    assert _report(out, "EARN") is None
    assert any("earnings" in r["reason"].lower() for r in out["rejected"])


def test_news_unknown_earnings_flagged_but_allowed():
    out = _run([_qc("UNKNOWN")])
    rep = _report(out, "UNKNOWN")
    assert rep["passes_news"] is True            # allowed despite unknown earnings
    assert rep["earnings_known"] is False        # but clearly flagged
    assert rep["earnings_date"] is None


def test_news_caps_to_top_n_by_yield():
    rules = dataclasses.replace(base_rules, news_max_candidates=2)
    out = _run([_qc("A", aroc=30), _qc("B", aroc=20), _qc("C", aroc=10)], rules=rules)
    screened = {r["symbol"] for r in out["news_reports"]}
    assert screened == {"A", "B"}                # top-2 by AROC
    assert any(r["symbol"] == "C" and "Not news-screened" in r["reason"] for r in out["rejected"])


def test_keyword_scan_word_boundary():
    # 'sued' must NOT match inside 'issued'/'pursued'; a real 'lawsuit' does.
    from app.nodes.news import scan_catastrophic_keywords
    benign = [{"title": "Company issued new guidance; pursued growth", "description": "", "content": ""}]
    assert scan_catastrophic_keywords(benign, ["sued", "lawsuit"]) == []
    real = [{"title": "Company hit with a lawsuit", "description": "", "content": ""}]
    assert "lawsuit" in scan_catastrophic_keywords(real, ["lawsuit"])


def test_keywords_advisory_by_default_llm_decides():
    # LLM (reading content) says benign → NOT catastrophic by default, even though
    # a keyword appears. Keywords are surfaced as advisory only.
    from app.nodes.news import evaluate_news
    headlines = [{"title": "Acme mentioned in unrelated lawsuit coverage",
                  "description": "Acme thriving in AI.", "content": "No issues for Acme."}]
    benign_llm = LocalLLM(backend=lambda msgs:
                          '{"sentiment": "POSITIVE", "catastrophic_risk": false, "rationale": "strong"}')
    v = evaluate_news(benign_llm, "ACME", headlines, base_rules)
    assert v["catastrophic_risk"] is False            # LLM's content read wins
    assert "lawsuit" in v["catastrophic_keywords"]    # still surfaced as advisory


def test_keyword_veto_when_enabled():
    # With the opt-in veto, any keyword match forces catastrophic regardless of LLM.
    from app.nodes.news import evaluate_news
    rules = dataclasses.replace(base_rules, catastrophic_keyword_veto=True)
    headlines = [{"title": "Acme faces class action lawsuit", "description": "", "content": ""}]
    benign_llm = LocalLLM(backend=lambda msgs:
                          '{"sentiment": "POSITIVE", "catastrophic_risk": false, "rationale": "ok"}')
    v = evaluate_news(benign_llm, "ACME", headlines, rules)
    assert v["catastrophic_risk"] is True


def test_llm_flagged_catastrophic_still_rejects():
    # If the LLM itself flags catastrophic risk, it's catastrophic regardless of veto.
    from app.nodes.news import evaluate_news
    headlines = [{"title": "Acme files Chapter 11", "description": "", "content": "Bankruptcy filing."}]
    bad_llm = LocalLLM(backend=lambda msgs:
                       '{"sentiment": "VERY_NEGATIVE", "catastrophic_risk": true, "rationale": "Ch.11"}')
    assert evaluate_news(bad_llm, "ACME", headlines, base_rules)["catastrophic_risk"] is True


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
