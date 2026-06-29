"""End-to-end tests for the Tree-of-Thoughts defense-monitor graph.

Market inputs are passed in (so Schwab isn't hit), news via MockTransport, LLM
via a fake backend that answers both the sentiment and branch-decision prompts.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.data.schwab_client import SchwabClient
from app.data.news_client import NewsClient
from app.llm import LocalLLM
from app.notify.discord_webhook import DiscordNotifier
from app.graphs.defense_monitor import run_defense_monitor

POSITION = {
    "position_id": "KO_run1", "symbol": "KO", "stock_purchase_price": 100.0,
    "shares": 100, "short_call_strike": 105.0, "short_call_expiration": "2026-07-28",
    "original_premium": 2.0, "historical_premiums_collected": 2.0,
}


def _llm_backend(msgs):
    c = " ".join(m["content"] for m in msgs)
    if "recommended_branch" in c:                 # the branch-decision prompt
        return '{"recommended_branch": "B", "rationale": "Roll collects a healthy credit; news is benign."}'
    if "sentiment" in c.lower():                  # the sentiment prompt
        return '{"sentiment": "NEUTRAL", "catastrophic_risk": false, "rationale": "No major news."}'
    return "ok"


def _news_handler(request):
    sym = request.url.params.get("ticker")
    return httpx.Response(200, json={"results": [
        {"title": f"{sym} dips with sector", "description": "Broad weakness.",
         "article_url": "http://n/1", "publisher": {"name": "Wire"}, "tickers": [sym], "insights": []}]})


def _clients():
    schwab = SchwabClient(token_provider=lambda: "t",
                          http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404, json={}))),
                          base_url="https://mock/marketdata/v1")
    news = NewsClient(api_key="k", base_url="https://mock", fetch_content=False,
                      http_client=httpx.Client(transport=httpx.MockTransport(_news_handler)))
    return schwab, news


def _run(price, call_ask, roll, capture=None):
    schwab, news = _clients()
    notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/abc",
                               poster=(lambda u, p: capture.append(p["content"]) or 204) if capture is not None
                               else (lambda u, p: 204))
    return run_defense_monitor(
        POSITION, current_stock_price=price, current_call_ask=call_ask, roll_down_premium=roll,
        today=date(2026, 6, 24), run_id="test_def",
        schwab_client=schwab, news_client=news, llm=LocalLLM(backend=_llm_backend), notifier=notifier)


def test_breach_generates_branches_and_picks_roll():
    captured = []
    # Down 10% (<= -8% threshold). Roll credit positive (1.5 - 0.5).
    final = _run(price=90.0, call_ask=0.5, roll=1.5, capture=captured)
    assert final["breach_detected"] is True
    ba = final["branch_analysis"]
    assert ba["drop_percent"] == -10.0
    # Branch P&L computed deterministically.
    assert ba["branches"]["Branch_A_Liquidate"]["realized_cash_loss"] == -850.0
    assert ba["branches"]["Branch_B_Roll_Down"]["is_valid"] is True
    # Risk Manager chose B and notified the human.
    rec = final["defense_recommendation"]
    assert rec["branch"] == "B"
    assert final["notified"] is True
    assert "DECISION REQUIRED" in "\n".join(captured)


def _run_with(pos, price, call_ask=0.5, roll=1.5):
    schwab, news = _clients()
    notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/abc", poster=lambda u, p: 204)
    return run_defense_monitor(
        pos, current_stock_price=price, current_call_ask=call_ask, roll_down_premium=roll,
        today=date(2026, 6, 24), run_id="test_def",
        schwab_client=schwab, news_client=news, llm=LocalLLM(backend=_llm_backend), notifier=notifier)


def test_dynamic_breach_threshold_helper():
    from app.nodes.defense import _breach_threshold
    from app.config import rules
    assert _breach_threshold({"downside_buffer_percent": 4.0}, rules) == -4.0
    assert _breach_threshold({"downside_buffer_percent": None}, rules) == rules.downside_breach_pct
    assert _breach_threshold({}, rules) == rules.downside_breach_pct


def test_breach_uses_stored_buffer_no_breach_within_cushion():
    pos = dict(POSITION, downside_buffer_percent=12.0)   # big premium cushion
    final = _run_with(pos, 95.0)                          # -5% drop, inside 12% buffer
    assert final["breach_detected"] is False


def test_breach_uses_stored_buffer_breach_past_cushion():
    pos = dict(POSITION, downside_buffer_percent=3.0)    # thin cushion
    final = _run_with(pos, 95.0)                          # -5% drop, past 3% buffer
    assert final["breach_detected"] is True


def test_no_breach_skips_to_end():
    final = _run(price=99.0, call_ask=0.5, roll=1.5)  # only -1% drop
    assert final["breach_detected"] is False
    assert "defense_recommendation" not in final      # news + risk never ran


def test_roll_guardrail_overrides_to_hold():
    # LLM says "B" but the roll has no positive credit (roll 0.1 < buyback 0.5) →
    # the deterministic guardrail must override B → C.
    final = _run(price=90.0, call_ask=0.5, roll=0.1)
    assert final["branch_analysis"]["branches"]["Branch_B_Roll_Down"]["is_valid"] is False
    assert final["defense_recommendation"]["branch"] == "C"


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
