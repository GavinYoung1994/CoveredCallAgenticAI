"""End-to-end integration test for the entry-screener LangGraph.

Everything is mocked (Schwab via MockTransport, news/earnings via MockTransport,
LLM via fake backend, Discord via capturing poster). We run a 2-symbol watchlist
where GOOD survives all four nodes and DOWN is rejected at Quant — verifying the
full Scout→Quant→News→Risk pipeline and the accumulating audit trail.
"""

import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.data.schwab_client import SchwabClient
from app.data.news_client import NewsClient
from app.data.earnings_client import EarningsClient
from app.llm import LocalLLM
from app.notify.discord_webhook import DiscordNotifier
from app.graphs.entry_screener import run_entry_screener

UPTREND = [{"close": 100 + i * 0.3} for i in range(90)]
DOWNTREND = [{"close": 130 - i * 0.3} for i in range(90)]


def _schwab_handler(request):
    p = request.url.path
    if p.endswith("/quotes"):
        syms = request.url.params.get("symbols", "").split(",")
        body = {s: {"assetMainType": "EQUITY",
                    "quote": {"lastPrice": 100.0, "totalVolume": 5_000_000},
                    "fundamental": {"avg10DaysVolume": 5_000_000, "divYield": 3.0}}
                for s in syms}
        return httpx.Response(200, json=body)
    if p.endswith("/pricehistory"):
        sym = request.url.params.get("symbol")
        return httpx.Response(200, json={"candles": DOWNTREND if sym == "DOWN" else UPTREND})
    if p.endswith("/chains"):
        contract = [{"symbol": "OPT", "delta": 0.35, "bid": 2.0, "ask": 2.1, "mark": 2.05,
                     "totalVolume": 500, "openInterest": 1000, "volatility": 25.0}]
        return httpx.Response(200, json={"callExpDateMap": {"2026-07-28:35": {"105.0": contract}}})
    return httpx.Response(404, json={})


def _news_handler(request):
    sym = request.url.params.get("ticker")
    return httpx.Response(200, json={"results": [
        {"title": f"{sym} steady", "description": "Normal trading.",
         "article_url": "http://n/1", "publisher": {"name": "Wire"}, "tickers": [sym], "insights": []}]})


def _earnings_handler(request):
    return httpx.Response(200, json={"earningsCalendar": [{"date": "2026-09-01"}]})  # clear


def _llm_backend(msgs):
    content = " ".join(m["content"] for m in msgs)
    if "catastrophic" in content.lower():       # News sentiment call
        m = re.search(r"Stock:\s*([A-Z]+)", content)
        return '{"sentiment": "POSITIVE", "catastrophic_risk": false, "rationale": "Healthy."}'
    return "Attractive annualized yield with a reasonable buffer; main risk is a sharp selloff."


def _mk(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_full_pipeline_end_to_end():
    captured = []
    notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/abc",
                               poster=lambda u, p: captured.append(p["content"]) or 204)
    final = run_entry_screener(
        watchlist=["GOOD", "DOWN"],
        account_cash=100_000.0,
        today=date(2026, 6, 23),
        run_id="test_e2e",
        schwab_client=SchwabClient(token_provider=lambda: "t", http_client=_mk(_schwab_handler),
                                   base_url="https://mock/marketdata/v1"),
        news_client=NewsClient(api_key="k", base_url="https://mock", http_client=_mk(_news_handler), fetch_content=False),
        earnings_client=EarningsClient(api_key="k", http_client=_mk(_earnings_handler)),
        llm=LocalLLM(backend=_llm_backend),
        notifier=notifier,
    )

    # GOOD survives all four nodes → exactly one recommendation.
    recs = final["recommendations"]
    assert [r["symbol"] for r in recs] == ["GOOD"], recs
    assert recs[0]["annualized_yield_percent"] >= 10.0
    assert recs[0]["rationale"].startswith("Attractive")

    # DOWN was rejected at QUANT (downtrend); audit trail spans stages.
    rej = {r["symbol"]: r["stage"] for r in final["rejected"]}
    assert rej.get("DOWN") == "QUANT"

    # Discord was notified and the run log persisted.
    assert final["notified"] is True
    assert "GOOD" in "\n".join(captured)
    assert "json" in final.get("run_log_paths", {})


def test_pipeline_empty_watchlist():
    final = run_entry_screener(
        watchlist=[], account_cash=100_000.0, today=date(2026, 6, 23), run_id="test_empty",
        schwab_client=SchwabClient(token_provider=lambda: "t", http_client=_mk(_schwab_handler),
                                   base_url="https://mock/marketdata/v1"),
        news_client=NewsClient(api_key="k", base_url="https://mock", http_client=_mk(_news_handler), fetch_content=False),
        earnings_client=EarningsClient(api_key="k", http_client=_mk(_earnings_handler)),
        llm=LocalLLM(backend=_llm_backend),
        notifier=DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/abc",
                                 poster=lambda u, p: 204),
    )
    assert final["recommendations"] == []


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
