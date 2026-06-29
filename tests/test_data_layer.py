"""Offline tests for the data layer (Schwab + massive.com news clients).

We inject ``httpx.MockTransport`` so the clients hit fixture handlers instead of
the network, and inject a stub Schwab token provider so no credentials are
needed. The rate limiter is tested with a fake clock — no real sleeping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.data.schwab_client import SchwabClient
from app.data.news_client import NewsClient, RateLimiter


# ── Schwab fixtures ───────────────────────────────────────────────────
def _schwab_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    # Auth header must be present (proves token provider wired in).
    assert request.headers.get("Authorization") == "Bearer test-token"
    if path.endswith("/quotes"):
        return httpx.Response(200, json={
            "AAPL": {
                "assetMainType": "EQUITY",
                "quote": {"lastPrice": 190.0, "totalVolume": 55_000_000},
                "fundamental": {
                    "avg10DaysVolume": 60_000_000,
                    "avg1YearVolume": 58_000_000,
                    "divYield": 2.4,
                    "divAmount": 0.96,
                    "nextDivExDate": "2026-08-09",
                    "peRatio": 28.5,
                },
            }
        })
    if path.endswith("/expirationchain"):
        return httpx.Response(200, json={
            "expirationList": [
                {"expirationDate": "2026-07-17", "daysToExpiration": 25},
                {"expirationDate": "2026-08-21", "daysToExpiration": 60},
            ]
        })
    if path.endswith("/pricehistory"):
        return httpx.Response(200, json={"candles": [{"close": 100.0 + i} for i in range(5)]})
    return httpx.Response(404, json={"error": "not found"})


def _schwab_client() -> SchwabClient:
    transport = httpx.MockTransport(_schwab_handler)
    return SchwabClient(
        token_provider=lambda: "test-token",
        http_client=httpx.Client(transport=transport),
        base_url="https://mock.schwab/marketdata/v1",
    )


def test_schwab_get_quotes_and_fundamentals():
    client = _schwab_client()
    payload = client.get_quotes(["AAPL"])
    fund = client.extract_fundamentals(payload, "AAPL")
    assert fund["last_price"] == 190.0
    assert fund["avg_daily_volume"] == 60_000_000
    assert fund["dividend_yield_percent"] == 2.4
    assert fund["asset_type"] == "EQUITY"


def test_schwab_extract_fundamentals_missing_safe():
    # Symbol absent from payload → defaults, no crash.
    client = _schwab_client()
    fund = client.extract_fundamentals({}, "TSLA")
    assert fund["last_price"] == 0.0 and fund["avg_daily_volume"] == 0


def test_schwab_is_optionable():
    client = _schwab_client()
    assert client.is_optionable("AAPL") is True


def test_schwab_price_history():
    client = _schwab_client()
    hist = client.get_price_history("AAPL")
    assert len(hist["candles"]) == 5


# ── News fixtures ─────────────────────────────────────────────────────
def _news_handler(request: httpx.Request) -> httpx.Response:
    assert request.url.params.get("ticker") == "AAPL"
    return httpx.Response(200, json={
        "status": "OK",
        "count": 2,
        "results": [
            {
                "id": "a1",
                "title": "Apple beats earnings expectations",
                "description": "Strong iPhone sales drive a record quarter.",
                "article_url": "https://news.example/apple-beats",
                "published_utc": "2026-06-20T12:00:00Z",
                "publisher": {"name": "Example Wire"},
                "tickers": ["AAPL"],
                "insights": [
                    {"ticker": "AAPL", "sentiment": "positive",
                     "sentiment_reasoning": "Earnings beat."},
                    {"ticker": "MSFT", "sentiment": "neutral", "sentiment_reasoning": "Mentioned."},
                ],
            },
            {
                "id": "a2",
                "title": "Analysts mixed on valuation",
                "description": "Some caution on the run-up.",
                "amp_url": "https://news.example/amp/val",
                "published_utc": "2026-06-19T09:00:00Z",
                "publisher": {"name": "Example Daily"},
                "tickers": ["AAPL"],
                "insights": [],
            },
        ],
    })


def _news_client(limiter=None, fetch_content=False) -> NewsClient:
    transport = httpx.MockTransport(_news_handler)
    return NewsClient(
        api_key="test-key",
        base_url="https://mock.massive",
        http_client=httpx.Client(transport=transport),
        rate_limiter=limiter,
        fetch_content=fetch_content,
    )


def test_news_get_headlines_normalizes():
    client = _news_client()
    headlines = client.get_headlines("AAPL", limit=5)
    assert len(headlines) == 2
    first = headlines[0]
    assert first["title"].startswith("Apple beats")
    assert first["url"] == "https://news.example/apple-beats"
    assert first["publisher"] == "Example Wire"
    # Per-ticker sentiment is picked out of insights[] (AAPL, not MSFT).
    assert first["api_sentiment"]["sentiment"] == "positive"
    # Second article uses amp_url fallback and has no sentiment.
    assert headlines[1]["url"].endswith("/amp/val")
    assert headlines[1]["api_sentiment"] is None


def test_rate_limiter_throttles_with_fake_clock():
    # Fake clock: time advances only when we sleep. 2 calls / 10s window.
    clock = {"t": 0.0}
    slept = []

    def fake_time():
        return clock["t"]

    def fake_sleep(secs):
        slept.append(secs)
        clock["t"] += secs  # advancing the clock simulates time passing

    limiter = RateLimiter(max_calls=2, period=10.0, time_func=fake_time, sleep_func=fake_sleep)
    assert limiter.acquire() == 0.0   # call 1, no wait
    assert limiter.acquire() == 0.0   # call 2, no wait
    waited = limiter.acquire()        # call 3 → must wait ~10s
    assert waited == 10.0 and slept == [10.0]


def test_news_publisher_allowlist_filters():
    # With an allowlist, only matching publishers are returned (case-insensitive).
    transport = httpx.MockTransport(_news_handler)
    client = NewsClient(api_key="test-key", base_url="https://mock.massive",
                        http_client=httpx.Client(transport=transport), fetch_content=False,
                        allowed_publishers=["example wire"])  # matches "Example Wire" only
    headlines = client.get_headlines("AAPL", limit=5)
    assert len(headlines) == 1
    assert headlines[0]["publisher"] == "Example Wire"


def test_news_empty_allowlist_accepts_all():
    transport = httpx.MockTransport(_news_handler)
    client = NewsClient(api_key="test-key", base_url="https://mock.massive",
                        http_client=httpx.Client(transport=transport), fetch_content=False,
                        allowed_publishers=[])
    assert len(client.get_headlines("AAPL", limit=5)) == 2


def test_news_disallowed_publishers_filters_out():
    # Block "Example Daily" → only "Example Wire" remains.
    transport = httpx.MockTransport(_news_handler)
    client = NewsClient(api_key="test-key", base_url="https://mock.massive",
                        http_client=httpx.Client(transport=transport), fetch_content=False,
                        disallowed_publishers=["example daily"])
    headlines = client.get_headlines("AAPL", limit=5)
    assert [h["publisher"] for h in headlines] == ["Example Wire"]


def test_news_blocklist_takes_precedence_over_allowlist():
    # Allow both, but block one → the blocked one is still dropped.
    transport = httpx.MockTransport(_news_handler)
    client = NewsClient(api_key="test-key", base_url="https://mock.massive",
                        http_client=httpx.Client(transport=transport), fetch_content=False,
                        allowed_publishers=["example wire", "example daily"],
                        disallowed_publishers=["example wire"])
    headlines = client.get_headlines("AAPL", limit=5)
    assert [h["publisher"] for h in headlines] == ["Example Daily"]


def test_news_client_respects_rate_limit():
    # 5 calls allowed instantly, 6th would block — verify via fake clock.
    clock = {"t": 0.0}
    waits = []
    limiter = RateLimiter(
        max_calls=5, period=60.0,
        time_func=lambda: clock["t"],
        sleep_func=lambda s: (waits.append(s), clock.__setitem__("t", clock["t"] + s)),
    )
    client = _news_client(limiter=limiter)
    for _ in range(5):
        client.get_headlines("AAPL")
    assert waits == []          # first 5 free
    client.get_headlines("AAPL")  # 6th
    assert len(waits) == 1 and waits[0] == 60.0


def test_html_to_text_strips_tags_and_scripts():
    from app.data.news_client import html_to_text
    html = "<html><head><style>.x{}</style></head><body><script>bad()</script>" \
           "<h1>Big&nbsp;News</h1><p>Filed for <b>bankruptcy</b> &amp; more.</p></body></html>"
    text = html_to_text(html)
    assert "Big News" in text and "bankruptcy" in text
    assert "bad()" not in text and ".x{}" not in text   # script/style dropped
    assert "&amp;" not in text and "&" in text           # entity decoded


def test_news_fetches_full_article_content():
    # The API call (has ?ticker) returns the article list; any other GET is the
    # article body fetch and returns HTML.
    def handler(request):
        if request.url.params.get("ticker"):
            return httpx.Response(200, json={"results": [
                {"title": "Acme update", "description": "summary",
                 "article_url": "https://pub.example/acme",
                 "publisher": {"name": "Wire"}, "tickers": ["ACME"], "insights": []}]})
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text="<html><body><script>x()</script><p>Acme filed for "
                                   "<b>bankruptcy</b> today.</p></body></html>")
    client = NewsClient(api_key="k", base_url="https://mock", fetch_content=True,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    headlines = client.get_headlines("ACME", limit=1)
    assert headlines[0]["content"]                       # body fetched
    assert "bankruptcy" in headlines[0]["content"]
    assert "x()" not in headlines[0]["content"]          # script stripped


# ── self-running harness ──────────────────────────────────────────────
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  ❌ {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  💥 {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    sys.exit(0 if passed == len(tests) else 1)
