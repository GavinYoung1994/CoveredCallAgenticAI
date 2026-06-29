"""Tests for the Finnhub earnings client (offline via MockTransport)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.data.earnings_client import EarningsClient


def _handler(request: httpx.Request) -> httpx.Response:
    sym = request.url.params.get("symbol")
    if sym == "AAPL":
        return httpx.Response(200, json={"earningsCalendar": [
            {"date": "2026-08-01", "symbol": "AAPL", "hour": "amc"},
            {"date": "2026-07-20", "symbol": "AAPL", "hour": "bmo"},  # earlier
        ]})
    return httpx.Response(200, json={"earningsCalendar": []})  # no scheduled earnings


def _client(api_key="test-key"):
    return EarningsClient(
        api_key=api_key,
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )


def test_next_earnings_returns_earliest():
    c = _client()
    d = c.get_next_earnings_date("AAPL", "2026-06-23", "2026-08-30")
    assert d == "2026-07-20"  # earliest of the two


def test_no_earnings_returns_none():
    c = _client()
    assert c.get_next_earnings_date("KO", "2026-06-23", "2026-08-30") is None


def test_disabled_without_key():
    # No key → disabled → always None (earnings unknown), no HTTP call made.
    c = _client(api_key="")
    assert c.enabled is False
    assert c.get_next_earnings_date("AAPL", "2026-06-23", "2026-08-30") is None


def test_placeholder_key_disabled():
    c = _client(api_key="your-finnhub-api-key")
    assert c.enabled is False


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
