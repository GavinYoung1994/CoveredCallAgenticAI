"""Offline tests for the Massive (Polygon-compatible) market-data client.

Every HTTP call is served by an httpx.MockTransport, so no network or API key is
needed. The key assertions are that the ADAPTERS produce the exact shapes the
deterministic math engine already consumes (candles with 'close', the option
chain as 'callExpDateMap' that find_optimal_covered_call can parse).
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.data.massive_client import MassiveClient
from app.data.massive_earnings import MassiveEarningsClient
from app.engine import math_engine as eng

CLOCK = date(2026, 8, 3)


def _candles(n=80, start=100.0):
    # Gentle uptrend so calculate_technical_indicators has clean data.
    out = []
    for i in range(n):
        c = start + i * 0.5
        out.append({"o": c - 0.2, "h": c + 0.4, "l": c - 0.5, "c": c, "v": 1_000_000 + i,
                    "t": 1_700_000_000_000 + i * 86_400_000, "n": 5000})
    return out


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    params = dict(request.url.params)

    # Batch snapshot
    if path == "/v2/snapshot/locale/us/markets/stocks/tickers":
        syms = params.get("tickers", "").split(",")
        tickers = [{"ticker": s, "lastTrade": {"p": 150.0 + i}, "day": {"c": 149.0, "v": 5_000_000},
                    "prevDay": {"c": 148.0}} for i, s in enumerate(syms) if s]
        return httpx.Response(200, json={"status": "OK", "tickers": tickers})

    # Single snapshot
    if path.startswith("/v2/snapshot/locale/us/markets/stocks/tickers/"):
        sym = path.rsplit("/", 1)[-1]
        return httpx.Response(200, json={"status": "OK", "ticker": {
            "ticker": sym, "lastTrade": {"p": 152.5}, "day": {"c": 151.0, "v": 4_000_000}}})

    # Aggregates (history)
    if path.startswith("/v2/aggs/ticker/"):
        return httpx.Response(200, json={"ticker": "X", "status": "OK", "results": _candles()})

    # Option chain snapshot
    if path.startswith("/v3/snapshot/options/"):
        # Two strikes at a ~33-DTE expiration: delta 0.35 (in band) and 0.55 (out).
        results = [
            {"details": {"ticker": "O:X260905C00170000", "strike_price": 170.0,
                         "expiration_date": "2026-09-05", "contract_type": "call"},
             "greeks": {"delta": 0.35, "gamma": 0.01, "theta": -0.05, "vega": 0.2},
             "implied_volatility": 0.42, "open_interest": 1200,
             "day": {"volume": 800, "close": 4.1},
             "last_quote": {"bid": 4.0, "ask": 4.2, "midpoint": 4.1}},
            {"details": {"ticker": "O:X260905C00160000", "strike_price": 160.0,
                         "expiration_date": "2026-09-05", "contract_type": "call"},
             "greeks": {"delta": 0.55}, "implied_volatility": 0.40, "open_interest": 900,
             "day": {"volume": 500, "close": 7.0},
             "last_quote": {"bid": 6.9, "ask": 7.1, "midpoint": 7.0}},
            # Off-hours contract: NBBO quote zeroed → premium must fall back to day.close.
            {"details": {"ticker": "O:X260905C00180000", "strike_price": 180.0,
                         "expiration_date": "2026-09-05", "contract_type": "call"},
             "greeks": {"delta": 0.22}, "implied_volatility": 0.44, "open_interest": 600,
             "day": {"volume": 300, "close": 2.35},
             "last_quote": {"bid": 0.0, "ask": 0.0, "midpoint": 0.0}},
        ]
        return httpx.Response(200, json={"status": "OK", "results": results})

    # Option contracts (expirations / optionable)
    if path == "/v3/reference/options/contracts":
        return httpx.Response(200, json={"status": "OK", "results": [
            {"ticker": "O:X...", "expiration_date": "2026-09-05", "strike_price": 170.0, "contract_type": "call"},
            {"ticker": "O:X...", "expiration_date": "2026-10-17", "strike_price": 175.0, "contract_type": "call"}]})

    # Indicators
    if path.startswith("/v1/indicators/"):
        return httpx.Response(200, json={"status": "OK", "results": {"values": [
            {"timestamp": 1_700_000_000_000, "value": 55.3},
            {"timestamp": 1_699_913_600_000, "value": 54.1}]}})

    # Earnings
    if path == "/benzinga/v1/earnings":
        return httpx.Response(200, json={"status": "OK", "results": [
            {"ticker": params.get("ticker"), "date": "2026-09-12", "date_status": "projected"}]})

    return httpx.Response(404, json={"error": f"unhandled {path}"})


def _client() -> MassiveClient:
    return MassiveClient(api_key="k", base_url="https://mock",
                         http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
                         clock=lambda: CLOCK)


def test_quotes_and_fundamentals():
    c = _client()
    payload = c.get_quotes_chunked(["AAPL", "MSFT"])
    fund = c.extract_fundamentals(payload, "AAPL")
    assert fund["last_price"] == 150.0 and fund["symbol"] == "AAPL"
    assert fund["avg_daily_volume"] == 5_000_000


def test_single_quote_schwab_shape():
    c = _client()
    q = c.get_quote("NVDA")
    # Defense reads payload[sym]["quote"]["lastPrice"].
    assert q["NVDA"]["quote"]["lastPrice"] == 152.5


def test_price_history_feeds_engine():
    c = _client()
    hist = c.get_price_history("AAPL", period_type="month", period=6, frequency_type="daily")
    candles = hist["candles"]
    assert len(candles) == 80 and "close" in candles[0]
    ind = eng.calculate_technical_indicators(candles, trend_lookback_days=20)
    assert "error" not in ind
    assert ind["trend_analysis"]["detected_trend"].startswith("Upward")


def test_option_chain_maps_and_engine_selects_in_band():
    c = _client()
    chain = c.get_option_chain("X", contract_type="CALL", from_date="2026-09-01", to_date="2026-09-30")
    assert "callExpDateMap" in chain
    # Key is "<exp>:<dte>"; 2026-09-05 is 33 days after the 2026-08-03 clock.
    assert "2026-09-05:33" in chain["callExpDateMap"]
    best = eng.find_optimal_covered_call(chain, target_delta=0.35, delta_band=(0.30, 0.40),
                                         min_dte=30, max_dte=45)
    assert best["strike"] == 170.0 and best["in_delta_band"] is True
    # IV decimal 0.42 → percent 42.0 for the engine.
    assert abs(best["volatility"] - 42.0) < 1e-6
    assert best["mark"] == 4.1
    # Off-hours contract (bid/ask/midpoint all 0) → mark falls back to day.close.
    off_hours = chain["callExpDateMap"]["2026-09-05:33"]["180.0"][0]
    assert off_hours["mark"] == 2.35


def test_option_expirations_and_optionable():
    c = _client()
    exps = c.get_option_expirations("X")["expirations"]
    assert exps == ["2026-09-05", "2026-10-17"]
    assert c.is_optionable("X") is True


def test_technical_indicator_endpoints():
    c = _client()
    rsi = c.get_rsi("AAPL", window=14)
    assert rsi["indicator"] == "rsi" and rsi["latest"] == 55.3
    sma = c.get_sma("AAPL", window=50)
    assert sma["latest"] == 55.3 and len(sma["values"]) == 2


def test_earnings_next_date():
    e = MassiveEarningsClient(api_key="k", base_url="https://mock",
                              http_client=httpx.Client(transport=httpx.MockTransport(_handler)))
    assert e.get_next_earnings_date("AAPL", "2026-08-03", "2026-12-01") == "2026-09-12"


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
