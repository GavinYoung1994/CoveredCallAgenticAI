"""Tests for the Quant node — drives SchwabClient over MockTransport.

Each test symbol is engineered to exercise a specific reject path or the happy
path. History + option-chain fixtures vary by the `symbol` query param.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataclasses

import httpx

from app.config import rules as base_rules
from app.data.schwab_client import SchwabClient
from app.nodes.quant import build_quant_node
from app.state import new_screener_state

# Pin rules so these tests are independent of live config edits (e.g. the user
# toggling require_rich_iv). The IV-richness tests below require it ON.
TEST_RULES = dataclasses.replace(base_rules, require_rich_iv=True, reject_downtrend=True,
                                 max_quant_candidates=0)

UPTREND = [{"close": 100 + i * 0.3} for i in range(90)]
DOWNTREND = [{"close": 130 - i * 0.3} for i in range(90)]
# Up-trending but with large daily swings → high realized volatility, so a tiny
# option IV (2%) is correctly judged "not rich" relative to it.
VOLATILE_UP = [{"close": 100 + i * 0.3 + (5 if i % 2 else -5)} for i in range(90)]


def _contract(delta, bid, ask, vol, oi=1000, volume=500, strike=105.0):
    return [{
        "symbol": f"OPT_{strike}", "delta": delta, "bid": bid, "ask": ask,
        "mark": round((bid + ask) / 2, 2), "totalVolume": volume,
        "openInterest": oi, "volatility": vol,
    }]


def _chain(strikes_map, dte=35):
    return {"callExpDateMap": {f"2026-07-28:{dte}": strikes_map}}


def _handler(request: httpx.Request) -> httpx.Response:
    sym = request.url.params.get("symbol")
    path = request.url.path
    if path.endswith("/pricehistory"):
        candles = {"DOWN": DOWNTREND, "LOWIV": VOLATILE_UP}.get(sym, UPTREND)
        return httpx.Response(200, json={"candles": candles})
    if path.endswith("/chains"):
        if sym == "WIDE":  # 50% bid-ask spread
            chain = _chain({"105.0": _contract(0.35, 1.0, 2.0, 25.0)})
        elif sym == "LOWIV":  # IV 2% below realized vol → not rich
            chain = _chain({"105.0": _contract(0.35, 2.0, 2.1, 2.0)})
        elif sym == "NOBAND":  # no delta in 0.30–0.40
            chain = _chain({"101.0": _contract(0.55, 2.0, 2.1, 25.0),
                            "120.0": _contract(0.12, 0.3, 0.35, 25.0)})
        elif sym == "THIN":  # tight spread but tiny open interest
            chain = _chain({"105.0": _contract(0.35, 2.0, 2.1, 25.0, oi=5)})
        else:  # GOOD
            chain = _chain({"105.0": _contract(0.35, 2.0, 2.1, 25.0)})
        return httpx.Response(200, json=chain)
    return httpx.Response(404, json={})


def _client() -> SchwabClient:
    return SchwabClient(
        token_provider=lambda: "t",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
        base_url="https://mock/marketdata/v1",
    )


def _run(symbols_prices, cash=100_000.0):
    scout = [{"symbol": s, "fundamentals": {"last_price": p}, "is_optionable": True}
             for s, p in symbols_prices]
    node = build_quant_node(_client(), rules=TEST_RULES, today=date(2026, 6, 23))
    state = new_screener_state(watchlist=[], account_cash=cash, run_id="r", run_timestamp="t")
    state["scout_candidates"] = scout
    return node(state)


def test_quant_happy_path():
    out = _run([("GOOD", 100.0)])
    assert len(out["quant_candidates"]) == 1
    c = out["quant_candidates"][0]
    assert c["symbol"] == "GOOD"
    assert c["contract"]["strike"] == 105.0
    assert 0.30 <= c["contract"]["delta"] <= 0.40
    assert c["yield_metrics"]["aroc_if_assigned_percent"] > 0
    assert c["greeks"]["prob_assignment_percent"] > 0      # BS greeks computed
    assert c["iv_rank"]["iv_to_hv_ratio"] > 1.1            # IV richer than realized
    assert c["trend"]["detected_trend"].startswith("Upward")


def test_quant_rejects_unaffordable():
    out = _run([("PRICEY", 2000.0)], cash=100_000.0)  # 100 shares = $200k > cash
    assert out["quant_candidates"] == []
    assert "Unaffordable" in out["rejected"][0]["reason"]


def test_quant_rejects_downtrend():
    out = _run([("DOWN", 100.0)])
    assert out["quant_candidates"] == []
    assert "Downtrend" in out["rejected"][0]["reason"]


def test_quant_rejects_wide_spread():
    out = _run([("WIDE", 100.0)])
    assert out["quant_candidates"] == []
    assert "Illiquid option" in out["rejected"][0]["reason"]


def test_quant_rejects_low_iv():
    out = _run([("LOWIV", 100.0)])
    assert out["quant_candidates"] == []
    assert "IV not rich" in out["rejected"][0]["reason"]


def test_quant_rejects_no_delta_band():
    out = _run([("NOBAND", 100.0)])
    assert out["quant_candidates"] == []
    assert "outside band" in out["rejected"][0]["reason"] or "No contract" in out["rejected"][0]["reason"]


def test_quant_rejects_thin_open_interest():
    out = _run([("THIN", 100.0)])
    assert out["quant_candidates"] == []
    assert "Open interest" in out["rejected"][0]["reason"]


def test_quant_mixed_batch_records_all():
    out = _run([("GOOD", 100.0), ("DOWN", 100.0), ("WIDE", 100.0)])
    assert [c["symbol"] for c in out["quant_candidates"]] == ["GOOD"]
    assert {r["symbol"] for r in out["rejected"]} == {"DOWN", "WIDE"}
    assert all(r["stage"] == "QUANT" for r in out["rejected"])


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
