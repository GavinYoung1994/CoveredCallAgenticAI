"""Tests for the Scout node — drives a SchwabClient over MockTransport.

We craft a watchlist where each symbol fails a *different* filter, so we verify
every rejection path plus the happy path, and confirm the audit trail is built.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataclasses

import httpx

from app.config import rules as base_rules
from app.data.schwab_client import SchwabClient
from app.nodes.scout import build_scout_node, load_watchlist
from app.state import new_screener_state

# Per-symbol fundamentals: GOOD passes all; others each fail one filter.
QUOTES = {
    "GOOD": {  # passes everything
        "assetMainType": "EQUITY",
        "quote": {"lastPrice": 100.0, "totalVolume": 5_000_000},
        "fundamental": {"avg10DaysVolume": 5_000_000, "divYield": 3.1},
    },
    "ILLQ": {  # fails liquidity (avg vol below 1M)
        "assetMainType": "EQUITY",
        "quote": {"lastPrice": 50.0, "totalVolume": 100_000},
        "fundamental": {"avg10DaysVolume": 200_000, "divYield": 4.0},
    },
    "LOWDIV": {  # fails dividend (<2%)
        "assetMainType": "EQUITY",
        "quote": {"lastPrice": 200.0, "totalVolume": 9_000_000},
        "fundamental": {"avg10DaysVolume": 9_000_000, "divYield": 0.5},
    },
    "NOOPT": {  # passes liquidity+dividend but is not optionable
        "assetMainType": "EQUITY",
        "quote": {"lastPrice": 80.0, "totalVolume": 3_000_000},
        "fundamental": {"avg10DaysVolume": 3_000_000, "divYield": 2.5},
    },
    "DEAD": {  # no market data
        "assetMainType": "EQUITY",
        "quote": {"lastPrice": 0.0, "totalVolume": 0},
        "fundamental": {"avg10DaysVolume": 0, "divYield": 0.0},
    },
}


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/quotes"):
        syms = request.url.params.get("symbols", "").split(",")
        return httpx.Response(200, json={s: QUOTES[s] for s in syms if s in QUOTES})
    if path.endswith("/expirationchain"):
        sym = request.url.params.get("symbol")
        # NOOPT has no expirations; everyone else does.
        if sym == "NOOPT":
            return httpx.Response(200, json={"expirationList": []})
        return httpx.Response(200, json={"expirationList": [{"expirationDate": "2026-07-17"}]})
    return httpx.Response(404, json={})


def _client() -> SchwabClient:
    return SchwabClient(
        token_provider=lambda: "test-token",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
        base_url="https://mock/marketdata/v1",
    )


def _run_scout(symbols, rules=base_rules):
    node = build_scout_node(_client(), rules=rules)
    state = new_screener_state(
        watchlist=symbols, account_cash=100_000, run_id="r", run_timestamp="t",
    )
    return node(state)


# Rules variants for the two Scout modes.
PREFILTERED = dataclasses.replace(base_rules, watchlist_is_prefiltered=True)
FULL_FILTER = dataclasses.replace(base_rules, watchlist_is_prefiltered=False, require_optionable=True)


def test_scout_prefiltered_only_liveness():
    # Default (prefiltered) mode: only DEAD (no price) is dropped; the rest pass
    # even though they'd fail liquidity/dividend/optionable filters.
    out = _run_scout(["GOOD", "ILLQ", "LOWDIV", "NOOPT", "DEAD"], rules=PREFILTERED)
    passed = sorted(c["symbol"] for c in out["scout_candidates"])
    assert passed == ["GOOD", "ILLQ", "LOWDIV", "NOOPT"], passed
    dropped = {r["symbol"]: r["reason"] for r in out["rejected"]}
    assert list(dropped) == ["DEAD"] and "No live price" in dropped["DEAD"]


def test_scout_full_filter_mode():
    # Full-filter mode reproduces the strict screening on a raw watchlist.
    out = _run_scout(["GOOD", "ILLQ", "LOWDIV", "NOOPT", "DEAD"], rules=FULL_FILTER)
    passed = [c["symbol"] for c in out["scout_candidates"]]
    assert passed == ["GOOD"], passed
    by_symbol = {r["symbol"]: r["reason"] for r in out["rejected"]}
    assert "Illiquid" in by_symbol["ILLQ"]
    assert "Dividend yield" in by_symbol["LOWDIV"]
    assert "Not optionable" in by_symbol["NOOPT"]
    assert "No live price" in by_symbol["DEAD"]
    assert all(r["stage"] == "SCOUT" for r in out["rejected"])


def test_scout_empty_watchlist():
    out = _run_scout([])
    assert out["scout_candidates"] == []
    assert any("empty watchlist" in e.lower() for e in out["errors"])


def test_load_watchlist_reads_real_file():
    # The seeded watchlist.json should load and be all-uppercase symbols.
    # Robust to the user editing watchlist.json: just verify shape/format.
    syms = load_watchlist()
    assert isinstance(syms, list) and len(syms) >= 1
    assert all(isinstance(s, str) and s == s.upper() for s in syms)


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
