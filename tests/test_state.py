"""Tests for the LangGraph state schema and helpers.

TypedDicts are plain dicts at runtime, so we verify the constructors, the
sentiment mapping, and — importantly — that the append-reducer semantics we
declared (operator.add on `rejected`/`errors`) behave as intended.
"""

import operator
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.state import (
    new_screener_state,
    reject,
    sentiment_score,
    SENTIMENT_TO_SCORE,
    ScreenerState,
)


def test_new_screener_state_initialized():
    s = new_screener_state(
        watchlist=["AAPL", "MSFT"], account_cash=50_000,
        run_id="run-1", run_timestamp="2026-06-22T10:00:00Z",
    )
    assert s["watchlist"] == ["AAPL", "MSFT"]
    assert s["mode"] == "ENTRY_SCREENER"
    # All collection fields must start as empty lists (reducer-safe).
    for key in ("scout_candidates", "quant_candidates", "news_reports",
                "recommendations", "rejected", "errors"):
        assert s[key] == [], key
    assert s["notified"] is False


def test_reject_record_shape():
    r = reject("NVDA", "QUANT", "delta 0.52 outside band")
    assert r == {"symbol": "NVDA", "stage": "QUANT", "reason": "delta 0.52 outside band"}


def test_sentiment_mapping():
    assert sentiment_score("VERY_POSITIVE") == 5
    assert sentiment_score("negative") == 2          # case-insensitive
    assert sentiment_score("garbage-label") == 3     # unknown → NEUTRAL
    assert SENTIMENT_TO_SCORE["NEUTRAL"] == 3


def test_append_reducer_semantics():
    # Simulate how LangGraph merges append-reduced fields across nodes.
    scout_rejects = [reject("X", "SCOUT", "illiquid")]
    quant_rejects = [reject("Y", "QUANT", "wide spread")]
    merged = operator.add(scout_rejects, quant_rejects)
    assert merged == [
        {"symbol": "X", "stage": "SCOUT", "reason": "illiquid"},
        {"symbol": "Y", "stage": "QUANT", "reason": "wide spread"},
    ]
    # Original lists are not mutated by operator.add.
    assert len(scout_rejects) == 1


def test_state_is_plain_dict():
    # ScreenerState is a TypedDict → behaves as a dict at runtime.
    s: ScreenerState = new_screener_state(
        watchlist=[], account_cash=0, run_id="r", run_timestamp="t",
    )
    s["scout_candidates"] = [{"symbol": "AAPL", "is_optionable": True}]
    assert s["scout_candidates"][0]["symbol"] == "AAPL"


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
