"""Tests for the deterministic math engine.

Runs as a plain script (``python tests/test_math_engine.py``) OR under pytest.
The pure-arithmetic tests have no third-party dependencies; the indicator and
volatility tests are skipped gracefully if pandas/numpy aren't installed.
"""

import math
import sys
from pathlib import Path

# Allow `import app...` when run directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine import math_engine as e


# ── helpers ───────────────────────────────────────────────────────────
def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def _have(mod_name):
    try:
        __import__(mod_name)
        return True
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════
#  Pure arithmetic (no heavy deps)
# ══════════════════════════════════════════════════════════════════════
def test_basic_calculator():
    assert e.basic_calculator("add", 2, 3)["result"] == 5
    assert e.basic_calculator("divide", 1, 0)["error"]
    assert e.basic_calculator("percent_change", 100, 90)["result"] == -10.0


def test_position_size():
    r = e.calculate_position_size(50_000, 100.0)  # 1 contract = $10k
    assert r["can_afford_trade"] is True
    assert r["max_contracts"] == 5
    assert r["total_shares_to_buy"] == 500
    poor = e.calculate_position_size(500, 100.0)
    assert poor["can_afford_trade"] is False and poor["max_contracts"] == 0


def test_adjusted_cost_basis():
    r = e.calculate_adjusted_cost_basis(100.0, 3.0, 2.0)
    assert r["new_adjusted_cost_basis"] == 95.0
    assert r["downside_protection_dollars"] == 5.0


def test_yield_metrics():
    # Stock $100, strike $105, $2 premium, 30 DTE.
    r = e.calculate_yield_metrics(100.0, 105.0, 2.0, 30)
    assert r["downside_breakeven_price"] == 98.0
    assert r["max_profit_dollars"] == 700.0  # (5 + 2) * 100
    # AROC if flat = 2/100 * 365/30 * 100 = 24.33%
    assert approx(r["aroc_if_flat_percent"], 24.33, 0.05)


def test_premium_composition():
    itm = e.calculate_premium_composition(110.0, 100.0, 12.0)
    assert itm["intrinsic_value"] == 10.0 and itm["extrinsic_value"] == 2.0
    assert itm["is_itm"] is True
    otm = e.calculate_premium_composition(95.0, 100.0, 1.5)
    assert otm["intrinsic_value"] == 0.0 and otm["extrinsic_value"] == 1.5


def test_liquidity_slippage():
    tight = e.calculate_liquidity_slippage(1.95, 2.00)  # 2.5% spread
    assert tight["is_tradable"] is True
    wide = e.calculate_liquidity_slippage(1.00, 2.00)  # 50% spread
    assert wide["is_tradable"] is False
    assert e.calculate_liquidity_slippage(1.0, 0.0).get("error")


def test_iv_rank():
    r = e.calculate_iv_rank(0.40, 0.60, 0.20)  # midpoint → 50
    assert approx(r["iv_rank"], 50.0)
    assert r["premiums_are_rich"] is True
    low = e.calculate_iv_rank(0.25, 0.60, 0.20)
    assert low["premiums_are_rich"] is False
    assert e.calculate_iv_rank(0.4, 0.2, 0.6).get("error")  # bad range


def test_earnings_guardrail():
    # Earnings BEFORE expiration → unsafe → disqualify.
    bad = e.is_earnings_within_cycle("2026-07-10", "2026-07-17")
    assert bad["disqualify"] is True
    # Earnings AFTER expiration → safe.
    good = e.is_earnings_within_cycle("2026-08-01", "2026-07-17")
    assert good["disqualify"] is False
    # Unknown earnings → don't hard-block, flag as unknown.
    unknown = e.is_earnings_within_cycle(None, "2026-07-17")
    assert unknown["disqualify"] is False and unknown["earnings_known"] is False


def test_macro_divergence():
    # Asset and market both down hard, tightly correlated (divergence -1.3,
    # inside the < 1.5 band) → macro drag.
    macro = e.calculate_macro_divergence(100, 96, 100, 97.3)
    assert macro["heuristic_classification"] == "Macro Sector Drag", macro
    # Asset crashes while market is flat → micro failure.
    micro = e.calculate_macro_divergence(100, 93, 100, 99.8)
    assert micro["heuristic_classification"] == "Micro Company Failure"


def test_tot_branches():
    # Entry $100, now $90, collected $2, buy-back ask $0.50, roll for $1.50.
    r = e.generate_tot_defense_branches(100.0, 90.0, 2.0, 0.50, 1.50)
    # Branch A: stock loss -1000 + call pnl +150 = -850
    assert r["Branch_A_Liquidate"]["realized_cash_loss"] == -850.0
    # Branch B: roll credit = (1.50 - 0.50) * 100 = 100 → valid
    assert r["Branch_B_Roll_Down"]["net_credit_received"] == 100.0
    assert r["Branch_B_Roll_Down"]["is_valid"] is True
    assert r["Branch_C_Hold"]["unrealized_net_pnl"] == -850.0


def test_score_and_grade():
    weights = {"yield": 0.35, "iv": 0.20, "sentiment": 0.20, "buffer": 0.15, "prob": 0.10}
    # Strong candidate: high yield, rich IV, positive sentiment, good buffer.
    strong = e.score_covered_call_candidate(30, 2.5, 5, 10, 90, weights)
    assert strong["score"] > 90
    assert e.grade_from_score(strong["score"], {"A": 75, "B": 60, "C": 45}) == "A"
    # Weak candidate: low everything.
    weak = e.score_covered_call_candidate(2, 1.0, 1, 0.5, 30, weights)
    assert weak["score"] < 30
    assert e.grade_from_score(weak["score"], {"A": 75, "B": 60, "C": 45}) == "D"
    # Components are all within [0, 1].
    assert all(0.0 <= v <= 1.0 for v in strong["components"].values())


def test_score_weight_normalization():
    # Weights that don't sum to 1 are normalized, not taken literally.
    w = {"yield": 70, "iv": 0, "sentiment": 0, "buffer": 0, "prob": 0}
    r = e.score_covered_call_candidate(30, 1.0, 1, 0, 0, w)  # only yield matters, maxed
    assert approx(r["score"], 100.0, 0.01)


def test_find_optimal_covered_call():
    # Build a tiny synthetic Schwab-style chain. Key fmt "DATE:DTE".
    chain = {
        "callExpDateMap": {
            "2026-07-17:35": {
                "100.0": [{"symbol": "X_100C", "delta": 0.55, "bid": 5, "ask": 5.2, "mark": 5.1}],
                "105.0": [{"symbol": "X_105C", "delta": 0.33, "bid": 2, "ask": 2.2, "mark": 2.1}],
                "110.0": [{"symbol": "X_110C", "delta": 0.15, "bid": 0.5, "ask": 0.7, "mark": 0.6}],
            },
            "2026-09-18:90": {  # outside the 30-45 DTE window → ignored
                "105.0": [{"symbol": "X_FAR", "delta": 0.35, "bid": 4, "ask": 4.2, "mark": 4.1}],
            },
        }
    }
    r = e.find_optimal_covered_call(chain, target_delta=0.35, min_dte=30, max_dte=45)
    assert r["symbol"] == "X_105C"          # 0.33 is closest to 0.35 within band
    assert r["in_delta_band"] is True
    assert r["days_to_expiration"] == 35


def test_black_scholes_greeks():
    # ATM call: S=K=100, 30 DTE, 20% IV, r=0, q=0 → delta just above 0.5.
    g = e.black_scholes_call_greeks(100, 100, 30, 0.20, risk_free_rate=0.0)
    assert 0.50 < g["delta"] < 0.53, g
    assert 47 < g["prob_assignment_percent"] < 51, g
    assert g["gamma"] > 0 and g["vega_per_1pct_vol"] > 0
    assert g["theta_per_day"] < 0  # long call loses value each day
    # Deep OTM call → low delta, low assignment probability.
    otm = e.black_scholes_call_greeks(100, 130, 30, 0.20, risk_free_rate=0.0)
    assert otm["delta"] < 0.10 and otm["prob_assignment_percent"] < 10
    assert e.black_scholes_call_greeks(0, 100, 30, 0.2).get("error")


def test_probability_of_profit():
    # Breakeven below spot → POP should exceed 50%.
    r = e.calculate_probability_of_profit(100, 98, 0.20, 30)
    assert r["prob_of_profit_percent"] > 50, r
    # Breakeven above spot → POP below 50%.
    r2 = e.calculate_probability_of_profit(100, 103, 0.20, 30)
    assert r2["prob_of_profit_percent"] < 50, r2


def test_expected_move():
    # 1 year, 20% vol → expected move = 20% of price, exactly.
    r = e.calculate_expected_move(100, 0.20, 365)
    assert approx(r["expected_move_dollars"], 20.0, 0.01)
    assert approx(r["upper_expected_price"], 120.0, 0.01)
    assert approx(r["lower_expected_price"], 80.0, 0.01)


def test_moneyness():
    r = e.calculate_moneyness(100, 105)
    assert r["is_otm"] is True and approx(r["otm_cushion_percent"], 5.0)
    itm = e.calculate_moneyness(100, 95)
    assert itm["is_otm"] is False and approx(itm["otm_cushion_percent"], -5.0)


def test_dividend_yield():
    r = e.calculate_dividend_yield(2.5, 100)
    assert approx(r["dividend_yield_percent"], 2.5)
    assert e.calculate_dividend_yield(2.5, 0).get("error")


# ══════════════════════════════════════════════════════════════════════
#  Indicator tests (need pandas/numpy) — skipped if unavailable
# ══════════════════════════════════════════════════════════════════════
def test_technical_indicators():
    if not (_have("pandas") and _have("numpy")):
        print("  ⏭  skipping technical-indicator test (pandas/numpy not installed)")
        return
    # Build a clean linear uptrend of 260 closes: 100, 100.5, 101, ...
    candles = [{"close": 100 + i * 0.5} for i in range(260)]
    r = e.calculate_technical_indicators(candles, trend_lookback_days=20)
    assert "error" not in r, r
    assert r["trend_analysis"]["detected_trend"].startswith("Upward")
    snap = r["current_snapshot"]
    # On a monotonic uptrend the 50-SMA must sit below the latest price.
    assert snap["SMA_50"] < snap["price"]
    # RSI on a pure uptrend should be very high (near 100).
    assert snap["RSI_14"] > 90


def test_historical_volatility():
    if not _have("numpy"):
        print("  ⏭  skipping HV test (numpy not installed)")
        return
    candles = [{"close": 100 + (i % 2)} for i in range(120)]  # oscillating
    r = e.calculate_historical_volatility(candles)
    assert "error" not in r and r["annualized_hv_percent"] > 0


def test_technical_indicators_90_days_no_sma200():
    if not (_have("pandas") and _have("numpy")):
        print("  ⏭  skipping 90-day indicator test (pandas/numpy not installed)")
        return
    # 90 daily candles is enough for SMA_50 + 20-day trend, but NOT the 200-SMA.
    candles = [{"close": 100 + i * 0.3} for i in range(90)]
    r = e.calculate_technical_indicators(candles, trend_lookback_days=20)
    assert "error" not in r, r
    assert r["current_snapshot"]["SMA_200"] is None      # gracefully omitted
    assert r["current_snapshot"]["SMA_50"] is not None
    assert r["trend_analysis"]["candles_used"] == 90
    assert r["trend_analysis"]["detected_trend"].startswith("Upward")


def test_insufficient_candles():
    # 60 candles < (50 + 20) required → error.
    r = e.calculate_technical_indicators([{"close": 100 + i} for i in range(60)])
    assert r.get("error")


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
