"""Tests for the Risk Manager node + Discord notifier.

Fake LLM backend for rationale prose; a capturing fake poster for Discord so we
assert on the exact message payload without any network call.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import LocalLLM
from app.notify.discord_webhook import DiscordNotifier, format_recommendations
from app.nodes.risk_manager import build_risk_manager_node
from app.state import new_screener_state


def _llm():
    return LocalLLM(backend=lambda msgs: "Solid risk/reward; main risk is a sharp selloff.")


def _quant_candidate(sym, aroc_flat, delta=0.35, buffer=5.0, iv_ratio=1.8):
    return {
        "symbol": sym,
        "underlying_price": 100.0,
        "contract": {"symbol": f"{sym}_C", "strike": 105.0, "expiration_key": "2026-07-28:35",
                     "days_to_expiration": 35, "delta": delta, "mark": 2.05},
        "greeks": {"prob_expire_otm_percent": 70.0},
        "iv_rank": {"iv_to_hv_ratio": iv_ratio},
        "yield_metrics": {"aroc_if_flat_percent": aroc_flat, "aroc_if_assigned_percent": aroc_flat + 10,
                          "downside_buffer_percent": buffer},
    }


def _news_report(sym, sentiment="POSITIVE", score=4, passes=True, earnings_known=True):
    return {"symbol": sym, "sentiment": sentiment, "sentiment_score": score,
            "passes_news": passes, "earnings_known": earnings_known, "earnings_date": "2026-09-01",
            "sources": [{"title": f"{sym} news", "publisher": "Wire", "url": "http://x"}]}


def _run(quant, news, notifier=None):
    node = build_risk_manager_node(_llm(), notifier=notifier)
    state = new_screener_state(watchlist=[], account_cash=50_000, run_id="run-9", run_timestamp="t")
    state["quant_candidates"] = quant
    state["news_reports"] = news
    return node(state)


def test_risk_grades_and_sorts():
    quant = [_quant_candidate("HIGH", 25.0), _quant_candidate("MID", 14.0)]
    news = [_news_report("HIGH"), _news_report("MID")]
    out = _run(quant, news)
    recs = out["recommendations"]
    assert [r["symbol"] for r in recs] == ["HIGH", "MID"]   # sorted by score desc
    assert recs[0]["score"] >= recs[1]["score"]
    assert recs[0]["grade"] in {"A", "B", "C", "D"}
    assert recs[0]["rationale"].startswith("Solid risk")    # LLM prose attached
    assert recs[0]["sources"][0]["publisher"] == "Wire"     # sources carried through


def test_risk_rejects_below_yield_target():
    quant = [_quant_candidate("LOWYLD", 6.0)]  # < 10% flat target
    out = _run(quant, [_news_report("LOWYLD")])
    assert out["recommendations"] == []
    assert any("< 10% target" in r["reason"] for r in out["rejected"])
    assert out["rejected"][0]["stage"] == "RISK_MANAGER"


def test_risk_skips_failed_news():
    # A candidate whose news failed should never be graded.
    quant = [_quant_candidate("BAD", 25.0)]
    out = _run(quant, [_news_report("BAD", sentiment="NEGATIVE", score=2, passes=False)])
    assert out["recommendations"] == []


def test_risk_top_n_cap():
    quant = [_quant_candidate(f"S{i}", 20.0 + i) for i in range(8)]
    news = [_news_report(f"S{i}") for i in range(8)]
    out = _run(quant, news)
    assert len(out["recommendations"]) == 5   # default top_n_candidates


def test_discord_notifier_sends_payload():
    calls = []

    def fake_poster(url, payload):
        calls.append(payload["content"])
        return 204  # Discord success

    notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/abc", poster=fake_poster)
    out = _run([_quant_candidate("HIGH", 25.0)], [_news_report("HIGH")], notifier=notifier)
    assert out["notified"] is True
    joined = "\n".join(calls)
    assert "HIGH" in joined and "APPROVAL REQUIRED" in joined


def test_discord_chunks_long_message_under_2000():
    calls = []
    notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/abc",
                               poster=lambda u, p: calls.append(p["content"]) or 204)
    long_content = "\n".join(f"line {i} " + "x" * 30 for i in range(300))  # ~12k chars
    assert notifier.send(long_content) is True
    assert len(calls) > 1                                  # split across calls
    assert all(len(c) <= 2000 for c in calls)              # every chunk within limit
    assert "\n".join(calls) == long_content                # full content preserved, in order


def test_discord_hard_splits_overlong_line():
    chunks = DiscordNotifier._chunk("y" * 5000)
    assert len(chunks) == 3 and all(len(c) <= 2000 for c in chunks)
    assert "".join(chunks) == "y" * 5000


def test_discord_disabled_when_placeholder():
    notifier = DiscordNotifier(webhook_url="https://discord.com/api/webhooks/xxxx/yyyy")
    assert notifier.enabled is False
    assert notifier.send("hi") is False


def test_format_recommendations_empty():
    msg = format_recommendations([], run_id="r1")
    assert "no candidates" in msg.lower()


def test_format_flags_unknown_earnings():
    rec = {"symbol": "ZZZ", "grade": "B", "score": 65, "annualized_yield_percent": 15.0,
           "contract": {"strike": 50, "expiration_key": "2026-07-28:35", "days_to_expiration": 35,
                        "delta": 0.35, "mark": 1.2},
           "yield_metrics": {"downside_buffer_percent": 4.0}, "sentiment": "NEUTRAL",
           "earnings_known": False, "sources": []}
    msg = format_recommendations([rec])
    assert "Earnings date UNKNOWN" in msg
    # A one-click Google search link for the ticker's earnings date is included.
    assert "google.com/search" in msg and "ZZZ" in msg


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
