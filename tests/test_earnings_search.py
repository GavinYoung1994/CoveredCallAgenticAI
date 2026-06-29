"""Tests for the Google-search earnings engine (offline via MockTransport)."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.data.earnings_search import (
    EarningsSearchClient, CompositeEarningsClient, find_earnings_dates, add_months,
)
from app.llm import LocalLLM


def _client(html: str, llm=None, use_llm=False) -> EarningsSearchClient:
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)
    return EarningsSearchClient(base_url="https://mock/search",
                                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                                llm=llm, use_llm=use_llm)


def test_add_months_clamps_day():
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)   # clamp to Feb
    assert add_months(date(2026, 4, 30), 3) == date(2026, 7, 30)   # Apr→Jul
    assert add_months(date(2026, 11, 15), 3) == date(2027, 2, 15)  # year rollover


def test_find_earnings_dates_prefers_near_earnings():
    text = "Some promo dated Jan 1, 2020. Next earnings date Jul 31, 2026 confirmed."
    dates = find_earnings_dates(text)
    assert date(2026, 7, 31) in dates
    assert dates[0] == date(2026, 7, 31)        # the one near 'earnings' ranks first


def test_search_returns_future_next_date():
    html = "<html><body><div>Earnings date: Jul 31, 2026</div></body></html>"
    d = _client(html).get_next_earnings_date("AAPL", from_date="2026-06-25")
    assert d == "2026-07-31"


def test_search_infers_next_from_past_date_quarterly():
    # Only a PAST earnings date is present → infer the next one (+3 months).
    html = "<html><body><p>AAPL reported earnings on Apr 30, 2026.</p></body></html>"
    d = _client(html).get_next_earnings_date("AAPL", from_date="2026-06-25")
    assert d == "2026-07-30"                    # Apr 30 + 3 months, first future date


def test_llm_picks_correct_candidate_among_many():
    # Page has an ex-div date AND the earnings date; the LLM picks earnings.
    html = ("<html><body>Ex-dividend date Jul 10, 2026. "
            "Next earnings date Aug 05, 2026. Price target updated Jul 20, 2026."
            "</body></html>")
    llm = LocalLLM(backend=lambda msgs: '{"earnings_date": "2026-08-05"}')
    d = _client(html, llm=llm, use_llm=True).get_next_earnings_date("AAPL", from_date="2026-06-25")
    assert d == "2026-08-05"


def test_llm_hallucination_is_rejected():
    # LLM returns a date NOT in the candidate set → grounding guard discards it,
    # falls back to the heuristic (earliest future = 2026-07-31).
    html = "<html><body>Earnings date Jul 31, 2026.</body></html>"
    llm = LocalLLM(backend=lambda msgs: '{"earnings_date": "2027-01-01"}')  # not on page
    d = _client(html, llm=llm, use_llm=True).get_next_earnings_date("AAPL", from_date="2026-06-25")
    assert d == "2026-07-31"


def test_llm_picks_past_date_then_inferred_quarterly():
    # LLM identifies the last reported date; deterministic code projects forward.
    html = "<html><body>AAPL reported earnings Apr 30, 2026.</body></html>"
    llm = LocalLLM(backend=lambda msgs: '{"earnings_date": "2026-04-30"}')
    d = _client(html, llm=llm, use_llm=True).get_next_earnings_date("AAPL", from_date="2026-06-25")
    assert d == "2026-07-30"        # Apr 30 + 3 months (LLM never did the math)


def test_search_no_dates_returns_none():
    d = _client("<html><body>No earnings info here.</body></html>").get_next_earnings_date(
        "AAPL", from_date="2026-06-25")
    assert d is None


def test_search_failure_returns_none():
    def boom(request):
        return httpx.Response(503, text="blocked")
    client = EarningsSearchClient(base_url="https://mock/search",
                                  http_client=httpx.Client(transport=httpx.MockTransport(boom)))
    assert client.get_next_earnings_date("AAPL", from_date="2026-06-25") is None


# ── composite chaining ────────────────────────────────────────────────
class _Stub:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def get_next_earnings_date(self, symbol, from_date=None, to_date=None):
        self.calls += 1
        return self.value


def test_composite_uses_primary_when_present():
    primary, fallback = _Stub("2026-08-01"), _Stub("2026-09-01")
    c = CompositeEarningsClient([primary, fallback])
    assert c.get_next_earnings_date("X") == "2026-08-01"
    assert fallback.calls == 0                  # never consulted


def test_composite_falls_back_when_primary_empty():
    primary, fallback = _Stub(None), _Stub("2026-09-01")
    c = CompositeEarningsClient([primary, fallback])
    assert c.get_next_earnings_date("X") == "2026-09-01"
    assert primary.calls == 1 and fallback.calls == 1


def test_composite_all_empty_returns_none():
    assert CompositeEarningsClient([_Stub(None), _Stub(None)]).get_next_earnings_date("X") is None


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
