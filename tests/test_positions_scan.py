"""Tests for loading open positions + scanning ALL holdings for downside defense."""

import os
import sqlite3
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.data.schwab_client import SchwabClient
from app.data.news_client import NewsClient
from app.llm import LocalLLM
from app.notify.discord_webhook import DiscordNotifier
from app.memory import decision_store as ds
from app.memory.positions_store import load_open_positions, list_holdings_detailed, _holding_return
from app.graphs.defense_monitor import run_defense_scan


def test_holding_return_closed_realized():
    # Closed: +1295 on $8,695 cost over 30 days → ~14.9% → ~181% annualized.
    ret, ann, days, basis = _holding_return(
        status="ASSIGNED", stock_price=86.95, shares=100, realized_pnl=1295.0,
        premium=4.9, contracts=1, entry_date="2026-06-01", close_date="2026-07-01",
        expiration="2026-07-18")
    assert days == 30 and basis == "realized"
    assert abs(ret - 14.89) < 0.05
    assert abs(ann - 181.2) < 1.0


def test_holding_return_open_premium_to_expiration():
    # Open: $200 premium income on $10,000 over 30 days to expiration → 2% → ~24.3% annualized.
    ret, ann, days, basis = _holding_return(
        status="OPEN", stock_price=100.0, shares=100, realized_pnl=0.0,
        premium=2.0, contracts=1, entry_date="2026-06-01", close_date=None,
        expiration="2026-07-01")
    assert days == 30 and "premium" in basis
    assert ret == 2.0 and abs(ann - 24.33) < 0.1


def test_holding_return_missing_data():
    ret, ann, days, basis = _holding_return(
        status="OPEN", stock_price=0, shares=0, realized_pnl=0, premium=None,
        contracts=None, entry_date=None, close_date=None, expiration=None)
    assert ret is None and ann is None


def _seed():
    fd, db = tempfile.mkstemp(suffix=".db", dir=os.environ.get("TMPDIR")); os.close(fd); os.unlink(db)
    ds.open_position(position_id="KO_1", symbol="KO", stock_purchase_price=100.0, shares=100,
                     call_strike=105.0, call_premium=2.0, call_expiration="2026-07-28", db_path=db)
    ds.open_position(position_id="XOM_1", symbol="XOM", stock_purchase_price=110.0, shares=100,
                     call_strike=115.0, call_premium=1.5, call_expiration="2026-07-28", db_path=db)
    ds.open_position(position_id="OLD_1", symbol="OLD", stock_purchase_price=50.0, shares=100,
                     call_strike=52.5, call_premium=1.0, call_expiration="2026-01-01", db_path=db)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE positions SET status='ASSIGNED' WHERE position_id='OLD_1'")  # closed
    conn.commit(); conn.close()
    return db


def test_load_open_positions():
    db = _seed()
    try:
        positions = load_open_positions(db)
        by_sym = {p["symbol"]: p for p in positions}
        assert set(by_sym) == {"KO", "XOM"}          # the closed OLD is excluded
        assert by_sym["KO"]["short_call_strike"] == 105.0
        assert by_sym["KO"]["original_premium"] == 2.0
        assert by_sym["KO"]["shares"] == 100
        assert by_sym["KO"]["short_call_expiration"] == "2026-07-28"
    finally:
        os.path.exists(db) and os.unlink(db)


def test_load_open_positions_migrates_legacy_db():
    # A DB created before the downside_buffer_percent column must auto-migrate
    # (the bug that crashed the defense scan), not raise.
    fd, db = tempfile.mkstemp(suffix=".db", dir=os.environ.get("TMPDIR")); os.close(fd); os.unlink(db)
    try:
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE positions (position_id TEXT PRIMARY KEY, symbol TEXT, status TEXT,
                entry_date DATETIME DEFAULT CURRENT_TIMESTAMP, close_date DATETIME,
                stock_purchase_price REAL, total_realized_pnl REAL DEFAULT 0.0);
            CREATE TABLE transactions (transaction_id TEXT PRIMARY KEY, position_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, asset_type TEXT, action TEXT,
                quantity INTEGER, price REAL, fees REAL DEFAULT 0.0, strike_price REAL, expiration_date DATE);
            INSERT INTO positions (position_id, symbol, status, stock_purchase_price)
                VALUES ('KO_1', 'KO', 'OPEN', 60.0);
            INSERT INTO transactions (transaction_id, position_id, asset_type, action, quantity, price)
                VALUES ('s', 'KO_1', 'STOCK', 'BUY_TO_OPEN', 100, 60.0);
            INSERT INTO transactions (transaction_id, position_id, asset_type, action, quantity, price,
                strike_price, expiration_date)
                VALUES ('o', 'KO_1', 'OPTION', 'SELL_TO_OPEN', 1, 1.2, 62.5, '2026-07-28');
        """)
        conn.commit(); conn.close()
        positions = load_open_positions(db)              # must migrate + succeed
        assert positions[0]["symbol"] == "KO"
        assert positions[0]["downside_buffer_percent"] is None
    finally:
        os.path.exists(db) and os.unlink(db)


def test_list_holdings_detailed():
    db = _seed()
    try:
        rows = list_holdings_detailed(db_path=db)
        by = {r["symbol"]: r for r in rows}
        assert {"KO", "XOM", "OLD"} <= set(by)           # all positions, any status
        assert by["KO"]["short_call_strike"] == 105.0
        assert by["KO"]["short_call_premium"] == 2.0
        assert by["KO"]["shares"] == 100
        assert by["OLD"]["status"] == "ASSIGNED"
        # Annualized return fields are present on every holding.
        assert "annualized_return_percent" in by["KO"] and "return_basis" in by["KO"]
    finally:
        os.path.exists(db) and os.unlink(db)


def _schwab_handler(request):
    p = request.url.path
    if p.endswith("/quotes"):
        sym = p.strip("/").split("/")[-2]            # /.../{SYM}/quotes
        return httpx.Response(200, json={sym: {"quote": {"lastPrice": 90.0}}})  # ~ -10% / -18%
    if p.endswith("/chains"):
        short = [{"symbol": "Cshort", "delta": 0.20, "bid": 0.4, "ask": 0.5, "mark": 0.45,
                  "totalVolume": 10, "openInterest": 100, "volatility": 25.0}]
        roll = [{"symbol": "Croll", "delta": 0.35, "bid": 1.4, "ask": 1.6, "mark": 1.5,
                 "totalVolume": 100, "openInterest": 500, "volatility": 25.0}]
        return httpx.Response(200, json={"callExpDateMap":
                              {"2026-07-28:34": {"105.0": short, "115.0": short, "95.0": roll}}})
    return httpx.Response(404, json={})


def _llm_backend(msgs):
    c = " ".join(m["content"] for m in msgs)
    if "recommended_branch" in c:
        return '{"recommended_branch": "B", "rationale": "Roll for credit."}'
    return '{"sentiment": "NEUTRAL", "catastrophic_risk": false, "rationale": "Sector weakness."}'


def _news_handler(request):
    sym = request.url.params.get("ticker")
    return httpx.Response(200, json={"results": [
        {"title": f"{sym} down", "description": "dip", "article_url": "http://n",
         "publisher": {"name": "Wire"}, "tickers": [sym], "insights": []}]})


def test_run_defense_scan_all_holdings():
    db = _seed()
    try:
        schwab = SchwabClient(token_provider=lambda: "t",
                              http_client=httpx.Client(transport=httpx.MockTransport(_schwab_handler)),
                              base_url="https://mock/marketdata/v1")
        news = NewsClient(api_key="k", base_url="https://mock", fetch_content=False,
                          http_client=httpx.Client(transport=httpx.MockTransport(_news_handler)))
        result = run_defense_scan(
            db_path=db, today=date(2026, 6, 24),
            schwab_client=schwab, news_client=news, llm=LocalLLM(backend=_llm_backend),
            notifier=DiscordNotifier(webhook_url="https://discord.com/api/webhooks/1/abc",
                                     poster=lambda u, p: 204))
        assert result["scanned"] == 2                       # only the 2 open positions
        assert set(result["breached"]) == {"KO", "XOM"}     # both down ~10-18% → breach
    finally:
        os.path.exists(db) and os.unlink(db)
        import glob
        for f in glob.glob("runs/defense_*"):
            os.unlink(f)


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
