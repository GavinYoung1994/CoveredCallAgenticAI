"""massive.com market-data client (Polygon.io-compatible REST).

A DROP-IN replacement for ``SchwabClient``: it exposes the exact same public
methods the LangGraph nodes already call — ``get_quotes_chunked``,
``extract_fundamentals``, ``get_price_history``, ``get_option_chain``,
``get_option_expirations``, ``is_optionable``, ``get_quote``,
``lookup_instrument`` — and returns data in the SAME normalized shapes (e.g. the
option chain as Schwab's ``callExpDateMap``, history as ``{"candles": [...]}``),
so the deterministic math engine and nodes work unchanged.

It also adds first-class TECHNICAL-INDICATOR methods backed by Massive's
``/v1/indicators/*`` endpoints: ``get_sma``, ``get_ema``, ``get_rsi``,
``get_macd``.

Auth: an API key sent both as a ``Authorization: Bearer`` header and an
``apiKey`` query param (Polygon accepts either), read from ``MASSIVE_API_KEY``.
Tests inject an ``httpx.Client`` with a MockTransport, so importing this module
never hits the network.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import httpx

from app.config import settings
from app.data.rate_limiter import RateLimiter

logger = logging.getLogger("massive-client")

# Schwab period_type → approximate days-per-unit, so we can translate the nodes'
# (period_type, period) history requests into Massive's from/to date window.
_PERIOD_DAYS = {"day": 1, "month": 31, "year": 366, "ytd": 366}
# Schwab frequency_type → Massive/Polygon aggregate timespan.
_TIMESPAN = {"minute": "minute", "daily": "day", "weekly": "week", "monthly": "month"}


class MassiveClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        base_url: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        clock: Optional[Callable[[], date]] = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.massive_api_key
        self._base_url = (base_url or settings.massive_api_base_url).rstrip("/")
        self._client = http_client or httpx.Client(timeout=30.0)
        # Reuse the same free-tier limiter config as the news client (5/min).
        self._limiter = rate_limiter or RateLimiter(
            settings.massive_rate_limit_calls,
            settings.massive_rate_limit_period_sec,
            name="massive",
        )
        # Injectable "today" so option DTEs are deterministic in tests.
        self._clock = clock or date.today

    # ── internals ─────────────────────────────────────────────────────
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._limiter.acquire()  # blocks if we're at the per-minute cap
        p = dict(params or {})
        p.setdefault("apiKey", self._api_key)  # belt-and-suspenders alongside the header
        resp = self._client.get(f"{self._base_url}{path}", headers=self._headers(), params=p)
        resp.raise_for_status()
        return resp.json()

    # ── quotes / snapshots ────────────────────────────────────────────
    def get_quotes(self, symbols: List[str], **_ignored) -> Dict[str, Any]:
        """Batch snapshot for many tickers → Schwab-shaped payload keyed by symbol.

        Uses the multi-ticker snapshot endpoint. Extra kwargs (e.g. ``fields``)
        are accepted and ignored for SchwabClient signature parity.
        """
        syms = [s.upper() for s in symbols]
        data = self._get(
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            params={"tickers": ",".join(syms)},
        )
        merged: Dict[str, Any] = {}
        for t in data.get("tickers", []) or []:
            sym = str(t.get("ticker", "")).upper()
            if sym:
                merged[sym] = self._snapshot_to_schwab_entry(t)
        # Ensure every requested symbol has an entry (missing → zeros → rejected).
        for sym in syms:
            merged.setdefault(sym, self._snapshot_to_schwab_entry({"ticker": sym}))
        return merged

    def get_quotes_chunked(
        self, symbols: List[str], batch_size: Optional[int] = None, **_ignored
    ) -> Dict[str, Any]:
        """Fetch snapshots for many symbols in rate-limited batches, merged into
        one Schwab-shaped dict keyed by symbol."""
        size = batch_size or settings.massive_quote_batch_size
        merged: Dict[str, Any] = {}
        for i in range(0, len(symbols), size):
            merged.update(self.get_quotes(symbols[i : i + size]))
        return merged

    def get_quote(self, symbol: str, **_ignored) -> Dict[str, Any]:
        """Single-ticker snapshot → ``{SYMBOL: {"quote": {...}, ...}}`` (matches
        the shape defense/nodes read via ``payload[sym]["quote"]["lastPrice"]``)."""
        sym = symbol.upper()
        data = self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{sym}")
        ticker = data.get("ticker", {}) or {}
        return {sym: self._snapshot_to_schwab_entry(ticker or {"ticker": sym})}

    @staticmethod
    def _snapshot_to_schwab_entry(t: Dict[str, Any]) -> Dict[str, Any]:
        """Map a Massive ticker snapshot → the Schwab /quotes per-symbol shape."""
        last_trade = t.get("lastTrade", {}) or {}
        day = t.get("day", {}) or {}
        prev = t.get("prevDay", {}) or {}
        # Prefer the last trade; fall back to day close, then previous close.
        last_price = last_trade.get("p") or day.get("c") or prev.get("c") or 0.0
        volume = day.get("v") or prev.get("v") or 0
        return {
            "assetMainType": "EQUITY",
            "quote": {"lastPrice": float(last_price or 0.0), "totalVolume": int(volume or 0)},
            # Massive snapshots don't carry avg volume / dividend yield; use day
            # volume as a rough avg proxy and leave dividend fields at 0. These
            # only matter in full-filter mode (the watchlist is prefiltered).
            "fundamental": {"avg10DaysVolume": int(volume or 0), "divYield": 0.0, "divAmount": 0.0},
            "reference": {"assetMainType": "EQUITY"},
        }

    @staticmethod
    def extract_fundamentals(quotes_payload: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Pull the Scout's filter inputs out of a payload for one symbol —
        identical output keys to ``SchwabClient.extract_fundamentals``."""
        sym = symbol.upper()
        entry = quotes_payload.get(sym, {}) if isinstance(quotes_payload, dict) else {}
        fund = entry.get("fundamental", {}) or {}
        quote = entry.get("quote", {}) or {}
        ref = entry.get("reference", {}) or {}
        avg_vol = fund.get("avg10DaysVolume") or fund.get("avg1YearVolume") or 0
        return {
            "symbol": sym,
            "asset_type": entry.get("assetMainType") or ref.get("assetMainType"),
            "last_price": float(quote.get("lastPrice", 0.0) or 0.0),
            "total_volume": int(quote.get("totalVolume", 0) or 0),
            "avg_daily_volume": int(avg_vol or 0),
            "dividend_yield_percent": float(fund.get("divYield", 0.0) or 0.0),
            "dividend_amount": float(fund.get("divAmount", 0.0) or 0.0),
            "next_div_ex_date": fund.get("nextDivExDate"),
            "pe_ratio": fund.get("peRatio"),
        }

    # ── historical candles ────────────────────────────────────────────
    def get_price_history(
        self,
        symbol: str,
        period_type: str = "year",
        period: int = 1,
        frequency_type: str = "daily",
        frequency: int = 1,
        start_date: Optional[int] = None,
        end_date: Optional[int] = None,
        **_ignored,
    ) -> Dict[str, Any]:
        """Aggregated OHLC bars → ``{"candles": [{open,high,low,close,volume,datetime}]}``.

        Translates the nodes' (period_type, period, frequency_type) request into
        Massive's ``/v2/aggs`` range window. ``start_date``/``end_date`` (ms
        epoch, SchwabClient parity) override the computed window if given.
        """
        timespan = _TIMESPAN.get(frequency_type, "day")
        to_d = (datetime.utcfromtimestamp(end_date / 1000).date() if end_date else self._clock())
        if start_date:
            from_d = datetime.utcfromtimestamp(start_date / 1000).date()
        else:
            span_days = _PERIOD_DAYS.get(period_type, 366) * max(int(period), 1)
            from_d = to_d - timedelta(days=span_days)
        path = (f"/v2/aggs/ticker/{symbol.upper()}/range/{max(int(frequency),1)}/{timespan}/"
                f"{from_d.isoformat()}/{to_d.isoformat()}")
        data = self._get(path, params={"adjusted": "true", "sort": "asc", "limit": 50000})
        candles = [
            {
                "open": r.get("o"), "high": r.get("h"), "low": r.get("l"),
                "close": r.get("c"), "volume": r.get("v"), "datetime": r.get("t"),
            }
            for r in (data.get("results") or [])
        ]
        return {"candles": candles, "symbol": symbol.upper(), "empty": not candles}

    # ── option chain ──────────────────────────────────────────────────
    def get_option_chain(
        self,
        symbol: str,
        contract_type: str = "CALL",
        range_filter: str = "OTM",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        strike: Optional[float] = None,
        **_ignored,
    ) -> Dict[str, Any]:
        """Options snapshot → Schwab-style ``{"callExpDateMap": {"<exp>:<dte>":
        {"<strike>": [contract]}}}`` so ``find_optimal_covered_call`` and the
        defense node consume it unchanged.

        ``range_filter`` (OTM/ALL) is accepted for parity but not needed — the
        engine selects by delta band. Follows ``next_url`` pagination up to the
        configured page cap.
        """
        params: Dict[str, Any] = {"contract_type": contract_type.lower(), "limit": 250}
        if from_date:
            params["expiration_date.gte"] = from_date
        if to_date:
            params["expiration_date.lte"] = to_date
        if strike is not None:
            params["strike_price"] = strike

        results: List[Dict[str, Any]] = []
        path = f"/v3/snapshot/options/{symbol.upper()}"
        pages = 0
        while path and pages < settings.massive_option_chain_max_pages:
            data = self._get(path, params=params if pages == 0 else None)
            results.extend(data.get("results") or [])
            nxt = data.get("next_url")
            if not nxt:
                break
            # next_url is absolute; strip the base so _get can re-add it + auth.
            path = nxt[len(self._base_url):] if nxt.startswith(self._base_url) else nxt
            params = {}
            pages += 1

        return self._results_to_call_map(results, contract_type)

    def _results_to_call_map(self, results: List[Dict[str, Any]], contract_type: str) -> Dict[str, Any]:
        today = self._clock()
        key_name = "putExpDateMap" if contract_type.lower() == "put" else "callExpDateMap"
        exp_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for c in results:
            details = c.get("details", {}) or {}
            exp = details.get("expiration_date")
            strike = details.get("strike_price")
            if not exp or strike is None:
                continue
            try:
                dte = (date.fromisoformat(str(exp)[:10]) - today).days
            except ValueError:
                continue
            greeks = c.get("greeks", {}) or {}
            q = c.get("last_quote", {}) or {}
            day = c.get("day", {}) or {}
            bid = float(q.get("bid", 0.0) or 0.0)
            ask = float(q.get("ask", 0.0) or 0.0)
            day_close = float(day.get("close", 0.0) or 0.0)
            midpoint = q.get("midpoint")
            # Prefer the live NBBO midpoint; off-hours (bid/ask zeroed) fall back to
            # the bid/ask midpoint, then the day's last trade so a premium is still
            # available for covered-call selection when the market is closed.
            if midpoint not in (None, "") and float(midpoint) > 0:
                mark = float(midpoint)
            elif ask > 0:
                mark = round((bid + ask) / 2.0, 4)
            else:
                mark = day_close
            iv = c.get("implied_volatility")
            contract = {
                "symbol": details.get("ticker"),
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
                "vega": greeks.get("vega"),
                "bid": bid,
                "ask": ask,
                "mark": mark,
                "totalVolume": int(day.get("volume", 0) or 0),
                "openInterest": int(c.get("open_interest", 0) or 0),
                # Massive IV is a decimal (0.35); the engine expects a percent.
                "volatility": (float(iv) * 100.0) if iv not in (None, "") else 0.0,
            }
            key = f"{str(exp)[:10]}:{dte}"
            exp_map.setdefault(key, {}).setdefault(str(strike), []).append(contract)
        out: Dict[str, Any] = {key_name: exp_map}
        if not exp_map:
            out["error"] = "No option chain data found."
        return out

    # ── option reference (expirations / optionable) ───────────────────
    def get_option_expirations(self, symbol: str, **_ignored) -> Dict[str, Any]:
        """Distinct upcoming expiration dates → ``{"expirations": [...]}``."""
        data = self._get(
            "/v3/reference/options/contracts",
            params={"underlying_ticker": symbol.upper(), "expired": "false", "limit": 1000},
        )
        exps = sorted({r.get("expiration_date") for r in (data.get("results") or []) if r.get("expiration_date")})
        return {"expirations": exps, "expirationList": exps}

    def is_optionable(self, symbol: str) -> bool:
        """True if the underlying has any listed (unexpired) option contracts."""
        try:
            return bool(self.get_option_expirations(symbol).get("expirations"))
        except httpx.HTTPError as exc:
            logger.warning("Optionable check failed for %s: %s", symbol, exc)
            return False

    def lookup_instrument(self, symbol: str, **_ignored) -> Dict[str, Any]:
        """Ticker overview (name, market, type, etc.)."""
        return self._get(f"/v3/reference/tickers/{symbol.upper()}")

    # ── technical indicators (Massive /v1/indicators/*) ───────────────
    def _indicator(self, name: str, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        p = {"timespan": "day", "series_type": "close", "order": "desc", "adjusted": "true"}
        p.update({k: v for k, v in params.items() if v is not None})
        data = self._get(f"/v1/indicators/{name}/{symbol.upper()}", params=p)
        values = ((data.get("results") or {}).get("values")) or []
        return {"symbol": symbol.upper(), "indicator": name, "values": values,
                "latest": (values[0].get("value") if values else None)}

    def get_sma(self, symbol: str, window: int = 50, timespan: str = "day",
                series_type: str = "close", limit: int = 30) -> Dict[str, Any]:
        return self._indicator("sma", symbol, {"window": window, "timespan": timespan,
                                               "series_type": series_type, "limit": limit})

    def get_ema(self, symbol: str, window: int = 50, timespan: str = "day",
                series_type: str = "close", limit: int = 30) -> Dict[str, Any]:
        return self._indicator("ema", symbol, {"window": window, "timespan": timespan,
                                               "series_type": series_type, "limit": limit})

    def get_rsi(self, symbol: str, window: int = 14, timespan: str = "day",
                series_type: str = "close", limit: int = 30) -> Dict[str, Any]:
        return self._indicator("rsi", symbol, {"window": window, "timespan": timespan,
                                               "series_type": series_type, "limit": limit})

    def get_macd(self, symbol: str, short_window: int = 12, long_window: int = 26,
                 signal_window: int = 9, timespan: str = "day",
                 series_type: str = "close", limit: int = 30) -> Dict[str, Any]:
        p = {"short_window": short_window, "long_window": long_window,
             "signal_window": signal_window, "timespan": timespan,
             "series_type": series_type, "order": "desc", "adjusted": "true", "limit": limit}
        data = self._get(f"/v1/indicators/macd/{symbol.upper()}", params=p)
        values = ((data.get("results") or {}).get("values")) or []
        return {"symbol": symbol.upper(), "indicator": "macd", "values": values,
                "latest": (values[0] if values else None)}

    # ── lifecycle ─────────────────────────────────────────────────────
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MassiveClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
