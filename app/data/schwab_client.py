"""Charles Schwab Market Data API client (synchronous).

This is the importable client the LangGraph nodes call directly. It wraps the
five market-data endpoints the strategy needs and adds small *normalizing*
helpers so the nodes don't each have to dig through Schwab's nested JSON.

Auth: a token provider callable returns a valid bearer token. By default we use
``token_manager.get_valid_access_token`` (which silently refreshes via the
gitignored ``schwab_tokens.json``). Tests inject a stub provider instead, so
importing this module never touches the network or real credentials.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import httpx

from app.config import settings
from app.data.rate_limiter import RateLimiter

logger = logging.getLogger("schwab-client")


class SchwabClient:
    def __init__(
        self,
        *,
        token_provider: Optional[Callable[[], str]] = None,
        http_client: Optional[httpx.Client] = None,
        base_url: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self._base_url = (base_url or settings.schwab_base_url).rstrip("/")
        self._client = http_client or httpx.Client(timeout=30.0)
        self._token_provider = token_provider  # resolved lazily (see _token)
        # Throttle every Schwab call so a large watchlist run stays under the cap.
        self._limiter = rate_limiter or RateLimiter(
            settings.schwab_rate_limit_calls,
            settings.schwab_rate_limit_period_sec,
            name="schwab",
        )

    # ── internals ─────────────────────────────────────────────────────
    def _token(self) -> str:
        """Resolve the bearer token, importing the default provider lazily so
        that tests injecting their own provider never import token_manager."""
        if self._token_provider is None:
            from charles_schwab_mcp.token_manager import get_valid_access_token
            self._token_provider = get_valid_access_token
        return self._token_provider()

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token()}", "Accept": "application/json"}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._limiter.acquire()  # blocks if we're at the per-minute cap
        url = f"{self._base_url}{path}"
        resp = self._client.get(url, headers=self._headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    # ── raw endpoints ─────────────────────────────────────────────────
    def get_quotes(
        self, symbols: List[str], fields: str = "quote,fundamental,reference"
    ) -> Dict[str, Any]:
        """Batch quotes. ``fields`` controls which blocks come back; we want
        ``fundamental`` (divYield, avg volume) and ``reference`` (asset type)."""
        params = {"symbols": ",".join(s.upper() for s in symbols), "fields": fields}
        return self._get("/quotes", params=params)

    def get_quotes_chunked(
        self,
        symbols: List[str],
        fields: str = "quote,fundamental,reference",
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fetch quotes for many symbols, splitting into rate-limited batches.

        Merges the per-batch payloads into one dict keyed by symbol. Used by the
        Scout to quote a 200+ symbol watchlist without exceeding request limits.
        """
        size = batch_size or settings.schwab_quote_batch_size
        merged: Dict[str, Any] = {}
        for i in range(0, len(symbols), size):
            chunk = symbols[i : i + size]
            merged.update(self.get_quotes(chunk, fields=fields))
        return merged

    def get_quote(self, symbol: str, fields: str = "quote,fundamental,reference") -> Dict[str, Any]:
        """Single-symbol quote (Schwab path is /{symbol}/quotes)."""
        return self._get(f"/{symbol.upper()}/quotes", params={"fields": fields})

    def get_price_history(
        self,
        symbol: str,
        period_type: str = "year",
        period: int = 1,
        frequency_type: str = "daily",
        frequency: int = 1,
        need_extended_hours_data: bool = False,
        need_previous_close: bool = True,
        start_date: Optional[int] = None,
        end_date: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Historical candles. Defaults give ~1 year of DAILY candles — enough
        for the 200-SMA the Quant node needs."""
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "periodType": period_type,
            "period": period,
            "frequencyType": frequency_type,
            "frequency": frequency,
            "needExtendedHoursData": str(need_extended_hours_data).lower(),
            "needPreviousClose": str(need_previous_close).lower(),
        }
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
        return self._get("/pricehistory", params=params)

    def get_option_chain(
        self,
        symbol: str,
        contract_type: str = "CALL",
        strike_count: int = 15,
        include_quotes: bool = True,
        strategy: str = "SINGLE",
        range_filter: str = "OTM",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        strike: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Option chain. Defaults target OTM CALLs (covered-call selling)."""
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "contractType": contract_type,
            "strikeCount": strike_count,
            "includeQuotes": str(include_quotes).lower(),
            "strategy": strategy,
            "range": range_filter,
        }
        if from_date:
            params["fromDate"] = from_date
        if to_date:
            params["toDate"] = to_date
        if strike is not None:
            params["strike"] = strike
        return self._get("/chains", params=params)

    def get_option_expirations(self, symbol: str) -> Dict[str, Any]:
        """Expiration timeline for an optionable equity."""
        return self._get("/expirationchain", params={"symbol": symbol.upper()})

    def lookup_instrument(self, symbol: str, projection: str = "symbol-search") -> Dict[str, Any]:
        """Instrument/fundamental lookup by symbol pattern."""
        return self._get("/instruments", params={"symbol": symbol.upper(), "projection": projection})

    # ── normalizing helpers (so nodes don't parse raw JSON everywhere) ─
    @staticmethod
    def extract_fundamentals(quotes_payload: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Pull the Scout's filter inputs out of a /quotes payload for one symbol.

        Returns price, total + average daily volume, and dividend yield. Missing
        fields default to 0.0 so downstream filters fail safe (i.e. get rejected
        rather than crash).
        """
        sym = symbol.upper()
        entry = quotes_payload.get(sym, {}) if isinstance(quotes_payload, dict) else {}
        fund = entry.get("fundamental", {}) or {}
        quote = entry.get("quote", {}) or {}
        ref = entry.get("reference", {}) or {}

        # Average daily volume: prefer 10-day, fall back to 1-year.
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

    def is_optionable(self, symbol: str) -> bool:
        """True if the symbol has any listed option expirations.

        Schwab quotes don't carry a clean 'optionable' flag, so we probe the
        expiration chain — empty/erroring => treat as not optionable.
        """
        try:
            data = self.get_option_expirations(symbol)
        except httpx.HTTPError as exc:
            logger.warning("Optionable check failed for %s: %s", symbol, exc)
            return False
        expirations = data.get("expirationList") or data.get("expirations") or []
        return bool(expirations)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SchwabClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
