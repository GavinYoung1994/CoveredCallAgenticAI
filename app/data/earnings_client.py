"""Finnhub earnings-calendar client — feeds the News node's earnings guardrail.

The design's hard rule: never sell a covered call that expires on/after an
upcoming earnings report (earnings gaps can wipe out the position). To enforce
it we need each stock's next earnings date, which neither Schwab market-data nor
a news API reliably provides — hence this dedicated calendar source.

Graceful degradation: if no API key is configured (or the call fails), this
returns ``None`` = "earnings unknown", which the News node treats as
flag-but-allow rather than a crash.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.data.rate_limiter import RateLimiter

logger = logging.getLogger("earnings-client")


class EarningsClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        path: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.finnhub_api_key
        self._base_url = (base_url or settings.earnings_api_base_url).rstrip("/")
        self._path = path or settings.earnings_path
        self._client = http_client or httpx.Client(timeout=30.0)
        self._limiter = rate_limiter or RateLimiter(
            settings.earnings_rate_limit_calls,
            settings.earnings_rate_limit_period_sec,
            name="finnhub",
        )

    @property
    def enabled(self) -> bool:
        """False when no API key is configured (earnings → unknown)."""
        return bool(self._api_key) and not self._api_key.startswith("your-")

    def get_earnings_calendar(
        self, symbol: str, from_date: str, to_date: str
    ) -> List[Dict[str, Any]]:
        """Raw earnings-calendar rows for a symbol in [from_date, to_date]."""
        if not self.enabled:
            return []
        self._limiter.acquire()
        params = {
            "symbol": symbol.upper(),
            "from": from_date,
            "to": to_date,
            "token": self._api_key,
        }
        url = f"{self._base_url}{self._path}"
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json() or {}
        return data.get("earningsCalendar", []) or []

    def get_next_earnings_date(
        self, symbol: str, from_date: str, to_date: str
    ) -> Optional[str]:
        """Earliest earnings date (YYYY-MM-DD) in the window, or None if unknown.

        None means either no key configured, no scheduled earnings in range, or
        an API error — all of which the caller treats as 'earnings unknown'.
        """
        try:
            rows = self.get_earnings_calendar(symbol, from_date, to_date)
        except httpx.HTTPError as exc:
            logger.warning("Earnings lookup failed for %s: %s", symbol, exc)
            return None
        dates = sorted(r.get("date") for r in rows if r.get("date"))
        return dates[0] if dates else None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EarningsClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
